# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.chat_template import apply_chat_template_for_generation
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.torch_dtypes import to_py


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        guidance_config = self.config.data.get("guidance", {})
        self.guidance_enabled = bool(guidance_config.get("enabled", False))
        self.guidance_test_enabled = bool(guidance_config.get("test_enabled", False))
        self.guidance_injection_mode = guidance_config.get("injection_mode", "user_prompt")
        self.guidance_fixed_k = int(guidance_config.get("fixed_k", -1))
        # Fixed-ratio guidance: per-sample k = ceil(len(all_actions) * ratio).
        # Disabled when ratio <= 0. Mutually exclusive with fixed_k>0 and adaptive.
        self.guidance_fixed_ratio = float(guidance_config.get("fixed_ratio", -1.0))

        # Backward compat: also check old flat key
        if not self.guidance_enabled:
            old_ag = self.config.data.get("adaptive_guidance", {})
            if old_ag.get("enabled", False):
                self.guidance_enabled = True
                self.guidance_injection_mode = self.config.data.get("guidance_injection_mode", "user_prompt")

        self.adaptive_guidance = None
        adaptive_config = guidance_config.get("adaptive", {})
        if self.guidance_enabled and adaptive_config.get("enabled", False):
            from verl.trainer.ppo.adaptive_guidance import AdaptiveGuidanceSearcher

            self.adaptive_guidance = AdaptiveGuidanceSearcher(
                **adaptive_config, injection_mode=self.guidance_injection_mode
            )

        # Mutual exclusivity: at most one of {adaptive, fixed_k>0, fixed_ratio>0}.
        # All three off means "use all actions" (full guidance, backward-compat default).
        if self.guidance_enabled:
            _active_modes = []
            if self.adaptive_guidance is not None:
                _active_modes.append("adaptive")
            if self.guidance_fixed_k is not None and self.guidance_fixed_k > 0:
                _active_modes.append(f"fixed_k={self.guidance_fixed_k}")
            if self.guidance_fixed_ratio > 0:
                _active_modes.append(f"fixed_ratio={self.guidance_fixed_ratio}")
            if len(_active_modes) > 1:
                raise ValueError(
                    "data.guidance options {adaptive.enabled, fixed_k>0, fixed_ratio>0} "
                    f"are mutually exclusive, got: {_active_modes}. "
                    "Enable at most one at a time."
                )
            if self.guidance_fixed_ratio > 0 and self.guidance_fixed_ratio > 1.0:
                raise ValueError(
                    f"data.guidance.fixed_ratio must be in (0, 1], got {self.guidance_fixed_ratio}"
                )

        gop_config = guidance_config.get("guided_offpolicy", {})
        self.guided_offpolicy_enabled = bool(gop_config.get("enabled", False))
        self.guided_offpolicy_mode = gop_config.get("mode", "single")
        self.guided_offpolicy_onpolicy_coef = float(gop_config.get("onpolicy_coef", 1.0))
        self.guided_offpolicy_offpolicy_coef = float(gop_config.get("offpolicy_coef", 0.1))
        self.guided_offpolicy_onpolicy_clip_ratio = float(gop_config.get("onpolicy_clip_ratio", -1))
        self.guided_offpolicy_onpolicy_clip_ratio_c = float(gop_config.get("onpolicy_clip_ratio_c", -1))
        self.guided_offpolicy_offpolicy_clip_ratio = float(gop_config.get("offpolicy_clip_ratio", -1))
        self.guided_offpolicy_offpolicy_clip_ratio_c = float(gop_config.get("offpolicy_clip_ratio_c", -1))
        self.guided_offpolicy_luffy_gamma = float(gop_config.get("luffy_gamma", 0.0))
        self.guided_offpolicy_hard_threshold = float(gop_config.get("hard_threshold", 0.0))
        self.guided_offpolicy_adaptive_success_threshold = float(
            gop_config.get("adaptive_success_threshold", self.guided_offpolicy_hard_threshold)
        )
        if self.guided_offpolicy_enabled and not self.guidance_enabled:
            raise ValueError(
                "guided_offpolicy requires guidance.enabled=True "
                "(need guided rollouts as behavior policy)."
            )

        if self.guided_offpolicy_enabled and self.guided_offpolicy_mode == "selective":
            if self.config.reward_model.get("launch_reward_fn_async", False):
                raise ValueError(
                    "guided_offpolicy.mode='selective' requires synchronous "
                    "reward computation (reward_model.launch_reward_fn_async "
                    "must be False). Selective phase needs reward_tensor "
                    "immediately to identify hard groups."
                )

        distill_config = self.config.data.get("on_policy_distillation", {})
        opd_features = [
            ("on_policy_distillation", distill_config.get("enabled", False)),
            ("guided_offpolicy", self.guided_offpolicy_enabled),
        ]
        active = [name for name, enabled in opd_features if enabled]
        if len(active) > 1:
            raise ValueError(
                f"Only one OPD-style / off-policy feature can be enabled at a time. "
                f"Currently active: {active}. They modify the PPO ratio semantics "
                f"or write teacher_log_probs and would conflict. "
                f"Please disable all but one."
            )
        self._unguided_tool_schemas = None
        self._apply_chat_template_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        multi_turn_cfg = self.config.actor_rollout_ref.rollout.get("multi_turn", {})
        if multi_turn_cfg.get("pass_tool_schema_to_template", True):
            tool_config_path = multi_turn_cfg.get("tool_config_path", None)
            if tool_config_path:
                from omegaconf import OmegaConf as _OmegaConf
                _tools_yaml = _OmegaConf.load(tool_config_path)
                self._unguided_tool_schemas = [
                    _OmegaConf.to_container(t.tool_schema, resolve=True)
                    for t in _tools_yaml.tools
                ]

        self.distillation_config = None
        if distill_config.get("enabled", False):
            self.distillation_config = distill_config
            self.distill_coef = float(distill_config.get("distill_coef", 0.1))
            self.distill_coef_min = float(distill_config.get("distill_coef_min", 0.0))
            self.distill_coef_decay = float(distill_config.get("distill_coef_decay", 1.0))
            self.policy_loss_coef = float(distill_config.get("policy_loss_coef", 1.0))
            self.reward_gate = bool(distill_config.get("reward_gate", False))
            self.reward_gate_margin = float(distill_config.get("reward_gate_margin", 0.0))
            self.teacher_rollout_n = int(distill_config.get("teacher_rollout_n", 1))

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(to_py(entry), ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _resolve_guidance_fixed_k(self, n_actions: int) -> int:
        """Return the number of prefix actions to inject as guidance for a sample.

        Resolution order (only one of adaptive/fixed_k/fixed_ratio may be active
        per __init__ validation; adaptive is handled elsewhere):
          - fixed_ratio > 0: k = ceil(n_actions * fixed_ratio), clamped to [0, n]
          - fixed_k >= 0: k = min(fixed_k, n_actions)
          - otherwise: k = n_actions (full guidance fallback)
        """
        if n_actions <= 0:
            return 0
        if self.guidance_fixed_ratio > 0:
            k = int(math.ceil(n_actions * self.guidance_fixed_ratio))
            return max(0, min(k, n_actions))
        if self.guidance_fixed_k < 0:
            return n_actions
        return min(self.guidance_fixed_k, n_actions)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _construct_teacher_batch(self, batch: DataProto) -> DataProto:
        """Build a DataProto for the teacher forward pass with guided prompts.

        Takes student rollout results (with unguided prompts) and constructs a
        teacher batch where the prompt is replaced with a guided version (containing
        action trajectory hints) while keeping the same response tokens.

        The resulting batch can be passed to compute_log_prob to get teacher
        log-probabilities under torch.no_grad().
        """
        from verl.trainer.ppo.adaptive_guidance import (
            _parse_all_actions,
            rebuild_raw_prompt_with_k,
        )
        from verl.utils.model import compute_position_id_with_mask

        N = len(batch)
        responses = batch.batch["responses"]  # (N, response_length)
        response_length = responses.shape[-1]
        # Response attention mask: last response_length cols of the full attention_mask
        response_attn_mask = batch.batch["attention_mask"][:, -response_length:]

        extra_infos = batch.non_tensor_batch["extra_info"]
        unguided_raw_prompts = batch.non_tensor_batch.get("unguided_raw_prompt")
        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)

        teacher_prompt_ids_list = []
        teacher_prompt_mask_list = []

        for i in range(N):
            info = extra_infos[i]
            all_actions = _parse_all_actions(info.get("all_actions"))
            base_question = info.get("base_question", info.get("question", ""))
            k = len(all_actions)

            # 优先使用 unguided prompt 作为基础，避免 guidance 被双重注入，
            # 同时确保 k=0 时 teacher 使用的是真正的 unguided prompt
            base_prompt = unguided_raw_prompts[i] if unguided_raw_prompts is not None else None
            if base_prompt is None and raw_prompts is not None:
                base_prompt = raw_prompts[i]

            if base_prompt is not None and k > 0:
                guided_raw_prompt = rebuild_raw_prompt_with_k(
                    base_prompt,
                    base_question,
                    all_actions,
                    k,
                    injection_mode=self.guidance_injection_mode,
                    prompt_style=info.get("guidance_prompt_style", "default"),
                )
            elif base_prompt is not None:
                guided_raw_prompt = base_prompt
            else:
                guided_raw_prompt = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": base_question},
                ]

            prompt_ids = apply_chat_template_for_generation(
                self.tokenizer,
                guided_raw_prompt,
                return_tensors="pt",
            )
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            # prompt_ids: (1, prompt_len)
            teacher_prompt_ids_list.append(prompt_ids.squeeze(0))
            teacher_prompt_mask_list.append(torch.ones(prompt_ids.shape[-1], dtype=torch.long))

        # Left-pad teacher prompts to max teacher prompt length
        max_teacher_prompt_len = max(ids.shape[0] for ids in teacher_prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_prompt_ids = torch.full((N, max_teacher_prompt_len), pad_id, dtype=torch.long)
        padded_prompt_mask = torch.zeros((N, max_teacher_prompt_len), dtype=torch.long)

        for i, (ids, mask) in enumerate(zip(teacher_prompt_ids_list, teacher_prompt_mask_list)):
            prompt_len = ids.shape[0]
            offset = max_teacher_prompt_len - prompt_len  # left-pad offset
            padded_prompt_ids[i, offset:] = ids
            padded_prompt_mask[i, offset:] = mask

        # Concatenate: [left-padded teacher prompt | response (same as student)]
        teacher_input_ids = torch.cat([padded_prompt_ids, responses], dim=-1)
        teacher_attention_mask = torch.cat([padded_prompt_mask, response_attn_mask], dim=-1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        teacher_batch = DataProto.from_dict(
            tensors={
                "input_ids": teacher_input_ids,
                "attention_mask": teacher_attention_mask,
                "position_ids": teacher_position_ids,
                "responses": responses,
            },
        )
        teacher_batch.meta_info["temperature"] = batch.meta_info.get("temperature", 1.0)

        return teacher_batch

    @staticmethod
    def _count_message_images(messages) -> int:
        """Count image references in a list of chat messages."""
        count = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                count += sum(
                    1 for item in content
                    if isinstance(item, dict) and item.get("type") == "image"
                )
            elif isinstance(content, str):
                count += content.count("<image>")
        return count

    def _encode_unguided_prompt(self, messages, all_images):
        """Encode a full unguided conversation with a single processor call.

        Returns (prompt_ids, attention_mask, multi_modal_inputs_dict_or_None).
        For VLM: prompt_ids and multi_modal_inputs come from the same processor
        call, so image_grid_thw is guaranteed consistent with vision tokens.
        """
        from verl.utils.chat_template import apply_chat_template_for_generation

        _tools = self._unguided_tool_schemas
        _kwargs = self._apply_chat_template_kwargs
        if self.processor is not None:
            prompt_text = apply_chat_template_for_generation(
                self.processor, messages, tools=_tools, tokenize=False, **_kwargs
            )
            model_inputs = self.processor(
                text=[prompt_text],
                images=all_images if all_images else None,
                return_tensors="pt",
            )
            ids = model_inputs["input_ids"].squeeze(0)
            attn = model_inputs["attention_mask"].squeeze(0).long()
            mmi = dict(model_inputs)
            mmi.pop("input_ids", None)
            mmi.pop("attention_mask", None)
            return ids, attn, mmi

        ids = apply_chat_template_for_generation(
            self.tokenizer, messages, tools=_tools, return_tensors="pt", **_kwargs
        )
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.squeeze(0), torch.ones(ids.shape[-1], dtype=torch.long), None

    def _construct_unguided_batch(self, batch: DataProto) -> DataProto:
        """Build a DataProto with unguided prompts + same responses.

        Only the initial prompt (system + user) is re-encoded with the
        unguided version.  The multi-turn content (assistant outputs, tool
        responses) lives entirely in ``batch.batch["responses"]`` and is
        concatenated unchanged.

        For VLM, after constructing the full ``new_input_ids``, we compute
        multi-modal metadata via decode-then-reencode of the whole attended
        sequence with ALL images — the same pattern used by
        ``_agent_loop_postprocess``.  This guarantees ``image_grid_thw`` is
        consistent with the vision tokens in ``new_input_ids`` without any
        cross-encode metadata merging.
        """
        from verl.utils.model import compute_position_id_with_mask

        N = len(batch)
        responses = batch.batch["responses"]
        response_length = responses.shape[-1]
        response_attn_mask = batch.batch["attention_mask"][:, -response_length:]

        extra_infos = batch.non_tensor_batch["extra_info"]
        unguided_raw_prompts = batch.non_tensor_batch.get("unguided_raw_prompt")
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")

        prompt_ids_list = []
        prompt_mask_list = []

        for i in range(N):
            info = extra_infos[i]

            # Build the unguided initial messages (system + user, no guidance)
            unguided_initial = None
            if unguided_raw_prompts is not None:
                unguided_initial = unguided_raw_prompts[i]
            if unguided_initial is None:
                unguided_initial = info.get("unguided_raw_prompt")
            if unguided_initial is None:
                base_q = info.get("base_question", info.get("question", ""))
                system_content = ""
                if raw_prompts is not None and raw_prompts[i]:
                    system_content = raw_prompts[i][0].get("content", "")
                unguided_initial = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": base_q},
                ]

            # Count image references in the initial messages so we only
            # pass the prompt images to the encoder (not tool-response images).
            # Images are ordered [prompt_img_0, ..., tool_img_0, ...] in
            # multi_modal_data, so slicing [:n_prompt_images] is correct.
            n_prompt_images = self._count_message_images(unguided_initial)
            prompt_images = None
            if "multi_modal_data" in batch.non_tensor_batch:
                md = batch.non_tensor_batch["multi_modal_data"][i]
                if isinstance(md, dict):
                    imgs = md.get("image")
                    if imgs and n_prompt_images > 0:
                        prompt_images = imgs[:n_prompt_images]

            # Encode ONLY the initial unguided prompt (not the full conversation).
            # For VLM, this produces correct vision tokens in prompt_ids.
            # We discard the per-prompt metadata here; VLM metadata is
            # recomputed below from the full sequence via decode→reencode.
            prompt_ids, prompt_attn, _ = self._encode_unguided_prompt(
                unguided_initial, prompt_images
            )

            prompt_ids_list.append(prompt_ids)
            prompt_mask_list.append(prompt_attn)

        max_prompt_len = max(t.shape[0] for t in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_ids = torch.full((N, max_prompt_len), pad_id, dtype=torch.long)
        padded_mask = torch.zeros((N, max_prompt_len), dtype=torch.long)

        for i, (ids, mask) in enumerate(zip(prompt_ids_list, prompt_mask_list)):
            plen = ids.shape[0]
            offset = max_prompt_len - plen
            padded_ids[i, offset:] = ids
            padded_mask[i, offset:] = mask

        new_input_ids = torch.cat([padded_ids, responses], dim=-1)
        new_attention_mask = torch.cat([padded_mask, response_attn_mask], dim=-1)

        # --- VLM metadata for unguided batch ---
        # Prefer reusing the guided rollout's multi_modal_inputs (pixel_values,
        # image_grid_thw, …) directly.  They were computed by
        # _agent_loop_postprocess with the exact same processor instance and
        # the exact same images that produced the <|image_pad|> tokens in
        # ``responses``, so image_grid_thw is guaranteed to match the vision
        # token counts in new_input_ids.
        #
        # Fallback: if multi_modal_inputs is unavailable (e.g. sync rollout
        # path), recompute from multi_modal_data via decode→reencode.
        has_vlm_content = self.processor is not None and (
            "multi_modal_inputs" in batch.non_tensor_batch
            or "multi_modal_data" in batch.non_tensor_batch
        )
        if has_vlm_content:
            guided_mmis = batch.non_tensor_batch.get("multi_modal_inputs")
            multi_modal_datas = batch.non_tensor_batch.get("multi_modal_data")
            unguided_multi_modal_inputs = []
            has_any_mmi = False
            for i in range(N):
                src_mmi = guided_mmis[i] if guided_mmis is not None else None
                if src_mmi is not None and isinstance(src_mmi, dict):
                    mmi = {k: v for k, v in src_mmi.items()
                           if k not in ("input_ids", "attention_mask")}
                    unguided_multi_modal_inputs.append(mmi)
                    has_any_mmi = True
                else:
                    md = multi_modal_datas[i] if multi_modal_datas is not None else None
                    all_images = md.get("image") if isinstance(md, dict) else None
                    if all_images:
                        attended = new_input_ids[i][new_attention_mask[i].bool()]
                        decoded_text = self.tokenizer.decode(
                            attended, skip_special_tokens=True
                        )
                        mmi = self.processor(
                            text=[decoded_text],
                            images=all_images,
                            return_tensors="pt",
                        )
                        mmi = dict(mmi)
                        mmi.pop("input_ids", None)
                        mmi.pop("attention_mask", None)
                        unguided_multi_modal_inputs.append(mmi)
                        has_any_mmi = True
                    else:
                        unguided_multi_modal_inputs.append(None)

            if has_any_mmi:
                position_ids_list = []
                for i in range(N):
                    mmi = unguided_multi_modal_inputs[i] or {}
                    image_grid_thw = mmi.get("image_grid_thw")
                    video_grid_thw = mmi.get("video_grid_thw")
                    second_per_grid_ts = mmi.get("second_per_grid_ts")

                    if hasattr(self.processor, "image_processor") and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                            from verl.models.transformers.qwen3_vl import get_rope_index
                        else:
                            from verl.models.transformers.qwen2_vl import get_rope_index

                        valid_ids = new_input_ids[i][new_attention_mask[i].bool()]
                        n_vision_starts = (valid_ids == self.processor.vision_start_token_id).sum().item()
                        n_grid_entries = (
                            (len(image_grid_thw) if image_grid_thw is not None else 0)
                            + (len(video_grid_thw) if video_grid_thw is not None else 0)
                        )
                        if n_vision_starts != n_grid_entries:
                            logger.warning(
                                "Unguided VLM sanity check failed for sample %d: "
                                "%d <vision_start> tokens in input_ids but %d "
                                "entries in image/video_grid_thw. Position IDs "
                                "may be incorrect.",
                                i, n_vision_starts, n_grid_entries,
                            )

                        vision_pos = get_rope_index(
                            self.processor,
                            input_ids=new_input_ids[i],
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            second_per_grid_ts=second_per_grid_ts,
                            attention_mask=new_attention_mask[i],
                        )  # (3, seq_len)
                        valid_mask = new_attention_mask[i].bool()
                        text_pos = torch.ones((1, new_input_ids.shape[1]), dtype=torch.long)
                        text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item())
                        pos = torch.cat((text_pos, vision_pos), dim=0)  # (4, seq_len)
                        position_ids_list.append(pos)
                    else:
                        pos = compute_position_id_with_mask(new_attention_mask[i:i+1]).squeeze(0)
                        position_ids_list.append(pos)

                new_position_ids = torch.stack(position_ids_list, dim=0)
                non_tensors = {
                    "multi_modal_inputs": np.array(unguided_multi_modal_inputs, dtype=object)
                }
                return DataProto.from_dict(
                    tensors={
                        "input_ids": new_input_ids,
                        "attention_mask": new_attention_mask,
                        "position_ids": new_position_ids,
                    },
                    non_tensors=non_tensors,
                )

        # Non-VLM or no multi-modal content
        new_position_ids = compute_position_id_with_mask(new_attention_mask)
        return DataProto.from_dict(
            tensors={
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "position_ids": new_position_ids,
            },
        )

    def _reconstruct_batch_unguided(self, batch: DataProto) -> None:
        """Replace batch's input_ids/attention_mask/position_ids with unguided prompts in-place.

        Used in single off-policy mode where unguided is the only branch.
        """
        ub = self._construct_unguided_batch(batch)
        batch.batch["input_ids"] = ub.batch["input_ids"]
        batch.batch["attention_mask"] = ub.batch["attention_mask"]
        batch.batch["position_ids"] = ub.batch["position_ids"]
        if "multi_modal_inputs" in ub.non_tensor_batch:
            batch.non_tensor_batch["multi_modal_inputs"] = ub.non_tensor_batch["multi_modal_inputs"]

    def _apply_guidance_to_gen_batch(self, gen_batch: DataProto) -> DataProto:
        """Create a deep copy of gen_batch with full action guidance applied to raw_prompt.

        Used for reward-gated OPD: run a guided teacher rollout to decide
        which samples are worth distilling.
        """
        from verl.trainer.ppo.adaptive_guidance import (
            _parse_all_actions,
            rebuild_raw_prompt_with_k,
        )

        guided = deepcopy(gen_batch)
        extra_infos = guided.non_tensor_batch["extra_info"]
        for i in range(len(guided)):
            info = extra_infos[i]
            all_actions = _parse_all_actions(info.get("all_actions"))
            if not all_actions:
                continue
            base_question = info.get("base_question", info.get("question", ""))
            guided.non_tensor_batch["raw_prompt"][i] = rebuild_raw_prompt_with_k(
                guided.non_tensor_batch["raw_prompt"][i],
                base_question,
                all_actions,
                k=len(all_actions),
                injection_mode=self.guidance_injection_mode,
                prompt_style=info.get("guidance_prompt_style", "default"),
            )
        return guided

    def _collect_group_reward_stats(self, batch: DataProto, reward_tensor: torch.Tensor) -> dict:
        """Collect reward coverage stats at the query-group level."""
        seq_reward = reward_tensor.sum(dim=-1)
        uids = batch.non_tensor_batch["uid"]
        unique_uids = list(dict.fromkeys(uids))
        uid_to_batch_idxs: dict[str, list[int]] = {}
        for i, uid in enumerate(uids):
            uid_to_batch_idxs.setdefault(uid, []).append(i)

        group_max_reward: dict[str, float] = {}
        reward_uids = []
        zero_reward_uids = []
        hard_uids = []
        hard_threshold = self.guided_offpolicy_hard_threshold

        for uid in unique_uids:
            group_max = max(seq_reward[j].item() for j in uid_to_batch_idxs[uid])
            group_max_reward[uid] = group_max
            if group_max > 0:
                reward_uids.append(uid)
            else:
                zero_reward_uids.append(uid)
            if group_max <= hard_threshold:
                hard_uids.append(uid)

        return {
            "seq_reward": seq_reward,
            "uids": uids,
            "unique_uids": unique_uids,
            "uid_to_batch_idxs": uid_to_batch_idxs,
            "group_max_reward": group_max_reward,
            "reward_uids": reward_uids,
            "zero_reward_uids": zero_reward_uids,
            "hard_uids": hard_uids,
            "prompt_count": len(unique_uids),
            "hard_threshold": hard_threshold,
        }

    @staticmethod
    def _log_group_reward_metrics(metrics: dict, group_stats: dict, prefix: str) -> None:
        """Log group-level reward and hard-group ratios."""
        prompt_count = group_stats["prompt_count"]
        n_reward = len(group_stats["reward_uids"])
        n_zero = len(group_stats["zero_reward_uids"])
        n_hard = len(group_stats["hard_uids"])

        metrics[f"{prefix}/n_groups"] = prompt_count
        metrics[f"{prefix}/n_reward_groups"] = n_reward
        metrics[f"{prefix}/frac_reward_groups"] = n_reward / prompt_count if prompt_count > 0 else 0.0
        metrics[f"{prefix}/n_zero_reward_groups"] = n_zero
        metrics[f"{prefix}/frac_zero_reward_groups"] = n_zero / prompt_count if prompt_count > 0 else 0.0
        metrics[f"{prefix}/n_hard_groups"] = n_hard
        metrics[f"{prefix}/frac_hard_groups"] = n_hard / prompt_count if prompt_count > 0 else 0.0
        metrics[f"{prefix}/hard_threshold"] = group_stats["hard_threshold"]

    def _selective_guidance_phase(
        self, batch, gen_batch, reward_tensor, reward_extra_infos_dict,
        metrics, timing_raw,
    ):
        """Selective guidance: 仅对困难组做 guided rollout。

        困难组判定：组内所有 n 条 rollout 的最大 reward <= hard_threshold。
        困难组的数据在 batch 中被 guided 版本替换，使后续 off-policy 分支
        能训练 unguided 能力；简单组数据不变，保持 on-policy 训练。

        Returns updated (reward_tensor, reward_extra_infos_dict).
        """
        import torch.nn.functional as F
        from verl.trainer.ppo.reward import compute_reward

        student_n = self.config.actor_rollout_ref.rollout.n
        pre_group_stats = self._collect_group_reward_stats(batch, reward_tensor)
        uid_to_batch_idxs = pre_group_stats["uid_to_batch_idxs"]
        prompt_count = pre_group_stats["prompt_count"]
        hard_threshold = pre_group_stats["hard_threshold"]
        hard_uids = pre_group_stats["hard_uids"]
        pre_reward_uids = set(pre_group_stats["reward_uids"])
        n_hard = len(hard_uids)
        metrics["selective_guidance/n_hard_groups"] = n_hard
        metrics["selective_guidance/n_easy_groups"] = prompt_count - n_hard
        metrics["selective_guidance/frac_hard"] = n_hard / prompt_count if prompt_count > 0 else 0.0
        metrics["selective_guidance/hard_threshold"] = hard_threshold
        metrics["selective_guidance/adaptive_success_threshold"] = self.guided_offpolicy_adaptive_success_threshold

        if n_hard == 0:
            return reward_tensor, reward_extra_infos_dict

        generate_fn = (
            self.async_rollout_manager.generate_sequences
            if self.async_rollout_mode
            else self.actor_rollout_wg.generate_sequences
        )

        # 映射 hard_uids → gen_batch 索引（gen_batch 是 pre-repeat 的）
        gen_uids = gen_batch.non_tensor_batch["uid"]
        gen_uid_to_idx = {uid: i for i, uid in enumerate(gen_uids)}
        hard_gen_idxs = []
        hard_uids_valid = []
        for uid in hard_uids:
            if uid in gen_uid_to_idx:
                hard_gen_idxs.append(gen_uid_to_idx[uid])
                hard_uids_valid.append(uid)
        hard_gen_idxs = np.array(hard_gen_idxs)
        hard_uids = hard_uids_valid
        if len(hard_gen_idxs) == 0:
            return reward_tensor, reward_extra_infos_dict

        hard_gen = gen_batch.select_idxs(hard_gen_idxs)
        n_hard_samples = len(hard_gen)

        # worker 对齐辅助
        agent_cfg = self.config.actor_rollout_ref.rollout.get("agent", None)
        n_workers = int(agent_cfg.get("num_workers", len(gen_batch))) if agent_cfg else len(gen_batch)

        def _pad_to_workers(proto, n_actual):
            rem = n_actual % n_workers
            if rem == 0:
                return proto, 0
            pad = n_workers - rem
            idxs = list(range(n_actual)) + [0] * pad
            return proto.select_idxs(np.array(idxs)), pad

        # --- Guidance injection for hard groups ---
        if self.adaptive_guidance is not None:
            # Adaptive search: binary-search for optimal k per sample
            hard_gen_padded, search_pad = _pad_to_workers(hard_gen, n_hard_samples)
            with marked_timer("selective_search", timing_raw, color="cyan"):
                optimal_k = self.adaptive_guidance.search(
                    hard_gen_padded,
                    batch,
                    generate_fn,
                    self.reward_fn,
                    reward_success_threshold=self.guided_offpolicy_adaptive_success_threshold,
                )
            optimal_k = optimal_k[:n_hard_samples]
            self.adaptive_guidance.apply(hard_gen, batch, optimal_k)
            metrics["selective_guidance/mean_k"] = float(np.mean(optimal_k))
            metrics["selective_guidance/max_k"] = float(np.max(optimal_k))
            metrics["selective_guidance/min_k"] = float(np.min(optimal_k))
        else:
            # Full guidance: inject all actions without adaptive search
            from verl.trainer.ppo.adaptive_guidance import (
                _parse_all_actions,
                rebuild_raw_prompt_with_k,
            )
            extra_infos = hard_gen.non_tensor_batch["extra_info"]
            for i in range(n_hard_samples):
                info = extra_infos[i]
                all_actions = _parse_all_actions(info.get("all_actions"))
                if not all_actions:
                    continue
                k = self._resolve_guidance_fixed_k(len(all_actions))
                base_question = info.get("base_question", info.get("question", ""))
                hard_gen.non_tensor_batch["raw_prompt"][i] = rebuild_raw_prompt_with_k(
                    hard_gen.non_tensor_batch["raw_prompt"][i],
                    base_question,
                    all_actions,
                    k,
                    injection_mode=self.guidance_injection_mode,
                    prompt_style=info.get("guidance_prompt_style", "default"),
                )
            metrics["selective_guidance/mean_k"] = float(k) if n_hard_samples > 0 else 0.0
            metrics["selective_guidance/mode"] = "full_guidance"

        # --- Guided rollout（repeat n → pad → rollout → unpad）---
        hard_gen_repeated = hard_gen.repeat(repeat_times=student_n, interleave=True)
        n_guided_total = len(hard_gen_repeated)
        hard_rollout_in, rollout_pad = _pad_to_workers(hard_gen_repeated, n_guided_total)

        with marked_timer("selective_rollout", timing_raw, color="red"):
            guided_output = generate_fn(hard_rollout_in)
        if rollout_pad > 0:
            guided_output = guided_output.select_idxs(np.arange(n_guided_total))

        # 补齐 reward 元数据
        for key in ("reward_model", "data_source", "extra_info"):
            if key in hard_gen.non_tensor_batch:
                guided_output.non_tensor_batch[key] = np.repeat(
                    hard_gen.non_tensor_batch[key], student_n, axis=0
                )

        # response_mask
        if "response_mask" not in guided_output.batch:
            guided_output.batch["response_mask"] = compute_response_mask(guided_output)

        # --- Compute guided reward ---
        with marked_timer("selective_reward", timing_raw, color="yellow"):
            guided_reward, guided_extra = compute_reward(guided_output, self.reward_fn)
        guided_seq = guided_reward.sum(dim=-1)
        metrics["selective_guidance/guided_reward_mean"] = guided_seq.mean().item()
        metrics["selective_guidance/guided_reward_max"] = guided_seq.max().item()

        # --- 替换 batch 中困难组的数据 ---
        hard_batch_idxs = []
        for uid in hard_uids:
            hard_batch_idxs.extend(uid_to_batch_idxs[uid])
        hard_batch_idxs = np.array(hard_batch_idxs)

        LEFT_PAD_KEYS = {"input_ids", "attention_mask", "position_ids"}
        batch_tensor_keys = set(batch.batch.keys())

        for key in list(guided_output.batch.keys()):
            if key not in batch_tensor_keys:
                continue
            src = guided_output.batch[key]
            dst = batch.batch[key]
            if src.dim() >= 2 and src.shape[1] != dst.shape[1]:
                max_len = max(src.shape[1], dst.shape[1])
                left = key in LEFT_PAD_KEYS
                if dst.shape[1] < max_len:
                    p = max_len - dst.shape[1]
                    batch.batch[key] = F.pad(dst, (p, 0) if left else (0, p), value=0)
                    dst = batch.batch[key]
                if src.shape[1] < max_len:
                    p = max_len - src.shape[1]
                    src = F.pad(src, (p, 0) if left else (0, p), value=0)
            dst[hard_batch_idxs] = src

        batch_non_tensor_keys = set(batch.non_tensor_batch.keys())
        for key in list(guided_output.non_tensor_batch.keys()):
            if key not in batch_non_tensor_keys:
                continue
            for i, idx in enumerate(hard_batch_idxs):
                batch.non_tensor_batch[key][idx] = guided_output.non_tensor_batch[key][i]

        # 替换 reward_tensor 中困难组的值
        if guided_reward.shape[1] != reward_tensor.shape[1]:
            max_rlen = max(guided_reward.shape[1], reward_tensor.shape[1])
            if reward_tensor.shape[1] < max_rlen:
                reward_tensor = F.pad(
                    reward_tensor, (0, max_rlen - reward_tensor.shape[1]), value=0
                )
            if guided_reward.shape[1] < max_rlen:
                guided_reward = F.pad(
                    guided_reward, (0, max_rlen - guided_reward.shape[1]), value=0
                )
        reward_tensor[hard_batch_idxs] = guided_reward

        # 替换 reward_extra_infos_dict 中困难组的值
        if guided_extra and reward_extra_infos_dict:
            for info_key, info_vals in guided_extra.items():
                if info_key in reward_extra_infos_dict:
                    for i, idx in enumerate(hard_batch_idxs):
                        reward_extra_infos_dict[info_key][idx] = info_vals[i]

        # per-sample off-policy 标记，供 dp_actor 做 LUFFY ratio shaping
        batch_size = len(batch.batch["responses"])
        offpolicy_mask = torch.zeros(batch_size, dtype=torch.float32)
        offpolicy_mask[hard_batch_idxs] = 1.0
        batch.batch["offpolicy_sample_mask"] = offpolicy_mask
        metrics["selective_guidance/n_offpolicy_samples"] = int(offpolicy_mask.sum().item())

        post_group_stats = self._collect_group_reward_stats(batch, reward_tensor)
        post_reward_uids = set(post_group_stats["reward_uids"])
        hard_rewarded_after = sum(uid in post_reward_uids for uid in hard_uids)
        rescued_hard = sum(uid not in pre_reward_uids and uid in post_reward_uids for uid in hard_uids)
        metrics["selective_guidance/n_hard_groups_with_reward_after_guidance"] = hard_rewarded_after
        metrics["selective_guidance/frac_hard_groups_with_reward_after_guidance"] = (
            hard_rewarded_after / n_hard if n_hard > 0 else 0.0
        )
        metrics["selective_guidance/n_rescued_hard_groups"] = rescued_hard
        metrics["selective_guidance/frac_rescued_hard_groups"] = rescued_hard / n_hard if n_hard > 0 else 0.0
        metrics["selective_guidance/n_reward_groups_after_guidance"] = len(post_group_stats["reward_uids"])
        metrics["selective_guidance/frac_reward_groups_after_guidance"] = (
            len(post_group_stats["reward_uids"]) / prompt_count if prompt_count > 0 else 0.0
        )

        return reward_tensor, reward_extra_infos_dict

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)

            # Apply guidance injection to validation prompts if test_enabled
            if self.guidance_test_enabled:
                from verl.trainer.ppo.adaptive_guidance import (
                    _parse_all_actions,
                    rebuild_raw_prompt_with_k,
                )

                extra_infos = test_gen_batch.non_tensor_batch.get("extra_info", [])
                for i in range(len(test_gen_batch)):
                    info = extra_infos[i] if i < len(extra_infos) else {}
                    all_actions = _parse_all_actions(info.get("all_actions"))
                    if not all_actions:
                        continue
                    k = self._resolve_guidance_fixed_k(len(all_actions))
                    base_question = info.get("base_question", info.get("question", ""))
                    test_gen_batch.non_tensor_batch["raw_prompt"][i] = rebuild_raw_prompt_with_k(
                        test_gen_batch.non_tensor_batch["raw_prompt"][i],
                        base_question,
                        all_actions,
                        k,
                        injection_mode=self.guidance_injection_mode,
                        prompt_style=info.get("guidance_prompt_style", "default"),
                    )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # breakpoint()
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            print('jyxjyxjyx, ', self.config.trainer.ray_wait_register_center_timeout)
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                # 保存 guidance 注入前的原始 prompt，供 teacher batch 构建、
                # off-policy 分支和 unguided batch 重建使用。
                original_raw_prompts = None
                if self.guidance_enabled:
                    from copy import deepcopy as _deepcopy

                    raw_prompts = gen_batch.non_tensor_batch.get("raw_prompt")
                    if raw_prompts is not None:
                        original_raw_prompts = np.empty(len(gen_batch), dtype=object)
                        for i in range(len(gen_batch)):
                            original_raw_prompts[i] = _deepcopy(raw_prompts[i])

                # Selective mode: 主 rollout 不注入 guidance，reward 计算后
                # 仅对困难组（所有 rollout 均 0 分）注入 guidance 重新 rollout。
                is_selective = (self.guided_offpolicy_enabled
                                and self.guided_offpolicy_mode == "selective")

                # Guidance injection: either adaptive (binary-search k) or fixed-k
                if self.guidance_enabled and not is_selective:
                    if self.adaptive_guidance is not None:
                        with marked_timer("adaptive_guidance", timing_raw, color="cyan"):
                            generate_fn = (
                                self.async_rollout_manager.generate_sequences
                                if self.async_rollout_mode
                                else self.actor_rollout_wg.generate_sequences
                            )
                            optimal_k = self.adaptive_guidance.search(
                                gen_batch, batch, generate_fn, self.reward_fn
                            )
                            self.adaptive_guidance.apply(gen_batch, batch, optimal_k)
                            metrics["adaptive_guidance/mean_k"] = np.mean(optimal_k)
                            metrics["adaptive_guidance/max_k"] = float(np.max(optimal_k))
                            metrics["adaptive_guidance/min_k"] = float(np.min(optimal_k))
                    else:
                        # Fixed-k guidance: inject without adaptive search
                        from verl.trainer.ppo.adaptive_guidance import (
                            _parse_all_actions,
                            rebuild_raw_prompt_with_k,
                        )

                        extra_infos = gen_batch.non_tensor_batch["extra_info"]
                        for i in range(len(gen_batch)):
                            info = extra_infos[i]
                            all_actions = _parse_all_actions(info.get("all_actions"))
                            if not all_actions:
                                continue
                            k = self._resolve_guidance_fixed_k(len(all_actions))
                            base_question = info.get("base_question", info.get("question", ""))
                            gen_batch.non_tensor_batch["raw_prompt"][i] = rebuild_raw_prompt_with_k(
                                gen_batch.non_tensor_batch["raw_prompt"][i],
                                base_question,
                                all_actions,
                                k,
                                injection_mode=self.guidance_injection_mode,
                                prompt_style=info.get("guidance_prompt_style", "default"),
                            )
                        metrics["guidance/fixed_k"] = float(self.guidance_fixed_k)
                        if self.guidance_fixed_ratio > 0:
                            metrics["guidance/fixed_ratio"] = float(self.guidance_fixed_ratio)

                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    if original_raw_prompts is not None:
                        batch.non_tensor_batch["unguided_raw_prompt"] = np.repeat(
                            original_raw_prompts,
                            self.config.actor_rollout_ref.rollout.n,
                            axis=0,
                        )

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    group_reward_stats = self._collect_group_reward_stats(batch, reward_tensor)
                    self._log_group_reward_metrics(metrics, group_reward_stats, prefix="group_reward")

                    # -----------------------------------------------------------
                    # Selective guidance: 仅对困难组做 guided rollout 并替换
                    # -----------------------------------------------------------
                    if is_selective:
                        with marked_timer("selective_guidance", timing_raw, color="cyan"):
                            reward_tensor, reward_extra_infos_dict = self._selective_guidance_phase(
                                batch, gen_batch, reward_tensor, reward_extra_infos_dict,
                                metrics, timing_raw,
                            )

                    # Reward-gated OPD: run guided teacher rollout to decide
                    # which samples are worth distilling.
                    # teacher_rollout_n is independent from the student rollout.n,
                    # so the teacher only needs 1-2 rollouts per prompt while student
                    # may have n=8.  Each student rollout is compared against the
                    # mean teacher reward for that prompt.
                    if self.distillation_config is not None and self.reward_gate:
                        with marked_timer("teacher_rollout_gate", timing_raw, color="magenta"):
                            generate_fn = (
                                self.async_rollout_manager.generate_sequences
                                if self.async_rollout_mode
                                else self.actor_rollout_wg.generate_sequences
                            )
                            # gen_batch is the original (non-repeated) batch
                            guided_gen = self._apply_guidance_to_gen_batch(gen_batch)
                            teacher_n = self.teacher_rollout_n
                            if teacher_n > 1:
                                guided_gen = guided_gen.repeat(
                                    repeat_times=teacher_n, interleave=True
                                )
                            guided_gen_output = generate_fn(guided_gen)

                            # Attach reward metadata so compute_reward can evaluate
                            n_teacher_samples = len(guided_gen_output)
                            for key in ("reward_model", "data_source", "extra_info"):
                                if key in gen_batch.non_tensor_batch:
                                    src = gen_batch.non_tensor_batch[key]
                                    if teacher_n > 1:
                                        guided_gen_output.non_tensor_batch[key] = np.repeat(src, teacher_n, axis=0)
                                    else:
                                        guided_gen_output.non_tensor_batch[key] = src

                            teacher_reward_tensor, _ = compute_reward(guided_gen_output, self.reward_fn)
                            teacher_seq_reward = teacher_reward_tensor.sum(dim=-1)  # (batch_size * teacher_n,)

                            # Average teacher rewards per prompt
                            prompt_count = len(gen_batch)
                            if teacher_n > 1:
                                teacher_seq_reward = teacher_seq_reward.reshape(prompt_count, teacher_n).mean(dim=-1)
                            # teacher_seq_reward: (prompt_count,)

                            # Expand to match student batch: each prompt has rollout.n student rollouts
                            student_n = self.config.actor_rollout_ref.rollout.n
                            teacher_reward_expanded = teacher_seq_reward.repeat_interleave(student_n)
                            # teacher_reward_expanded: (prompt_count * student_n,) = batch size

                            student_seq_reward = reward_tensor.sum(dim=-1)  # (prompt_count * student_n,)

                            opd_mask = (
                                teacher_reward_expanded > student_seq_reward + self.reward_gate_margin
                            ).float()
                            batch.batch["opd_mask"] = opd_mask

                            metrics["distillation/gate_frac"] = opd_mask.mean().item()
                            metrics["distillation/teacher_reward_mean"] = teacher_seq_reward.mean().item()
                            metrics["distillation/student_reward_mean"] = student_seq_reward.mean().item()

                            del guided_gen, guided_gen_output, teacher_reward_tensor

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    # -----------------------------------------------------------
                    # Guided Off-Policy: prepare unguided tensors for
                    # off-policy branch. old_log_probs = pi(r|behavior, θ_old).
                    #
                    # dual mode: keep main batch as behavior prompt and run an
                    #   additional unguided forward for the auxiliary branch.
                    #   KL 对两个分支分别计算并按 coef 加权（见 dp_actor.py）。
                    # single mode: keep trainer-side batch untouched for
                    #   reward/logging bookkeeping, but stash unguided tensors
                    #   for actor/ref forwards. This is used by selective mode:
                    #   easy groups are effectively on-policy because their
                    #   behavior prompt is already unguided, while hard groups
                    #   become true off-policy updates from guided behavior to
                    #   unguided target.
                    # -----------------------------------------------------------
                    if self.guided_offpolicy_enabled:
                        with marked_timer("guided_offpolicy_prep", timing_raw, color="cyan"):
                            guided_seqlen = batch.batch["input_ids"].shape[1]
                            unguided_batch = self._construct_unguided_batch(batch)
                            batch.batch["unguided_input_ids"] = unguided_batch.batch["input_ids"]
                            batch.batch["unguided_attention_mask"] = unguided_batch.batch["attention_mask"]
                            batch.batch["unguided_position_ids"] = unguided_batch.batch["position_ids"]
                            if "multi_modal_inputs" in unguided_batch.non_tensor_batch:
                                batch.non_tensor_batch["unguided_multi_modal_inputs"] = (
                                    unguided_batch.non_tensor_batch["multi_modal_inputs"]
                                )
                            unguided_seqlen = unguided_batch.batch["input_ids"].shape[1]
                            del unguided_batch

                            metrics["guided_offpolicy/guided_seqlen"] = float(guided_seqlen)
                            metrics["guided_offpolicy/unguided_seqlen"] = float(unguided_seqlen)
                            metrics["guided_offpolicy/seqlen_diff"] = float(guided_seqlen - unguided_seqlen)
                            metrics["guided_offpolicy/mode"] = 1.0 if self.guided_offpolicy_mode == "dual" else 0.0

                        if self.guided_offpolicy_mode == "selective":
                            # selective uses guided behavior only on hard groups,
                            # but the actor/ref branches should still optimize
                            # the unguided target prompt.
                            batch.meta_info["offpolicy_mode"] = "single"
                            batch.meta_info["offpolicy_coef"] = self.guided_offpolicy_offpolicy_coef
                            batch.meta_info["offpolicy_clip_ratio"] = self.guided_offpolicy_offpolicy_clip_ratio
                            batch.meta_info["offpolicy_clip_ratio_c"] = self.guided_offpolicy_offpolicy_clip_ratio_c
                            batch.meta_info["luffy_gamma"] = self.guided_offpolicy_luffy_gamma
                        else:
                            batch.meta_info["offpolicy_mode"] = self.guided_offpolicy_mode
                            batch.meta_info["onpolicy_coef"] = self.guided_offpolicy_onpolicy_coef
                            batch.meta_info["offpolicy_coef"] = self.guided_offpolicy_offpolicy_coef
                            batch.meta_info["onpolicy_clip_ratio"] = self.guided_offpolicy_onpolicy_clip_ratio
                            batch.meta_info["onpolicy_clip_ratio_c"] = self.guided_offpolicy_onpolicy_clip_ratio_c
                            batch.meta_info["offpolicy_clip_ratio"] = self.guided_offpolicy_offpolicy_clip_ratio
                            batch.meta_info["offpolicy_clip_ratio_c"] = self.guided_offpolicy_offpolicy_clip_ratio_c

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                        # Guided off-policy: 计算 unguided prompt 下的 ref log prob，
                        # 使 off-policy 分支的 KL 正则使用匹配的 prompt 条件
                        if self.guided_offpolicy_enabled and "unguided_input_ids" in batch.batch:
                            with marked_timer("unguided_ref_log_prob", timing_raw, color="olive"):
                                non_tensors = {}
                                if "unguided_multi_modal_inputs" in batch.non_tensor_batch:
                                    non_tensors["multi_modal_inputs"] = batch.non_tensor_batch[
                                        "unguided_multi_modal_inputs"
                                    ]
                                unguided_ref_batch = DataProto.from_dict(
                                    tensors={
                                        "input_ids": batch.batch["unguided_input_ids"],
                                        "attention_mask": batch.batch["unguided_attention_mask"],
                                        "position_ids": batch.batch["unguided_position_ids"],
                                        "responses": batch.batch["responses"],
                                    },
                                    non_tensors=non_tensors,
                                )
                                unguided_ref_batch.meta_info = batch.meta_info.copy()
                                if not self.ref_in_actor:
                                    unguided_ref_output = self.ref_policy_wg.compute_ref_log_prob(unguided_ref_batch)
                                else:
                                    unguided_ref_output = self.actor_rollout_wg.compute_ref_log_prob(unguided_ref_batch)
                                batch.batch["unguided_ref_log_prob"] = unguided_ref_output.batch["ref_log_prob"]

                    # On-policy distillation: compute teacher log-probs with guided prompt.
                    if self.distillation_config is not None and "teacher_log_probs" not in batch.batch:
                        with marked_timer("teacher_log_prob", timing_raw, color="magenta"):
                            teacher_batch = self._construct_teacher_batch(batch)
                            teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
                            batch.batch["teacher_log_probs"] = teacher_output.batch["old_log_probs"]
                            metrics["distillation/coef"] = self.distill_coef

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            if self.distillation_config is not None:
                                batch.meta_info["distill_coef"] = self.distill_coef
                                batch.meta_info["policy_loss_coef"] = self.policy_loss_coef
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # Decay distillation coefficient
                        if self.distillation_config is not None:
                            metrics["distillation/policy_loss_coef"] = self.policy_loss_coef
                            self.distill_coef = max(
                                self.distill_coef_min,
                                self.distill_coef * self.distill_coef_decay,
                            )

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
