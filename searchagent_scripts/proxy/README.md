# 单端口 8000 网关：多服务统一出口

训练脚本依赖**工具服务**（DeepResearch search/visit）和** Reward 模型**（vLLM）。若本机只能对外暴露 **8000** 一个端口，可用本目录的反向代理把多个后端统一到 8000，供外部训练机访问。

## 路径与后端对应关系

| 对外路径（端口 8000） | 后端服务 | 默认本机端口 |
|----------------------|----------|--------------|
| `/tools/`             | DeepResearch API (search/visit) | 8010 |
| `/reward1/`           | Reward vLLM 实例 1               | 7001 |
| `/reward2/`           | Reward vLLM 实例 2               | 7002 |

即：
- `http://<本机>:8000/tools/search` → 本机 `8010/search`
- `http://<本机>:8000/tools/visit` → 本机 `8010/visit`
- `http://<本机>:8000/reward1/v1/...` → 本机 `7001/v1/...`
- `http://<本机>:8000/reward2/v1/...` → 本机 `7002/v1/...`

## 方式一：Nginx

1. 确保本机已安装 nginx。
2. 启动各后端（工具服务 8010、Reward 7001/7002）。
3. 使用本目录配置启动 nginx（监听 8000）：

```bash
# 若 nginx 已安装且无冲突
sudo nginx -c "$(pwd)/nginx_8000.conf"
# 或把 nginx_8000.conf 拷贝到 /etc/nginx/conf.d/ 后
sudo nginx -s reload
```

若报错找不到 `mime.types`，可编辑 `nginx_8000.conf`，将 `include /etc/nginx/mime.types;` 改为你本机路径或注释掉该行。

## 方式二：Python 反向代理（推荐，无需 nginx）

1. 依赖：已具备 `fastapi`、`uvicorn`（与 tool_server 一致）。
2. 先在本机启动：
   - 工具服务：如 `PORT=8010` 启动 `api_server_deepresearch_redis.py`
   - Reward：在 7001、7002 分别启动 vLLM
3. 再启动网关（监听 8000）：

```bash
cd searchagent_scripts/proxy
python run_proxy.py
# 或
uvicorn run_proxy:app --host 0.0.0.0 --port 8000
```

可选环境变量：
- `PROXY_PORT`：网关端口，默认 8000
- `TOOLS_BACKEND`：工具服务地址，默认 `http://127.0.0.1:8010`
- `REWARD1_BACKEND` / `REWARD2_BACKEND`：Reward 地址，默认 7001/7002

## 训练机侧配置

训练在**另一台机器**上跑时，需要让训练脚本通过「本机:8000」访问上述服务。

### 1. 工具配置（Tool Config）

使用网关版配置并替换 `GATEWAY_HOST` 为本机 IP（或域名），训练机要能解析并访问该 IP 的 8000 端口。

```bash
# 例：本机对训练机可见的 IP 为 192.168.1.100
export GATEWAY_HOST=192.168.1.100
sed "s/GATEWAY_HOST/${GATEWAY_HOST}/g" \
  searchagent_scripts/config/deepresearch_tool_config_gateway.yaml \
  > /tmp/deepresearch_tool_config.yaml
```

训练时指定：
`actor_rollout_ref.rollout.multi_turn.tool_config_path=/tmp/deepresearch_tool_config.yaml`  
这样工具 URL 为 `http://192.168.1.100:8000/tools/search` 与 `.../tools/visit`。

### 2. Reward 模型端点（endpoint_list）

在 `train_searchagent_local.sh` 中把原来的 `127.0.0.1:6001/v1`、`127.0.0.1:6002/v1` 改为通过 8000 的路径，例如：

```bash
# 将 <本机IP> 换成实际 IP
custom_reward_function.reward_kwargs.endpoint_list='[http://<本机IP>:8000/reward1/v1,http://<本机IP>:8000/reward2/v1]'
```

例如本机 IP 为 `192.168.1.100`：

```bash
custom_reward_function.reward_kwargs.endpoint_list='[http://192.168.1.100:8000/reward1/v1,http://192.168.1.100:8000/reward2/v1]'
```

### 3. 小结

- **本机**：只开放 8000；在本机起 8010（工具）、7001/7002（Reward），再起 nginx 或 `run_proxy.py` 在 8000 做转发。
- **训练机**：工具 config 里 URL 用 `http://<本机IP>:8000/tools/search` 和 `.../tools/visit`；Reward 用 `http://<本机IP>:8000/reward1/v1` 和 `.../reward2/v1`。

这样所有服务都通过 8000 端口对外提供，训练机只需能访问本机的 8000 即可。
