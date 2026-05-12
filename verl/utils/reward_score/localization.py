import re
import json
import random
import warnings
from geopy.distance import geodesic
import logging
import torch
import os


logger = logging.getLogger(__name__)
level_name = os.getenv("VERL_LOGGING_LEVEL", "INFO").upper()
level_num = getattr(logging, level_name, logging.INFO)
logger.setLevel(level_num)

try:
    rank = torch.distributed.get_rank()
except:
    rank = 0


def extract_all_json_objects(text):
    """
    从一个复杂字符串中提取所有可能的 JSON 对象（匹配大括号包围的内容）
    如果括号嵌套复杂，可进一步用手动栈解析。
    """
    objects = []
    stack = []
    start_idx = None

    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    objects.append(candidate)
                    start_idx = None
    return objects

def find_coords_json(text):
    """
    提取包含 lat/lon/city/country 的 JSON 对象
    """
    json_candidates = extract_all_json_objects(text)

    for cand in json_candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue

        if all(k in obj for k in ("lat", "lon", "city", "country")):
            return obj
    return None

def extract_solution(solution_str):
    result = find_coords_json(solution_str)
    if result:
        return result["lat"], result["lon"], result["city"], result["country"]
    else:
        return None, None, None, None


def compute_distance_score(solution_str, ground_truth, data_source=None, **kwargs):
    lat, lon, city, country = extract_solution(solution_str)

    result = {
        "score": 0,
        "acc_500m": 0,
        "acc_2km": 0,
        "acc_10km": 0,
        "acc_25km": 0,
        "acc_200km": 0,
        "acc_750km": 0,
    }

    if lat != None and lon != None:
        try:
            distance_m = geodesic((lat, lon), (ground_truth["lat"], ground_truth["lon"])).meters
        except Exception as e:
            warnings.warn(f"Error when cal distance: {e}", RuntimeWarning)
            distance_score = 0.
            distance_m = 99999999
    else:
        distance_score = 0.
        distance_m = 99999999

    if distance_m < 500:
        distance_score = 1.0
        result["acc_500m"] = 1.
        result["acc_2km"] = 1.
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 2000:
        distance_score = 0.8
        result["acc_2km"] = 1.
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 10000:
        distance_score = 0.6
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 25000:
        distance_score = 0.4
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 200000:
        distance_score = 0.2
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 750000:
        distance_score = 0.1
        result["acc_750km"] = 1.
    else:
        distance_score = 0.
    
    result["score"] = distance_score

    do_print = random.randint(1, 32) == 1
    if do_print:
        logger.info(f"----------------{ground_truth['split']} data sample----------------")
        logger.info(f"input image: {ground_truth['image']}")
        logger.info("----------------solution str----------------")
        for line in solution_str.splitlines():
            logger.info(line)
        logger.info(f"predict lat: {lat}, lon: {lon}, city: {city}, country: {country}")
        logger.info(f"ground truth lat: {ground_truth['lat']}, lon: {ground_truth['lon']}, city: {ground_truth['city']}")
        logger.info(f"distance_m: {distance_m}")

    return result


def valid_hypothesis(h):
    return (
        isinstance(h, dict) and
        "lat" in h and isinstance(h["lat"], (int, float)) and
        "lon" in h and isinstance(h["lon"], (int, float)) and
        "evidence" in h and isinstance(h["evidence"], list)
    )


def compute_distance_score_v2(solution_str, ground_truth, beta=0.05, data_source=None, **kwargs):
    tool_names = set()
    for match in re.finditer(r'<tool_call>\s*\{[^}]*"name"\s*:\s*"([^"]+)"', solution_str):
        tool_names.add(match.group(1))
    init_json_match = re.search(r'\{\s*"coarse_location"[^}]+\}', solution_str, re.S)
    init_json = None
    if init_json_match:
        try:
            init_json = json.loads(init_json_match.group(0))
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse initial JSON: {e}", RuntimeWarning)
            init_json = None

    pattern = re.compile(
        r'\{\s*"hypotheses"\s*:\s*\[.*?\]\s*,\s*"final_answer"\s*:\s*\{.*?\}\s*\}',
        re.S
    )
    report_json_match = pattern.search(solution_str)
    report_json = None
    if report_json_match:
        try:
            report_json = json.loads(report_json_match.group(0))
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse report JSON: {e}", RuntimeWarning)
            report_json = {}
        lat = report_json.get("final_answer", {}).get("lat")
        lon = report_json.get("final_answer", {}).get("lon")
        city = report_json.get("final_answer", {}).get("city")
        country = report_json.get("final_answer", {}).get("country")
        hypotheses = report_json.get("post_interaction_hypotheses", [])
        hypotheses_cnt = 0
        for hypothesis in hypotheses:
            if valid_hypothesis(hypothesis):
                hypotheses_cnt += 1
    else:
        lat = None
        lon = None
        city = None
        country = None
        hypotheses_cnt = 0

    bonus_score = 0.
    ## Bonus reward
    if init_json:
        bonus_score += beta
    if hypotheses_cnt > 1:
        bonus_score += beta
    if "image_zoom_in_tool" in tool_names:
        bonus_score += beta
    tool_type_cnt = 0
    for tool_name in tool_names:
        if tool_name == "image_zoom_in_tool":
            tool_type_cnt += 1
        if tool_name in ["amap_keyword_search", "amap_poi_detail", "amap_input_tips"]:
            tool_type_cnt += 1
        if tool_name in ["amap_static_map", "amap_satellite_map"]:
            tool_type_cnt += 1
    if tool_type_cnt > 1:
        bonus_score += beta

    result = {
        "score": 0,
        "acc_500m": 0,
        "acc_2km": 0,
        "acc_10km": 0,
        "acc_25km": 0,
        "acc_200km": 0,
        "acc_750km": 0,
    }

    if lat != None and lon != None:
        try:
            distance_m = geodesic((lat, lon), (ground_truth["lat"], ground_truth["lon"])).meters
        except Exception as e:
            warnings.warn(f"Error when cal distance: {e}", RuntimeWarning)
            distance_score = 0.
            distance_m = 99999999
    else:
        distance_score = 0.
        distance_m = 99999999

    if distance_m < 500:
        distance_score = 1.0
        result["acc_500m"] = 1.
        result["acc_2km"] = 1.
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 2000:
        distance_score = 0.8
        result["acc_2km"] = 1.
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 10000:
        distance_score = 0.6
        result["acc_10km"] = 1.
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 25000:
        distance_score = 0.4
        result["acc_25km"] = 1.
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 200000:
        distance_score = 0.2
        result["acc_200km"] = 1.
        result["acc_750km"] = 1.
    elif distance_m < 750000:
        distance_score = 0.1
        result["acc_750km"] = 1.
    else:
        distance_score = 0.
    
    result["score"] = distance_score + bonus_score

    do_print = random.randint(1, 16) == 1
    if do_print:
        logger.info(f"----------------{ground_truth['split']} data sample----------------")
        logger.info(f"input image: {ground_truth['image']}")
        logger.info("----------------solution str----------------")
        for line in solution_str.splitlines():
            logger.info(line)
        logger.info(f"predict lat: {lat}, lon: {lon}, city: {city}, country: {country}")
        logger.info(f"ground truth lat: {ground_truth['lat']}, lon: {ground_truth['lon']}, city: {ground_truth['city']}")
        logger.info(f"distance_m: {distance_m}")
        logger.info(f"tool_names: {tool_names}")
        logger.info(f"bonus_score: {bonus_score}")
    return result


if __name__ == "__main__":
    ground_truth = {
        "lat": 25.03,
        "lon": 102.7,
    }
    solution_str = '''
    {
        "coarse_location": "China",
        "city": "Xi'an",
        "country": "China",
        "clues": [
            "The text on the sign reads 'Grand Tang Mall', which is a well-known shopping complex in China.",
            "The sign also mentions '大雁塔', which is a famous historical landmark in Xi'an."
        ]
    }
    think
    <tool_call>
    {"name": "image_zoom_in_tool", "parameters": "query"}
    <tool_call>
    {
        "post_interaction_hypotheses": [
            {
                "lat": 25.034701,
                "lon": 102.702350,
                "evidence": [
                    "通过搜索芙颜堂并查询详细信息，定位到'芙颜堂皮肤管理(国防路店)",
                    "该店地址为国防路34-2号，位于昆明市"
                ]
            }
        ],
        "final_answer": {
            "lat": 25.034701,
            "lon": 102.702350,
            "city": "昆明市",
            "country": "中国"
        }
    }
    '''
    print(compute_distance_score_v2(solution_str, ground_truth))

