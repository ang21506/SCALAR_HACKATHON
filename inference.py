import json
import os
import sys
from typing import Any, Dict, Optional

from openai import OpenAI

from env import SmartIrrigationEnv
from schemas import Action


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK_NAME = os.getenv("BENCHMARK", "smart-irrigation-env")
TASK_NAME = os.getenv("TASK_NAME", os.getenv("TASK", "task1_easy"))

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _extract_error(info: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(info, dict):
        return None
    error_value = info.get("last_action_error")
    if error_value is None:
        return None
    return str(error_value).replace("\n", " ")


def _rule_based_action(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    action_amounts = []
    heavy_rain_expected = obs_dict["rain_probability"] > 0.5 and obs_dict["expected_rainfall"] > 1.0

    for moisture in obs_dict["soil_moistures"]:
        if moisture < 0.35:
            action_amounts.append(1 if heavy_rain_expected else (2 if obs_dict["tariff_band"] == "off-peak" else 1))
        elif moisture < 0.55 and not heavy_rain_expected and obs_dict["tariff_band"] == "off-peak":
            action_amounts.append(1)
        else:
            action_amounts.append(0)

    return {"irrigation_amounts": action_amounts}


def _llm_action(obs_dict: Dict[str, Any], num_plots: int) -> Dict[str, Any]:
    prompt = f"""
You are controlling an irrigation system for {num_plots} plots.
Return only valid JSON with this schema:
{{"irrigation_amounts":[0,0,0]}}

Observation:
{json.dumps(obs_dict, indent=2)}

Guidelines:
- Keep soil moisture in the healthy range.
- Avoid overwatering if rain is likely.
- Use integer irrigation units only.
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    action_dict = json.loads(content)
    irrigation_amounts = action_dict.get("irrigation_amounts")
    if not isinstance(irrigation_amounts, list):
        raise ValueError("LLM response missing irrigation_amounts list")

    normalized_amounts = []
    for value in irrigation_amounts[:num_plots]:
        normalized_amounts.append(max(0, int(value)))
    while len(normalized_amounts) < num_plots:
        normalized_amounts.append(0)

    return {"irrigation_amounts": normalized_amounts}


def _choose_action(obs_dict: Dict[str, Any], num_plots: int) -> Dict[str, Any]:
    try:
        return _llm_action(obs_dict, num_plots)
    except Exception as exc:
        print(f"LLM fallback: {exc}", file=sys.stderr)
        return _rule_based_action(obs_dict)


def _emit_start() -> None:
    print(f"[START] task={TASK_NAME} env={BENCHMARK_NAME} model={MODEL_NAME}")


def _emit_step(step_number: int, action_dict: Dict[str, Any], reward_value: float, done: bool, error: Optional[str]) -> None:
    error_text = "null" if error is None else error.replace("\n", " ")
    print(
        f"[STEP] step={step_number} action={_compact_json(action_dict)} "
        f"reward={reward_value:.2f} done={_format_bool(done)} error={error_text}"
    )


def _emit_end(success: bool, step_count: int, rewards: list[float]) -> None:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={_format_bool(success)} steps={step_count} rewards={reward_text}")


def run_episode() -> None:
    env = SmartIrrigationEnv(task=TASK_NAME)
    rewards: list[float] = []
    step_count = 0
    success = False

    _emit_start()

    try:
        observation = env.reset()
        done = False

        while not done:
            step_count += 1
            obs_dict = observation.model_dump()
            action_dict = _choose_action(obs_dict, env.num_plots)

            reward_value = 0.0
            step_error: Optional[str] = None

            try:
                action = Action(**action_dict)
                observation, reward, done, info = env.step(action)
                reward_value = float(reward.value)
                step_error = _extract_error(info)
            except Exception as exc:
                done = True
                step_error = str(exc).replace("\n", " ")

            rewards.append(reward_value)
            _emit_step(step_count, action_dict, reward_value, done, step_error)

        try:
            success = bool(env.grade().success)
        except Exception:
            success = bool(done)
    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass
        _emit_end(success, step_count, rewards)


if __name__ == "__main__":
    run_episode()
