import os
import json
from openai import OpenAI
from env import SmartIrrigationEnv
from schemas import Action

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = HF_TOKEN.strip() if HF_TOKEN is not None else None

if HF_TOKEN is None or HF_TOKEN == "":
    raise ValueError("HF_TOKEN environment variable is required")

if API_BASE_URL == "":
    raise ValueError("API_BASE_URL must not be empty")

if MODEL_NAME == "":
    raise ValueError("MODEL_NAME must not be empty")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(task_name: str, env_id: str, seed: int = 42):
    env = SmartIrrigationEnv(task=task_name, seed=seed)

    print(f"[START] task={task_name} env={env_id} model={MODEL_NAME}")

    step_idx = 0
    rewards = []
    success = False

    try:
        obs = env.reset(seed=seed)
        done = False

        while not done:
            step_idx += 1
            obs_dict = obs.model_dump()

            prompt = f"""
            You are managing an irrigation system for {env.num_plots} agricultural plots.
            Current observation:
            {json.dumps(obs_dict, indent=2)}

            Rules:
            1. Maintain soil moisture between 0.3 and 0.8.
            2. Try to minimize energy costs (avoid watering heavily in "peak" tariff).
            3. Take expected rainfall into account to avoid wasting water.
            4. Provide the result as a JSON object with the format:
               {{"irrigation_amounts": [integer, integer, ... (for each plot)]}}
            where each integer represents discrete units of water to apply.

            Only output valid JSON.
            """

            action_dict = None
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                res_str = response.choices[0].message.content
                action_dict = json.loads(res_str)
            except Exception:
                action_dict = None

            if action_dict and "irrigation_amounts" not in action_dict:
                action_dict = None

            if action_dict is None:
                action_dict = {"irrigation_amounts": [0] * env.num_plots}

            action = Action(**action_dict)
            obs, reward, done, info = env.step(action)
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))

            r_val = float(reward.value)
            rewards.append(r_val)

            done_str = "true" if done else "false"
            last_action_error = info.get("last_action_error") if isinstance(info, dict) else None
            err_str = str(last_action_error).replace("\n", " ") if last_action_error is not None else "null"

            print(f"[STEP] step={step_idx} action={action_str} reward={r_val:.2f} done={done_str} error={err_str}")

        grade = env.grade()
        success = bool(grade.success)
    except Exception:
        success = False
    finally:
        try:
            env.close()
        finally:
            success_str = "true" if success else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={success_str} steps={step_idx} rewards={rewards_str}")

if __name__ == "__main__":
    task_name = os.getenv("TASK_NAME", "task1_easy")
    env_id = os.getenv("BENCHMARK", "smart-irrigation")
    seed = int(os.getenv("SEED", "42"))
    run_inference(task_name=task_name, env_id=env_id, seed=seed)
