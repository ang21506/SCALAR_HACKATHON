import json
from env import SmartIrrigationEnv
from schemas import Action
from openai import OpenAI
import os

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

def rule_based_agent(obs_dict):
    action_amounts = []
    
    # If a lot of rain is expected, don't irrigate unless critically low
    heavy_rain_expected = obs_dict['rain_probability'] > 0.5 and obs_dict['expected_rainfall'] > 1.0

    for moisture in obs_dict['soil_moistures']:
        if moisture < 0.35: # Critically low
            if heavy_rain_expected:
                # Give just enough to survive
                action_amounts.append(1)
            else:
                # If off-peak give 2, if peak give 1
                action_amounts.append(2 if obs_dict['tariff_band'] == 'off-peak' else 1)
        elif moisture < 0.55 and not heavy_rain_expected and obs_dict['tariff_band'] == 'off-peak':
            # Proactive watering during off-peak to save energy later
            action_amounts.append(1)
        else:
            action_amounts.append(0)
    
    return {"irrigation_amounts": action_amounts}

def run_episode(env, agent_type="rule_based"):
    print("START")
    obs = env.reset()
    done = False
    
    total_reward = 0.0
    while not done:
        print("STEP")
        obs_dict = obs.model_dump()
        
        if agent_type == "rule_based":
            action_dict = rule_based_agent(obs_dict)
        elif agent_type == "llm":
            action_dict = llm_agent(obs_dict, env.num_plots)
        else:
            action_dict = {"irrigation_amounts": [0] * env.num_plots}
            
        action = Action(**action_dict)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        
    print("END")
    return total_reward

def llm_agent(obs_dict, num_plots):
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )
    
    prompt = f"""
    You are managing an irrigation system for {num_plots} agricultural plots.
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
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        res_str = response.choices[0].message.content
        return json.loads(res_str)
    except Exception as e:
        print(f"LLM Error: {e}, falling back to no watering")
        return {"irrigation_amounts": [0] * num_plots}

if __name__ == "__main__":
    has_api_key = HF_TOKEN is not None
    if not has_api_key:
        print("HF_TOKEN not found in environment. Only running rule-based baseline.\n")

    for task in ["task1_easy", "task2_medium", "task3_hard"]:
        print(f"--- Task: {task} ---")
        
        # Run Rule-Based
        env = SmartIrrigationEnv(task=task)
        score = run_episode(env, agent_type="rule_based")
        grade = env.grade()
        print(f"Rule-Based Agent Score: {grade.score:.3f} (Raw: {score:.1f}) | Success: {grade.success}")
        
        # Run LLM
        if has_api_key:
            env = SmartIrrigationEnv(task=task)
            score_llm = run_episode(env, agent_type="llm")
            grade_llm = env.grade()
            print(f"LLM Agent Score: {grade_llm.score:.3f} (Raw: {score_llm:.1f}) | Success: {grade_llm.success}")
        
        print()
