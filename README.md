# Smart Irrigation Environment

A realistic agricultural reinforcement learning environment for smart irrigation, compliant with the OpenEnv specification. The environment simulates an irrigation system subject to water quotas, energy tariffs, and weather uncertainty.

## Environment Details

### State and Action Spaces
- **Observation:** `schemas.Observation` Pydantic model representing current soil moisture for each plot, crop types, growth stages, expected rainfall with probability, daily water quota, pump capacity, and the current energy tariff band (peak/off-peak).
- **Action:** `schemas.Action` Pydantic model containing `irrigation_amounts`, a list of units to be applied to each plot respectively. 

### Reward Function
The environment provides a meaningful reward over the full trajectory. It rewards maintaining crop soil moisture between healthy ranges, while penalizing deviations outside of these bounds. It also applies penalties for excess energy usage (irrigation during peak hours), raw water usage, and water wasted (spilled) during heavy rains.

### Historic Weather Simulation
Weather is sourced from a `historical_weather.csv` dataset recreating a realistic extreme-drought week followed by a severe storm, providing a robust time-series test to challenge agent planners.

### Task Information
The environment supports three tasks of increasing difficulty:
- `task1_easy`: Single crop, identical soil across 2 plots, simple tariff schedule, relatively accurate weather forecast.
- `task2_medium`: Mixed crops spread across 4 plots, tighter water quotas, and pronounced peak/off-peak tariff differences.
- `task3_hard`: 5 diverse plots, highly uncertain rainfall, strict water limits, and complex tariffs.

Each task is programmatically graded on a `0.0-1.0` scale through deterministic criteria using `env.grade()`. This returns a `Grade` Pydantic response containing a numeric `score`, a `success` boolean, and an `analytics` detailed dictionary breaking down the root causes of all accumulated internal penalties for advanced interpretation.

## Usage

### Using the environment
```python
from env import SmartIrrigationEnv
from schemas import Action

env = SmartIrrigationEnv(task="task1_easy")
obs = env.reset()

# Sample action
action = Action(irrigation_amounts=[1, 2])
obs, reward, done, info = env.step(action)

if done:
    grade = env.grade()
    print(f"Episode completed with score: {grade.score}")
    print("Feedback Analytics:", grade.analytics)
```

### Running Baselines
The environment includes rule-based and LLM-based agent baselines.
```bash
# Provide API key if you want to run the LLM baseline
export OPENAI_API_KEY="sk-..."
python baselines.py
```

### Hugging Face Spaces Deployment
To deploy this environment to Hugging Face Spaces, simply push to the target space with this repository. A Dockerfile provides support for serving the built-in Gradio Interface which comes with live **Matplotlib visualizations** natively charting the soil moisture trajectories against the healthy green boundaries!

Local deployment:
```bash
docker build -t smart-irrigation .
docker run -p 7860:7860 smart-irrigation
```
