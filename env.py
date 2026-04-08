import random
import os
import pandas as pd
from typing import Tuple, Dict, Any
from schemas import Observation, Action, Reward, State, Grade

class SmartIrrigationEnv:
    def __init__(self, task: str = "task1_easy"):
        self.task = task
        self.episode_length = 28 # 1 week at 6 hr intervals
        self.step_length_hours = 6
        
        # Configure based on task
        if "easy" in task:
            self.num_plots = 2
            self.crop_types = ["corn"] * 2
            self.pump_capacity = 4.0
            self.base_quota = 10.0
            self.tariff_diff = 1.5
            self.weather_volatility = 0.1
        elif "medium" in task:
            self.num_plots = 4
            self.crop_types = ["corn", "wheat", "soy", "corn"]
            self.pump_capacity = 6.0
            self.base_quota = 12.0
            self.tariff_diff = 2.5
            self.weather_volatility = 0.3
        else: # hard
            self.num_plots = 5
            self.crop_types = ["corn", "wheat", "soy", "cotton", "rice"]
            self.pump_capacity = 6.0
            self.base_quota = 8.0
            self.tariff_diff = 4.0
            self.weather_volatility = 0.6
            
        self.max_moisture = 1.0
        self.min_healthy = 0.3
        self.max_healthy = 0.8
        
        try:
            self.weather_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "historical_weather.csv"))
        except Exception:
            self.weather_data = None
            
        self.reset()
        
    def reset(self) -> Observation:
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.total_energy_penalty = 0.0
        self.total_water_penalty = 0.0
        self.total_stress_penalty = 0.0
        self.total_waste_penalty = 0.0
        self.true_soil_moistures = [random.uniform(0.4, 0.7) for _ in range(self.num_plots)]
        self.growth_stages = [random.uniform(0.1, 0.5) for _ in range(self.num_plots)]
        self.remaining_daily_quota = self.base_quota
        self.upcoming_rain = random.random() < 0.2
        self.history = {i: [] for i in range(self.num_plots)}
        for i in range(self.num_plots):
            self.history[i].append(self.true_soil_moistures[i])
        return self._get_obs()

    def _get_obs(self) -> Observation:
        hour_of_day = (self.step_count * self.step_length_hours) % 24
        tariff_band = "peak" if 14 <= hour_of_day < 20 else "off-peak"
        
        rain_prob = 0.8 if self.upcoming_rain else 0.1
        expected_rain = random.uniform(2.0, 4.0) if self.upcoming_rain else 0.0
        
        if self.weather_data is not None and self.step_count < len(self.weather_data):
            row = self.weather_data.iloc[self.step_count]
            rain_prob = row['rain_probability']
            expected_rain = row['expected_rainfall']
            
        # Add noise based on volatility
        rain_prob = max(0.0, min(1.0, rain_prob + random.uniform(-self.weather_volatility, self.weather_volatility)))
        
        return Observation(
            soil_moistures=list(self.true_soil_moistures),
            crop_types=list(self.crop_types),
            growth_stages=list(self.growth_stages),
            rain_probability=rain_prob,
            expected_rainfall=expected_rain,
            tariff_band=tariff_band,
            daily_water_quota=self.remaining_daily_quota,
            pump_capacity=self.pump_capacity
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # Enforce constraints
        total_irrigation = sum(action.irrigation_amounts)
        actual_irrigation = action.irrigation_amounts
        if total_irrigation > self.pump_capacity:
            factor = self.pump_capacity / max(1, total_irrigation)
            actual_irrigation = [int(a * factor) for a in actual_irrigation]
        
        total_actual = sum(actual_irrigation)
        if total_actual > self.remaining_daily_quota:
            factor = self.remaining_daily_quota / max(1, total_actual)
            actual_irrigation = [int(a * factor) for a in actual_irrigation]

        self.remaining_daily_quota -= sum(actual_irrigation)
        
        # Reset quota daily (every 4 steps of 6 hours)
        if self.step_count % 4 == 0:
            self.remaining_daily_quota = self.base_quota
            
        hour_of_day = (self.step_count * self.step_length_hours) % 24
        tariff_multiplier = self.tariff_diff if 14 <= hour_of_day < 20 else 1.0

        energy_penalty = sum(actual_irrigation) * tariff_multiplier * 0.1
        water_penalty = sum(actual_irrigation) * 0.05
        
        # Weather update
        actual_rain = 0.0
        if self.weather_data is not None and (self.step_count - 1) < len(self.weather_data):
            row = self.weather_data.iloc[self.step_count - 1]
            if random.random() <= row['rain_probability']:
                actual_rain = row['expected_rainfall']
        else:
            if self.upcoming_rain and random.random() < 0.8:
                actual_rain = random.uniform(2.0, 4.0)
            
        # Update state and calc crop penalties
        stress_penalty = 0.0
        waste_penalty = 0.0
        healthy_reward = 0.0
        
        for i in range(self.num_plots):
            # ET based on time of day
            et = 0.05 if 10 <= hour_of_day <= 16 else 0.01
            
            # Update moisture
            self.true_soil_moistures[i] += (actual_irrigation[i] * 0.1) + (actual_rain * 0.1) - et
            self.true_soil_moistures[i] = max(0.0, self.true_soil_moistures[i])
            
            if self.true_soil_moistures[i] < self.min_healthy:
                stress_penalty += (self.min_healthy - self.true_soil_moistures[i]) * 5.0
            elif self.true_soil_moistures[i] > self.max_healthy:
                waste_penalty += (self.true_soil_moistures[i] - self.max_healthy) * 2.0
            else:
                healthy_reward += 0.5
                
            # Waste due to irrigation right before rain
            if actual_irrigation[i] > 0 and actual_rain > 0:
                waste_penalty += actual_irrigation[i] * 0.2
                
            self.history[i].append(self.true_soil_moistures[i])

        self.total_energy_penalty += energy_penalty
        self.total_water_penalty += water_penalty
        self.total_stress_penalty += stress_penalty
        self.total_waste_penalty += waste_penalty

        val = healthy_reward - energy_penalty - water_penalty - stress_penalty - waste_penalty
        self.cumulative_reward += val
        
        reward = Reward(
            value=val,
            energy_penalty=energy_penalty,
            water_penalty=water_penalty,
            stress_penalty=stress_penalty,
            waste_penalty=waste_penalty,
            healthy_reward=healthy_reward
        )
        
        done = self.step_count >= self.episode_length
        info = {"total_irrigation": sum(actual_irrigation)}
        
        self.upcoming_rain = random.random() < 0.2
        
        return self._get_obs(), reward, done, info

    def state(self) -> State:
        return State(
            step_count=self.step_count,
            true_soil_moistures=self.true_soil_moistures,
            upcoming_rain=self.upcoming_rain,
            remaining_daily_quota=self.remaining_daily_quota
        )

    def grade(self) -> Grade:
        max_possible = self.episode_length * 0.5 * self.num_plots
        score = max(0.0, self.cumulative_reward / max_possible) if max_possible > 0 else 0.0
        success = score >= 0.7
        
        analytics = {
            "Total Energy Penalty": f"-{self.total_energy_penalty:.2f} points (Peak hour usage)",
            "Total Water Penalty": f"-{self.total_water_penalty:.2f} points (Raw usage)",
            "Total Stress Penalty": f"-{self.total_stress_penalty:.2f} points (Under-watering)",
            "Total Waste Penalty": f"-{self.total_waste_penalty:.2f} points (Over-watering / Rain overlap)",
            "Maximum Possible Points": f"{max_possible:.2f}",
            "Achieved Points": f"{self.cumulative_reward:.2f}"
        }
        
        return Grade(score=score, success=success, analytics=analytics)
