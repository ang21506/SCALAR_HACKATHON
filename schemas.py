from pydantic import BaseModel, ConfigDict
from typing import List, Dict

class Observation(BaseModel):
    model_config = ConfigDict(strict=True)
    soil_moistures: List[float]
    crop_types: List[str]
    growth_stages: List[float]
    rain_probability: float
    expected_rainfall: float
    tariff_band: str
    daily_water_quota: float
    pump_capacity: float

class Action(BaseModel):
    model_config = ConfigDict(strict=True)
    irrigation_amounts: List[int]

class Reward(BaseModel):
    model_config = ConfigDict(strict=True)
    value: float
    energy_penalty: float
    water_penalty: float
    stress_penalty: float
    waste_penalty: float
    healthy_reward: float

class State(BaseModel):
    model_config = ConfigDict(strict=True)
    step_count: int
    true_soil_moistures: List[float]
    upcoming_rain: bool
    remaining_daily_quota: float

class Grade(BaseModel):
    model_config = ConfigDict(strict=True)
    score: float
    success: bool
    analytics: Dict[str, str]
