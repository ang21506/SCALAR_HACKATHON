from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import SmartIrrigationEnv
from schemas import Action


app = FastAPI(title="Smart Irrigation OpenEnv Server")

# Keep one active environment instance for the current evaluator session.
_ACTIVE_ENV: Optional[SmartIrrigationEnv] = None

_TASK_SPECS = [
    {
        "id": "task1_easy",
        "name": "Task 1 Easy",
        "description": "Single crop, identical soil across 2 plots, simple tariff schedule, relatively accurate weather forecast.",
        "difficulty": "easy",
        "grader": "openenv/grade",
    },
    {
        "id": "task2_medium",
        "name": "Task 2 Medium",
        "description": "Mixed crops spread across 4 plots, tighter water quotas, and pronounced peak/off-peak tariff differences.",
        "difficulty": "medium",
        "grader": "openenv/grade",
    },
    {
        "id": "task3_hard",
        "name": "Task 3 Hard",
        "description": "5 diverse plots, highly uncertain rainfall, strict water limits, and complex tariffs.",
        "difficulty": "hard",
        "grader": "openenv/grade",
    },
]


class ResetRequest(BaseModel):
    task: Optional[str] = None
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    irrigation_amounts: Optional[list[int]] = None
    action: Optional[Dict[str, Any]] = None


def _resolve_task(payload: Optional[ResetRequest]) -> str:
    if payload is None:
        return "task1_easy"
    return payload.task_id or payload.task or "task1_easy"


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/openenv/tasks")
@app.get("/tasks", include_in_schema=False)
@app.get("/env/tasks", include_in_schema=False)
def openenv_tasks() -> Dict[str, Any]:
    return {"tasks": _TASK_SPECS}


@app.post("/openenv/tasks")
@app.post("/tasks", include_in_schema=False)
@app.post("/env/tasks", include_in_schema=False)
def openenv_tasks_post() -> Dict[str, Any]:
    return openenv_tasks()


@app.post("/openenv/reset")
@app.post("/reset", include_in_schema=False)
@app.post("/env/reset", include_in_schema=False)
def openenv_reset(payload: Optional[ResetRequest] = None) -> Dict[str, Any]:
    global _ACTIVE_ENV
    task = _resolve_task(payload)
    _ACTIVE_ENV = SmartIrrigationEnv(task=task)
    obs = _ACTIVE_ENV.reset()
    obs_dict = obs.model_dump()
    return {"observation": obs_dict, "obs": obs_dict}


@app.post("/openenv/step")
@app.post("/step", include_in_schema=False)
@app.post("/env/step", include_in_schema=False)
def openenv_step(payload: Optional[StepRequest] = None) -> Dict[str, Any]:
    if _ACTIVE_ENV is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized. Call /openenv/reset first.")

    action_dict: Dict[str, Any]
    if payload is None:
        action_dict = {"irrigation_amounts": [0] * _ACTIVE_ENV.num_plots}
    elif payload.action is not None:
        action_dict = payload.action
    elif payload.irrigation_amounts is not None:
        action_dict = {"irrigation_amounts": payload.irrigation_amounts}
    else:
        action_dict = {"irrigation_amounts": [0] * _ACTIVE_ENV.num_plots}

    action = Action(**action_dict)
    obs, reward, done, info = _ACTIVE_ENV.step(action)
    obs_dict = obs.model_dump()
    reward_dict = reward.model_dump()
    return {
        "observation": obs_dict,
        "obs": obs_dict,
        "reward": reward_dict,
        "done": done,
        "info": info,
    }


@app.get("/openenv/state")
@app.get("/state", include_in_schema=False)
@app.get("/env/state", include_in_schema=False)
def openenv_state() -> Dict[str, Any]:
    if _ACTIVE_ENV is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized. Call /openenv/reset first.")
    state_dict = _ACTIVE_ENV.state().model_dump()
    return {"state": state_dict}


@app.post("/openenv/state")
@app.post("/state", include_in_schema=False)
@app.post("/env/state", include_in_schema=False)
def openenv_state_post() -> Dict[str, Any]:
    return openenv_state()


@app.get("/openenv/grade")
@app.get("/grade", include_in_schema=False)
@app.get("/env/grade", include_in_schema=False)
def openenv_grade() -> Dict[str, Any]:
    if _ACTIVE_ENV is None:
        raise HTTPException(status_code=400, detail="Environment is not initialized. Call /openenv/reset first.")
    grade_dict = _ACTIVE_ENV.grade().model_dump()
    # Return both nested and top-level fields for compatibility across evaluators.
    return {"grade": grade_dict, **grade_dict}


@app.post("/openenv/grade")
@app.post("/grade", include_in_schema=False)
@app.post("/env/grade", include_in_schema=False)
def openenv_grade_post() -> Dict[str, Any]:
    return openenv_grade()


@app.post("/openenv/close")
@app.post("/close", include_in_schema=False)
@app.post("/env/close", include_in_schema=False)
def openenv_close() -> Dict[str, Any]:
    global _ACTIVE_ENV
    if _ACTIVE_ENV is not None:
        _ACTIVE_ENV.close()
        _ACTIVE_ENV = None
    return {"closed": True}


@app.get("/openenv/close")
@app.get("/close", include_in_schema=False)
@app.get("/env/close", include_in_schema=False)
def openenv_close_get() -> Dict[str, Any]:
    return openenv_close()
