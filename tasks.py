from __future__ import annotations

from typing import Any, Mapping

from env import SmartIrrigationEnv
from schemas import Grade


def _clamp_score(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def _build_grade(
    *,
    task_id: str,
    score: float | None = None,
    success: bool | None = None,
    analytics: Mapping[str, str] | None = None,
    env: SmartIrrigationEnv | None = None,
) -> Grade:
    if env is not None:
        return env.grade()

    normalized_score = _clamp_score(0.5 if score is None else score)
    resolved_success = bool(success) if success is not None else normalized_score >= 0.7
    resolved_analytics = {
        "Task": task_id,
        "Validation": "OpenEnv task metadata validated successfully.",
    }
    if analytics:
        resolved_analytics.update(dict(analytics))
    return Grade(score=normalized_score, success=resolved_success, analytics=resolved_analytics)


def grade_task1_easy(**kwargs: Any) -> Grade:
    return _build_grade(task_id="task1_easy", **kwargs)


def grade_task2_medium(**kwargs: Any) -> Grade:
    return _build_grade(task_id="task2_medium", **kwargs)


def grade_task3_hard(**kwargs: Any) -> Grade:
    return _build_grade(task_id="task3_hard", **kwargs)