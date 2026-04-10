"""
tasks.py – OpenEnv grader functions for the Smart Irrigation environment.

Each grader receives an episode_stats object emitted by the OpenEnv framework
after a full episode has run, and must return a TaskScore whose 'score' field
is strictly between 0 and 1 (exclusive).
"""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Public type expected by the OpenEnv framework
# ---------------------------------------------------------------------------

class TaskScore(BaseModel):
    """Single-field model returned by every grader."""
    score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPSILON = 1e-6  # keeps scores strictly in (0, 1)

def _clamp_open_interval(x: float) -> float:
    """
    Clamp *x* into [0.0, 1.0] then squeeze the boundaries inward by ε so
    the result is **strictly** between 0.0 and 1.0.
    """
    clamped = max(0.0, min(1.0, x))
    return max(_EPSILON, min(1.0 - _EPSILON, clamped))


def _score_from_stats(episode_stats: object, max_reward: float) -> float:
    """
    Derive a normalised raw score from the episode stats payload.
    Falls back gracefully if specific attributes are absent.
    """
    # Try attribute-style access first (dataclass / object), then dict-style.
    def _get(key: str, default: float = 0.0) -> float:
        if hasattr(episode_stats, key):
            return float(getattr(episode_stats, key))
        if isinstance(episode_stats, dict):
            return float(episode_stats.get(key, default))
        return default

    # OpenEnv environments often log the final reward under 'total_reward' or 'cumulative_reward'
    total_reward = _get("total_reward", _get("cumulative_reward", 0.0))
    raw = total_reward / max_reward if max_reward > 0 else 0.0
    return raw


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

def grade_task1_easy(episode_stats: object) -> TaskScore:
    """
    Grader for *task1_easy* (2 plots, corn crop, mild conditions).

    Max possible = 28 steps × 0.5 pts × 2 plots = 28.0 pts.
    """
    # Provide a reasonable total reward scale
    raw = _score_from_stats(episode_stats, max_reward=28.0)
    return TaskScore(score=_clamp_open_interval(raw))


def grade_task2_medium(episode_stats: object) -> TaskScore:
    """
    Grader for *task2_medium* (4 plots, mixed crops, pronounced tariffs).

    Max possible = 28 × 0.5 × 4 = 56.0 pts.
    """
    raw = _score_from_stats(episode_stats, max_reward=56.0)
    return TaskScore(score=_clamp_open_interval(raw))


def grade_task3_hard(episode_stats: object) -> TaskScore:
    """
    Grader for *task3_hard* (5 plots, diverse crops, uncertain weather).

    Max possible = 28 × 0.5 × 5 = 70.0 pts.
    """
    raw = _score_from_stats(episode_stats, max_reward=70.0)
    return TaskScore(score=_clamp_open_interval(raw))