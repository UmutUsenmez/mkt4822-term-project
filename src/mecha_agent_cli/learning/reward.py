from __future__ import annotations

from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.schema import LearningConfig

def episode_reward(result: AgentRunResult, config: LearningConfig) -> float:
    reward = 0.0
    
    if result.status == "success":
        reward += config.success_reward
    else:
        passed_checks = sum(1 for res in result.validation_report.results if res.passed)
        reward += passed_checks * config.progress_bonus

    if result.total_attempts > 1:
        extra_attempts = result.total_attempts - 1
        reward -= extra_attempts * config.attempt_penalty

    failed_names = [res.name for res in result.validation_report.results if not res.passed]
    
    if "extract" in failed_names:
        reward -= config.extract_failure_penalty
        
    if "behavior" in failed_names:
        reward -= config.behavior_failure_penalty

    if result.duration_sec > 1.0:
        penalty_ratio = 1.0 - (1.0 / min(result.duration_sec, config.latency_horizon_sec))
        reward -= penalty_ratio * config.latency_penalty

    return max(-1.0, min(1.0, reward))


def reward_to_beta_update(reward: float) -> float:
    """Map reward in ``[-1, 1]`` to a pseudo-success in ``[0, 1]``."""
    return max(0.0, min(1.0, (reward + 1.0) / 2.0))


__all__ = ["episode_reward", "reward_to_beta_update"]
