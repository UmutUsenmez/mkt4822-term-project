from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mecha_agent_cli.config.schema import ModelProfile

@dataclass(frozen=True)
class Arm:
    arm_id: str
    profile_name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def apply(self, base: ModelProfile) -> ModelProfile:
        for key in self.overrides:
            if not hasattr(base, key):
                raise ValueError(f"Field {key} not found in ModelProfile")
        return base.model_copy(update=self.overrides)

ARM_REGISTRY: tuple[Arm, ...] = (
    Arm(
        arm_id="direct.baseline",
        profile_name="direct",
        overrides={},
        description="Baseline strategy.",
    ),
    Arm(
        arm_id="direct.cool",
        profile_name="direct",
        overrides={"temperature": 0.2, "top_p": 0.85},
        description="Low temperature strategy for deterministic output.",
    ),
)

_BY_ID: dict[str, Arm] = {arm.arm_id: arm for arm in ARM_REGISTRY}

def get_arm(arm_id: str) -> Arm:
    if arm_id not in _BY_ID:
        msg = f"Unknown arm_id: {arm_id!r}"
        raise KeyError(msg)
    return _BY_ID[arm_id]

def list_arm_ids() -> list[str]:
    return [arm.arm_id for arm in ARM_REGISTRY]

__all__ = ["ARM_REGISTRY", "Arm", "get_arm", "list_arm_ids"]