from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mecha_agent_cli.config.schema import LearningConfig
from mecha_agent_cli.learning.arm_registry import ARM_REGISTRY, Arm, get_arm
from mecha_agent_cli.learning.reward import reward_to_beta_update

@dataclass(frozen=True)
class ArmStat:
    context_key: str
    arm_id: str
    alpha: float
    beta: float
    pulls: int
    cumulative_reward: float
    last_reward: float
    last_success: bool

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

class BanditStore:
    def __init__(self, db_path: Path | None) -> None:
        self.db_path = db_path
        self._rows: dict[tuple[str, str], ArmStat] = {}
        if self.db_path:
            self._init_db()

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_state (
                    context_key TEXT,
                    arm_id TEXT,
                    alpha REAL,
                    beta REAL,
                    pulls INTEGER,
                    cumulative_reward REAL,
                    last_reward REAL,
                    last_success BOOLEAN,
                    PRIMARY KEY (context_key, arm_id)
                )
            """)

    def fetch(self, context_key: str) -> dict[str, ArmStat]:
        out: dict[str, ArmStat] = {}
        if not self.db_path:
            for (ctx, arm_id), stat in self._rows.items():
                if ctx == context_key:
                    out[arm_id] = stat
            return out

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT context_key, arm_id, alpha, beta, pulls, cumulative_reward, last_reward, last_success FROM bandit_state WHERE context_key = ?",
                (context_key,)
            )
            for row in cursor:
                out[row[1]] = ArmStat(
                    context_key=row[0], arm_id=row[1], alpha=row[2], beta=row[3],
                    pulls=row[4], cumulative_reward=row[5], last_reward=row[6], last_success=bool(row[7])
                )
        return out

    def upsert(self, stat: ArmStat) -> None:
        if not self.db_path:
            self._rows[(stat.context_key, stat.arm_id)] = stat
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO bandit_state (context_key, arm_id, alpha, beta, pulls, cumulative_reward, last_reward, last_success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(context_key, arm_id) DO UPDATE SET
                    alpha=excluded.alpha, beta=excluded.beta, pulls=excluded.pulls,
                    cumulative_reward=excluded.cumulative_reward, last_reward=excluded.last_reward, last_success=excluded.last_success
            """, (stat.context_key, stat.arm_id, stat.alpha, stat.beta, stat.pulls, stat.cumulative_reward, stat.last_reward, stat.last_success))

    def all_rows(self) -> list[ArmStat]:
        if not self.db_path:
            return list(self._rows.values())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM bandit_state")
            return [ArmStat(row[0], row[1], row[2], row[3], row[4], row[5], row[6], bool(row[7])) for row in cursor]


class ThompsonBandit:
    BASELINE_ARM_ID = "direct.baseline"

    def __init__(self, store: BanditStore, cfg: LearningConfig, *, arms: tuple[Arm, ...] = ARM_REGISTRY, rng: random.Random | None = None) -> None:
        self.store = store
        self.cfg = cfg
        self.arms = arms
        self._arm_ids = tuple(a.arm_id for a in arms)
        self.rng = rng or random.Random()

    def select(self, context_key: str) -> Arm:
        if not self.cfg.enabled or self.cfg.mode == "off":
            return get_arm(self.BASELINE_ARM_ID)

        stats = self.store.fetch(context_key)
        
        # Keşif (Exploration) Aşaması: Yeterince denenmemiş kolları zorla seç
        under_pulled = [arm_id for arm_id in self._arm_ids if stats.get(arm_id, _zero_stat(context_key, arm_id)).pulls < self.cfg.min_pulls_before_exploit]
        if under_pulled:
            return get_arm(self.rng.choice(under_pulled))

        # Kullanım (Exploitation) Aşaması: Beta dağılımından örneklem al (Thompson Sampling)
        best_arm = self.BASELINE_ARM_ID
        max_sample = -1.0
        for arm_id in self._arm_ids:
            stat = stats.get(arm_id, _zero_stat(context_key, arm_id))
            sample = self.rng.betavariate(stat.alpha, stat.beta)
            if sample > max_sample:
                max_sample = sample
                best_arm = arm_id

        return get_arm(best_arm)

    def update(self, *, context_key: str, arm_id: str, reward: float, success: bool) -> ArmStat:
        prior = self.store.fetch(context_key).get(arm_id, _zero_stat(context_key, arm_id))
        
        y = reward_to_beta_update(reward)
        decay = self.cfg.decay_factor

        new_alpha = 1.0 + decay * (prior.alpha - 1.0) + y
        new_beta = 1.0 + decay * (prior.beta - 1.0) + (1.0 - y)

        new = ArmStat(
            context_key=context_key,
            arm_id=arm_id,
            alpha=new_alpha,
            beta=new_beta,
            pulls=prior.pulls + 1,
            cumulative_reward=prior.cumulative_reward + reward,
            last_reward=reward,
            last_success=success,
        )
        self.store.upsert(new)
        return new


def _zero_stat(context_key: str, arm_id: str) -> ArmStat:
    return ArmStat(context_key, arm_id, 1.0, 1.0, 0, 0.0, 0.0, False)


__all__ = ["ArmStat", "BanditStore", "ThompsonBandit"]