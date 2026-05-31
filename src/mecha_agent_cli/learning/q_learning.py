"""Q-Learning implementation for mecha-agent-cli strategy selection.

State  : context_key strings produced by context.build_context_key()
         e.g. "actor_critic|qwen3:4b|new"
Action : arm_id strings from ARM_REGISTRY
         e.g. "direct.baseline", "repair.aggressive"

The Q-table maps (state, action) -> expected cumulative reward.
Actions are chosen with epsilon-greedy exploration; Q-values are
updated with the standard Bellman one-step TD rule.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QLearning:
    """Tabular Q-Learning agent for arm/strategy selection.

    Parameters
    ----------
    alpha:
        Learning rate (0, 1]. How much each new experience overwrites
        the existing Q-value estimate.  Default 0.1 is deliberately
        conservative so early noisy rewards don't corrupt the table.
    gamma:
        Discount factor [0, 1]. How much future rewards matter relative
        to the immediate reward.  Set close to 0 for purely myopic
        selection (each attempt is independent); close to 1 for
        long-horizon planning across many repair cycles.
    epsilon:
        Initial exploration rate [0, 1].  At epsilon=1 the agent picks
        a random arm every time; at epsilon=0 it always exploits the
        best known arm.
    epsilon_min:
        Floor for epsilon decay — we never stop exploring entirely.
    epsilon_decay:
        Multiplicative decay applied after every update so the agent
        gradually shifts from exploration to exploitation.
    """

    alpha: float = 0.1
    gamma: float = 0.6
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    _q_table: dict[str, dict[str, float]] = field(
        default_factory=dict, repr=False
    )

    def _ensure_state(self, state: str, actions: list[str]) -> None:
        """Initialise Q-values to 0 for any unseen (state, action) pair."""
        if state not in self._q_table:
            self._q_table[state] = {}
        for action in actions:
            if action not in self._q_table[state]:
                self._q_table[state][action] = 0.0

    def choose_action(self, state: str, actions: list[str]) -> str:
        """Return an action using epsilon-greedy selection.

        Parameters
        ----------
        state:
            Current context key, e.g. ``"actor_critic|qwen3:4b|new"``.
        actions:
            All available arm_ids the bandit can choose from.

        Returns
        -------
        str
            The chosen arm_id.
        """
        if not actions:
            raise ValueError("actions list must not be empty")

        self._ensure_state(state, actions)

        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = self._q_table[state]
        best_q = max(q_values[a] for a in actions)
        best_actions = [a for a in actions if q_values[a] == best_q]
        return random.choice(best_actions)

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_actions: list[str],
        *,
        done: bool = False,
    ) -> float:
        """Update the Q-table with one Bellman step and return the new Q-value.

        Bellman equation (one-step TD):

            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Parameters
        ----------
        state:
            The context_key before the attempt.
        action:
            The arm_id that was selected.
        reward:
            Scalar reward from ``episode_reward()``, clamped to [-1, 1].
        next_state:
            The context_key after the attempt (may differ if the task
            family or model changed, otherwise identical to ``state``).
        next_actions:
            Available arms in the next step (usually the full ARM_REGISTRY
            list).
        done:
            Set True if this was the final attempt in the episode so the
            future-value term is zeroed out.

        Returns
        -------
        float
            Updated Q(state, action).
        """
        self._ensure_state(state, [action])
        self._ensure_state(next_state, next_actions)

        current_q = self._q_table[state][action]

        if done or not next_actions:
            future_q = 0.0
        else:
            future_q = max(self._q_table[next_state][a] for a in next_actions)

        td_target = reward + self.gamma * future_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        self._q_table[state][action] = new_q

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return new_q

    def save(self, path: Path) -> None:
        """Serialise the Q-table and hyperparameters to a JSON file."""
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table": self._q_table,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "QLearning":
        """Restore a Q-table from a JSON file saved by ``save()``."""
        payload = json.loads(path.read_text())
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
        )
        agent._q_table = payload["q_table"]
        return agent

    def best_action(self, state: str, actions: list[str]) -> str | None:
        """Return the greedy-best action without side-effects or epsilon.

        Useful for inspection and unit tests.  Returns ``None`` if the
        state has never been seen.
        """
        if state not in self._q_table:
            return None
        self._ensure_state(state, actions)
        q_values = self._q_table[state]
        return max(actions, key=lambda a: q_values.get(a, 0.0))

    def q_value(self, state: str, action: str) -> float:
        """Return Q(state, action), defaulting to 0.0 if unseen."""
        return self._q_table.get(state, {}).get(action, 0.0)


@dataclass(frozen=True)
class QLearningPlaceholder:
    """Kept for backwards compatibility — use QLearning instead."""

    name: str = "q_learning_placeholder"
    status: str = "superseded_by_QLearning"


def placeholder_status() -> str:
    return "q_learning module is now implemented — see QLearning class"


__all__ = ["QLearning", "QLearningPlaceholder", "placeholder_status"]