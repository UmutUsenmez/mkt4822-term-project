from typing import Dict, List, Tuple, Optional
import random

class SimpleMarketEnvironment:
    def __init__(self, initial_state: int = 0):
        self.state = initial_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Take an action and return next state, reward, done."""
        if action == 0:
            next_state = self.state + 1
        else:
            next_state = self.state
        reward = 1.0 if next_state == 3 else 0.0
        done = next_state == 3
        return next_state, reward, done

def actor_critic(
    env: SimpleMarketEnvironment,
    num_episodes: int = 100,
    gamma: float = 0.9,
    lr: float = 0.01,
) -> None:
    """
    Train an actor-critic model on a simple market environment.

    Args:
        env: The environment instance.
        num_episodes: Number of episodes to run.
        gamma: Discount factor.
        lr: Learning rate for updates.
    """
    policy: Dict[int, Dict[int, float]] = {state: {action: 0.5 for action in [0, 1]} for state in range(5)}
    value: Dict[int, float] = {state: 0.0 for state in range(5)}

    for episode in range(num_episodes):
        state = env.state
        done = False
        while not done:
            action_probs = policy[state]
            action = 0 if random.random() < action_probs[0] else 1
            next_state, reward, done = env.step(action)
            td_error = reward + gamma * value[next_state] - value[state]
            value[state] += lr * td_error
            policy[state][action] += lr * td_error * (action_probs[action] - 1)
            state = next_state

    print(f"Trained for {num_episodes} episodes. Final value: {value}")

if __name__ == "__main__":
    env = SimpleMarketEnvironment()
    actor_critic(env, num_episodes=2)
