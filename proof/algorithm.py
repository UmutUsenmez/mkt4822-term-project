import random
import sys
import os
import csv
from typing import List, Dict, Tuple, Optional

class SimpleEnvironment:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.state = (0.0, 0.0)  # (x, v)
        self.max_steps = 100

    def reset(self) -> Tuple[float, float]:
        self.state = (0.0, 0.0)
        return self.state

    def step(self, action: int) -> Tuple[Tuple[float, float], float, bool]:
        x, v = self.state
        if action == 0:
            x -= 0.1
        else:
            x += 0.1
        v += 0.01
        self.state = (x, v)
        done = (x**2 + v**2 > 1) or (self.max_steps <= 0)
        self.max_steps -= 1
        return self.state, 1.0, done

class ActorCritic:
    def __init__(self, env: SimpleEnvironment):
        self.env = env
        self.actor_threshold = 0.5  # threshold for action selection
        self.critic_values = {}  # state -> value

    def select_action(self, state: Tuple[float, float]) -> int:
        x = state[0]
        return 1 if x > self.actor_threshold else 0

    def update_critic(self, state: Tuple[float, float], reward: float, next_state: Tuple[float, float]):
        next_value = self.critic_values.get(next_state, 0.0)
        self.critic_values[state] = reward + 0.9 * next_value

    def update_actor(self, state: Tuple[float, float], action: int, next_state: Tuple[float, float], reward: float):
        pass

def train_actor_critic(env: SimpleEnvironment, episodes: int = 10) -> Tuple[ActorCritic, List[float]]:
    actor_critic = ActorCritic(env)
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = actor_critic.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            actor_critic.update_critic(state, reward, next_state)
            state = next_state
        rewards.append(total_reward)
    return actor_critic, rewards

def evaluate_actor_critic(env: SimpleEnvironment, actor_critic: ActorCritic, episodes: int = 1) -> float:
    total_reward = 0
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = actor_critic.select_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    return total_reward / episodes

def generate_csv_report(actor_critic: ActorCritic, rewards: List[float], eval_reward: float):
    with open('generation_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'training_reward', 'evaluation_reward'])
        for episode, reward in enumerate(rewards, start=1):
            writer.writerow([episode, reward, eval_reward])

if __name__ == "__main__":
    random.seed(42)
    env = SimpleEnvironment()
    actor_critic, rewards = train_actor_critic(env, episodes=10)
    eval_reward = evaluate_actor_critic(env, actor_critic, episodes=1)
    generate_csv_report(actor_critic, rewards, eval_reward)
    print(f"Training completed. Average training reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Evaluation reward: {eval_reward:.2f}")
