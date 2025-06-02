import argparse
import os
import numpy as np
import gymnasium as gym
from esp import ESPPopulation
from visualizations import visualize_network, plot_metric
from utils import load_network

def train(args):
    env = gym.make('LunarLanderContinuous-v3')
    pop = ESPPopulation(
        input_size=8,
        hidden_size=args.hidden_size,
        output_size=2,
        subpop_size=args.subpop_size
    )
    reward_history = []
    loss_history = []
    os.makedirs(args.struct_dir, exist_ok=True)
    for epoch in range(args.epochs):
        # Оценка (по одному эпизоду для каждого индивида)
        fitness = pop.evaluate(env, n_episodes=args.episodes_per_eval)
        # Средний фитнес по всем особям (прокси: по первой подпопуляции)
        mean_reward = np.mean(fitness[0])
        reward_history.append(mean_reward)
        # В данном случае loss = -reward
        loss = -mean_reward
        loss_history.append(loss)
        print(f"Epoch {epoch+1}: Mean reward {mean_reward:.2f}, loss {loss:.2f}")
        # Визуализация структуры сети
        net = pop.get_current_network()
        visualize_network(net, f"{args.struct_dir}/epoch_{epoch+1:04d}.png")
        # Эволюция
        pop.select(fitness)
        pop.crossover_and_mutate()

    # Графики
    plot_metric(reward_history, "Mean Reward", os.path.join(args.struct_dir, "reward_curve.png"))
    plot_metric(loss_history, "Loss", os.path.join(args.struct_dir, "loss_curve.png"))
    env.close()