import argparse
import os
import numpy as np
import gymnasium as gym
from esp import ESPPopulation
from visualizations import visualize_network, plot_metric
from utils import save_network, load_network

def record_landing_gif(network, epoch, video_dir="videos"):
    import os
    os.makedirs(video_dir, exist_ok=True)
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array_list")  # для новых gymnasium
    obs, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        while isinstance(frame, list) or (isinstance(frame, np.ndarray) and frame.ndim > 3):
            frame = frame[0]
        frames.append(frame)
        action = network.forward(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    import imageio
    gif_path = f"{video_dir}/lander_epoch_{epoch+1:04d}.gif"
    imageio.mimsave(gif_path, [frame for frame in frames], fps=30)
    print(f"Saved landing gif: {gif_path}")

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
        if (epoch + 1) % 50 == 0:
            net = pop.get_current_network()
            record_landing_gif(net, epoch)
    # Сохранить веса
    save_network(pop.get_best_network(), args.save_weights)
    # Графики
    plot_metric(reward_history, "Mean Reward", os.path.join(args.struct_dir, "reward_curve.png"))
    plot_metric(loss_history, "Loss", os.path.join(args.struct_dir, "loss_curve.png"))
    env.close()

def test(args):
    env = gym.make('LunarLanderContinuous-v3', render_mode='human')
    net = load_network(args.load_weights)
    for ep in range(args.test_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = net.forward(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Test episode {ep+1}, reward: {total_reward:.2f}")
    env.close()

def visualize(args):
    net = load_network(args.load_weights)
    visualize_network(net, args.outfile)
    print(f"Network structure saved as {args.outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP for LunarLanderContinuous-v3")
    parser.add_argument("--train", action="store_true", help="Train ESP")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=12)
    parser.add_argument("--subpop_size", type=int, default=20)
    parser.add_argument("--episodes_per_eval", type=int, default=1)
    parser.add_argument("--struct_dir", type=str, default="structures")
    parser.add_argument("--save_weights", type=str, default="model.pkl")
    parser.add_argument("--load_weights", type=str, default="model.pkl")
    parser.add_argument("--test", action="store_true", help="Test ESP")
    parser.add_argument("--test_episodes", type=int, default=5)
    parser.add_argument("--visualize_structure", action="store_true")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--outfile", type=str, default="network.png")
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.test:
        test(args)
    elif args.visualize_structure:
        visualize(args)
    else:
        print("No mode specified. Use --train, --test or --visualize_structure.")
