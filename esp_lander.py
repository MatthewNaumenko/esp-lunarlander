import argparse
import os
import numpy as np
import gymnasium as gym
from esp import ESPPopulation
from visualizations import visualize_network, plot_metric, save_network_legend
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
    """
    Основная функция обучения. Реализована логика циклов согласно алгоритму 7.1–7.3:
      1. Инициализация ESPPopulation
      2. Для каждой эпохи:
         - evaluate
         - вычисление avg_fitness, best_fitness_current
         - проверка стагнации → burst_mutation или adapt_structure
         - select_and_breed
         - визуализация, запись графиков и т.п.
    """
    env = gym.make('LunarLanderContinuous-v3')

    pop = ESPPopulation(
        input_size=8,
        hidden_size=args.hidden_size,
        output_size=2,
        subpop_size=args.subpop_size,
        trials_per_individual=args.trials_per_individual if hasattr(args, 'trials_per_individual') else 10,
        alpha_cauchy=args.alpha_cauchy if hasattr(args, 'alpha_cauchy') else 1.0,
        stagnation_b=args.stagnation_b if hasattr(args, 'stagnation_b') else 20,
        mutation_rate=args.mutation_rate if hasattr(args, 'mutation_rate') else 0.1,
        crossover_rate=args.crossover_rate if hasattr(args, 'crossover_rate') else 0.5
    )

    reward_history = []
    loss_history = []

    os.makedirs(args.struct_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n=== Эпоха {epoch+1} ===")

        # оценка
        avg_fitness = pop.evaluate(env, n_episodes=args.episodes_per_eval, render=False)
        best_fitness_current = pop._compute_global_best_fitness_from_avg(avg_fitness)
        reward_history.append(best_fitness_current)  # «лучший reward»
        loss_history.append(-best_fitness_current)

        print(f"Лучший avg_fitness на эпохе {epoch+1}: {best_fitness_current:.3f}")

        # проверка стагнации
        pop.best_history.append(best_fitness_current)
        if len(pop.best_history) == pop.stagnation_b:
            if best_fitness_current <= pop.best_history[0]:
                pop.burst_counter += 1
                pop.burst_mutation()
                # Если вторая подряд burst-мутация без улучшения то сразу adapt_structure
                if pop.burst_counter >= 2:
                    print("Две подряд burst-мутации без улучшения → адаптация структуры")
                    pop.adapt_structure(env, n_episodes=args.episodes_per_eval)
                    pop.burst_counter = 0
                    pop.best_history.clear()
                    avg_fitness = pop.evaluate(env, n_episodes=args.episodes_per_eval, render=False)
                    pop.select_and_breed(avg_fitness)
                    continue
                else:
                    # первой burst-мутации очищаем историю, новые b поколения
                    pop.best_history.clear()
            else:
                # улучшение всё же произошло в последних b поколениях
                pop.burst_counter = 0

        # селекция, скрещивание, мутация
        pop.select_and_breed(avg_fitness)

        # визуализация структуры
        net_vis = pop.get_best_network()
        visualize_network(net_vis, f"{args.struct_dir}/epoch_{epoch+1:04d}.png")

        if (epoch + 1) % 10 == 0:
            net_for_gif = pop.get_current_network()
            record_landing_gif(net_for_gif, epoch + 1)

    best_network = pop.get_best_network()
    save_network(best_network, args.save_weights)

    plot_metric(reward_history, "Best Avg Fitness", os.path.join(args.struct_dir, "reward_curve.png"))
    plot_metric(loss_history, "Loss (-Fitness)", os.path.join(args.struct_dir, "loss_curve.png"))

    env.close()


def test(args):
    """
    Тестирование с улучшенным форматированием вывода и обработкой всех параметров
    """
    env = gym.make('LunarLanderContinuous-v3', render_mode='human')
    net = load_network(args.load_weights)

    for ep in range(args.test_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        # Физические параметры для анализа
        landing_analysis = {
            'final_position': (0, 0),
            'final_velocity': (0, 0),
            'final_angle': 0,
            'leg_contacts': [False, False],
            'main_engine_total': 0.0,
            'side_engine_total': 0.0
        }

        while not done:
            action = net.forward(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Текущее состояние
            x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact = obs

            # Аккумулируем расход топлива
            main_engine = max(0, action[0])  # Главный двигатель (0-1)
            side_engine = abs(action[1])  # Боковые двигатели (0-1)

            landing_analysis['main_engine_total'] += main_engine
            landing_analysis['side_engine_total'] += side_engine

            total_reward += reward
            step_count += 1

        # Фиксируем финальные параметры
        x, y, vx, vy, angle, _, leg1_contact, leg2_contact = obs
        landing_analysis.update({
            'final_position': (x, y),
            'final_velocity': (vx, vy),
            'final_angle': angle,
            'leg_contacts': [bool(leg1_contact), bool(leg2_contact)]
        })

        # Определение результата посадки
        if abs(vx) < 1 and abs(vy) < 1 and abs(angle) < 0.2:
            landing_status = "УСПЕШНАЯ"
            landing_bonus = 200
        else:
            landing_status = "НЕУДАЧНАЯ"
            landing_bonus = -100

        distance = np.sqrt(x ** 2 + y ** 2)

        output_lines = [
            f"\n=== Тест эпизода {ep + 1} ===",
            f"Общий результат: {total_reward:.2f} ({landing_status})",
            f"Шагов выполнено: {step_count}",
            "\nФизические параметры посадки:",
            f"  Позиция:       ({x:.4f}, {y:.4f})",
            f"  Расстояние:    {distance:.4f}",
            f"  Скорость:      (x={vx:.4f}, y={vy:.4f})",
            f"  Угол:          {angle:.4f} рад",
            f"  Контакт опор:  {'1' if leg1_contact else '0'}, {'1' if leg2_contact else '0'}",
            f"  Топливо:       главный: {landing_analysis['main_engine_total']:.4f}, боковой: {landing_analysis['side_engine_total']:.4f}",
            "\nКритерии успешной посадки:",
            f"  Горизонтальная скорость: {abs(vx):.4f} м/с {'✓' if abs(vx) < 1 else '✗'} (< 1 м/с)",
            f"  Вертикальная скорость:   {abs(vy):.4f} м/с {'✓' if abs(vy) < 1 else '✗'} (< 1 м/с)",
            f"  Угол наклона:            {abs(angle):.4f} рад {'✓' if abs(angle) < 0.2 else '✗'} (< 0.2 рад)",
            f"  Обе опоры:               {'✓' if leg1_contact and leg2_contact else '✗'}",
            f"  Бонус за посадку:        {landing_bonus} (фактически учтен в общем вознаграждении)"
        ]

        for line in output_lines:
            print(line)

    env.close()

def visualize(args):
    net = load_network(args.load_weights)
    visualize_network(net, args.outfile)
    save_network_legend("network_legend.png")
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
