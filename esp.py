import numpy as np
import gymnasium as gym
from network import FeedforwardNetwork

class ESPPopulation:
    """
    Реализация популяции для ESP: для каждого нейрона скрытого слоя — подпопуляция (особи = вектор весов входов + веса выхода)
    """

    def __init__(self, input_size, hidden_size, output_size, subpop_size=20, mutation_rate=0.1, crossover_rate=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.subpop_size = subpop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Подпопуляция для каждого скрытого нейрона: каждый — (input_size + output_size) весов
        self.subpopulations = [
            [self._random_individual() for _ in range(subpop_size)]
            for _ in range(hidden_size)
        ]

    def _random_individual(self):
        # Веса входов к скрытому нейрону + веса скрытого к каждому выходу
        return np.random.randn(self.input_size + self.output_size) * 0.1

    def assemble_network(self, hidden_indices):
        """
        Собрать сеть: по одному представителю из каждой подпопуляции (hidden_indices — индексы в каждой подпопуляции)
        """
        # Веса input-hidden
        w_ih = np.stack([self.subpopulations[i][hidden_indices[i]][:self.input_size] for i in range(self.hidden_size)])
        # Веса hidden-output
        w_ho = np.stack([self.subpopulations[i][hidden_indices[i]][self.input_size:] for i in range(self.hidden_size)]).T
        # w_ih: [hidden_size, input_size]
        # w_ho: [output_size, hidden_size]
        return FeedforwardNetwork(self.input_size, self.hidden_size, self.output_size, w_ih, w_ho)

    def evaluate(self, env, n_episodes=1, render=False):
        """
        Оценить всех особей в подпопуляциях
        """
        fitness = [np.zeros(self.subpop_size) for _ in range(self.hidden_size)]

        for trial in range(self.subpop_size):
            # Для каждого hidden-нейрона выбираем trial-индивида
            hidden_indices = [trial] * self.hidden_size
            network = self.assemble_network(hidden_indices)
            rewards = []
            for ep in range(n_episodes):
                obs, _ = env.reset()
                total_reward = 0
                done = False
                while not done:
                    action = network.forward(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    if render:
                        env.render()
                rewards.append(total_reward)
            for i in range(self.hidden_size):
                fitness[i][trial] = np.mean(rewards)
        return fitness

    def select(self, fitness, tournament_k=3):
        """
        Турнирный отбор для каждой подпопуляции
        """
        new_subpops = []
        for subpop, fit in zip(self.subpopulations, fitness):
            idxs = np.arange(self.subpop_size)
            selected = []
            for _ in range(self.subpop_size):
                tournament = np.random.choice(idxs, tournament_k, replace=False)
                best = tournament[np.argmax(fit[tournament])]
                selected.append(subpop[best].copy())
            new_subpops.append(selected)
        self.subpopulations = new_subpops

    def crossover_and_mutate(self):
        """
        Одноточечный кроссовер и мутация по Гауссу
        """
        for s, subpop in enumerate(self.subpopulations):
            # Кроссовер
            for i in range(0, self.subpop_size, 2):
                if np.random.rand() < self.crossover_rate:
                    a, b = subpop[i], subpop[(i+1) % self.subpop_size]
                    point = np.random.randint(1, len(a))
                    child1 = np.concatenate([a[:point], b[point:]])
                    child2 = np.concatenate([b[:point], a[point:]])
                    subpop[i] = child1
                    subpop[(i+1) % self.subpop_size] = child2
            # Мутация
            for i in range(self.subpop_size):
                if np.random.rand() < self.mutation_rate:
                    subpop[i] += np.random.randn(*subpop[i].shape) * 0.1

    def get_best_network(self):
        """
        Сеть из лучших особей
        """
        best_indices = [np.argmax([np.random.rand() for _ in subpop]) for subpop in self.subpopulations]
        return self.assemble_network(best_indices)

    def get_current_network(self):
        """
        Сеть из первых особей (для визуализации)
        """
        indices = [0 for _ in range(self.hidden_size)]
        return self.assemble_network(indices)
