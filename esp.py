import numpy as np
import gymnasium as gym
import os
from network import FeedforwardNetwork  # предполагаемая ваша реализация сети
from collections import deque

class ESPPopulation:
    """
    Полная реализация ESP (Enforced Sub-Populations) согласно алгоритмам 7.1–7.3.
    Для каждого скрытого нейрона — отдельная подпопуляция (особи = вектор весов input->hidden + hidden->output).
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 subpop_size: int = 20,
                 trials_per_individual: int = 10,
                 alpha_cauchy: float = 1.0,
                 stagnation_b: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.5):
        """
        :param input_size: число входов
        :param hidden_size: начальное число скрытых нейронов (количество подпопуляций)
        :param output_size: число выходов
        :param subpop_size: размер каждой подпопуляции (n)
        :param trials_per_individual: число испытаний, в которых должен участвовать каждый нейрон (min_trials = 10)
        :param alpha_cauchy: масштаб параметра α для Коши-мутации
        :param stagnation_b: число поколений без улучшения, после которого происходит burst-мутация
        :param mutation_rate: вероятность Gaussian-мутации (используется лишь для доп мутации)
        :param crossover_rate: вероятность выполнения кроссовера при подборе попарно в лучших 1/4
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.subpop_size = subpop_size
        self.trials_per_individual = trials_per_individual
        self.alpha_cauchy = alpha_cauchy
        self.stagnation_b = stagnation_b
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Создаём подпопуляции: список длины hidden_size, каждая – list из subpop_size векторов
        # В каждом векторе: first input_size элементов – веса input->hidden, далее output_size элементов – hidden->output.
        self.subpopulations = [
            [self._random_individual() for _ in range(subpop_size)]
            for _ in range(hidden_size)
        ]

        # Для оценки: кумулятивная приспособленность и счётчик trials для каждого нейрона
        # cum_fitness[i][j] — суммарная приспособленность j-го индивида i-й подпопуляции
        # count_trials[i][j] — число trials, в которых j-й индивид i-й подпопуляции участвовал
        self.cum_fitness = [
            np.zeros(subpop_size, dtype=np.float64)
            for _ in range(hidden_size)
        ]
        self.count_trials = [
            np.zeros(subpop_size, dtype=np.int32)
            for _ in range(hidden_size)
        ]

        # История лучших значений (deque длины stagnation_b+1 для удобства)
        self.best_history = deque(maxlen=stagnation_b)
        # Счётчик, сколько раз подряд мы делали burst_mutation без улучшения
        self.burst_counter = 0

    def _random_individual(self) -> np.ndarray:
        """
        Создаёт случайного «нейрона» (индида) с небольшими весами.
        Вектор длины input_size + output_size.
        """
        return np.random.randn(self.input_size + self.output_size) * 0.1

    def assemble_network(self, hidden_indices: list[int]) -> FeedforwardNetwork:
        """
        Собрать полную сеть, беря для каждого скрытого нейрона свой вектор весов.
        :param hidden_indices: список длины hidden_size, где hidden_indices[i] = индекс индивида из i-й подпопуляции
        :return: экземпляр FeedforwardNetwork с заданными матрицами весов
        """
        # Веса input->hidden: матрица [hidden_size, input_size]
        w_ih = np.stack([
            self.subpopulations[i][hidden_indices[i]][:self.input_size]
            for i in range(self.hidden_size)
        ])

        w_ho = np.stack([
            self.subpopulations[i][hidden_indices[i]][self.input_size:]
            for i in range(self.hidden_size)
        ]).T
        return FeedforwardNetwork(self.input_size, self.hidden_size, self.output_size, w_ih, w_ho)

    def evaluate(self, env: gym.Env, n_episodes: int = 1, render: bool = False):
        """
        Оцениваем всех особей в подпопуляциях. Каждый индивид каждого скрытого нейрона участвует в trials_per_individual trials.
        В каждом trial для данного индивида мы формируем случайный набор соседей из других субпопуляций.
        Приспособленность (fitness) аккумулируется в cum_fitness и count_trials.
        :param env: среда Gym
        :param n_episodes: число эпизодов для каждого испытания (обычно =1)
        :param render: визуализировать ли игру
        """
        # Сбрасываем текущие кумулятивные значения перед новой оценкой
        for i in range(self.hidden_size):
            self.cum_fitness[i].fill(0.0)
            self.count_trials[i].fill(0)

        # Для каждого скрытого нейрона i и каждого индивида j проводим trials_per_individual trials
        for i in range(self.hidden_size):
            for j in range(self.subpop_size):
                for t in range(self.trials_per_individual):
                    # Формируем список hidden_indices: для i-й подпопуляции фиксируем j,
                    # для остальных – случайные индексы от 0 до subpop_size-1
                    hidden_indices = []
                    for k in range(self.hidden_size):
                        if k == i:
                            hidden_indices.append(j)
                        else:
                            hidden_indices.append(np.random.randint(0, self.subpop_size))
                    # Собираем сеть
                    network = self.assemble_network(hidden_indices)

                    # Оцениваем сеть: запускаем n_episodes эпизодов, усредняем награды
                    total_rewards = []
                    for ep in range(n_episodes):
                        obs, _ = env.reset()
                        done = False
                        episode_reward = 0.0
                        while not done:
                            action = network.forward(obs)
                            obs, reward, terminated, truncated, _ = env.step(action)
                            episode_reward += reward
                            done = terminated or truncated
                            if render:
                                env.render()
                        total_rewards.append(episode_reward)
                    avg_reward = np.mean(total_rewards)

                    # Накопим приспособленность для данной подпопуляции и индивида
                    self.cum_fitness[i][j] += avg_reward
                    self.count_trials[i][j] += 1

        # После всех trials_per_individual trials для каждого индивида возвращаем среднюю приспособленность
        avg_fitness = []
        for i in range(self.hidden_size):
            avg = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            avg_fitness.append(avg)
        return avg_fitness

    def select_and_breed(self, avg_fitness: list[np.ndarray]):
        """
        Селекция + кроссовер + мутация (алгоритм 7.1):
          1. Сортировка по убыванию avg_fitness.
          2. Скрещиваем 1/4 лучших (one-point), создаём детей.
          3. Убираем из старой популяции точное число худших, равное числу детей, и добавляем детей.
          4. Коши-мутация “нижней” половины.
          5. Дополнительная Gaussian-мутация.
        """
        for i in range(self.hidden_size):
            subpop = self.subpopulations[i]
            fitness_i = avg_fitness[i]

            # 1) сортировка индексов по убыванию средней приспособленности
            sorted_idxs = np.argsort(-fitness_i)
            subpop_sorted = [subpop[idx].copy() for idx in sorted_idxs]

            # 2) скрещиваем 1/4 лучших
            top_k = max(1, self.subpop_size // 4)
            parents = subpop_sorted[:top_k]
            children = []
            for idx in range(0, top_k - 1, 2):
                if np.random.rand() < self.crossover_rate:
                    a, b = parents[idx], parents[idx + 1]
                    point = np.random.randint(1, len(a))
                    children.append(np.concatenate([a[:point], b[point:]]))
                    children.append(np.concatenate([b[:point], a[point:]]))
                else:
                    children.append(parents[idx].copy())
                    children.append(parents[idx + 1].copy())
            if top_k % 2 == 1:  # нечётный родитель — просто клон
                children.append(parents[-1].copy())

            # 3) Усечение: убираем из subpop_sorted ровно len(children) худших
            m = len(children)
            keep_count = max(0, self.subpop_size - m)
            retained = subpop_sorted[:keep_count]
            # Объединяем оставшихся и детей
            subpop_new = retained + children

            # 4) Коши-мутация нижней половины
            half = self.subpop_size // 2
            for idx in range(half, self.subpop_size):
                perturb = self.alpha_cauchy * np.random.standard_cauchy(size=subpop_new[idx].shape)
                subpop_new[idx] += perturb

            # 5) Дополнительная Gaussian-мутация
            for idx in range(self.subpop_size):
                if np.random.rand() < self.mutation_rate:
                    subpop_new[idx] += np.random.randn(*subpop_new[idx].shape) * 0.01

            self.subpopulations[i] = subpop_new

    def burst_mutation(self):
        """
        Алгоритм 7.2: «взрывная» мутация (burst mutation).
        Для каждой подпопуляции:
          1. Находим лучшую особь (по текущему среднему фитнесу).
          2. Формируем новую подпопуляцию вокруг этой лучшей особи, добавляя Cauchy-поправки.
        """
        print("=== BURST MUTATION ===")
        for i in range(self.hidden_size):
            # Находим индекс лучшей особи в i-й подпопуляции, по среднему фитнесу
            avg_i = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            best_idx = int(np.argmax(avg_i))
            best_vector = self.subpopulations[i][best_idx]

            # Формируем новую подпопуляцию вокруг best_vector
            new_subpop = []
            for _ in range(self.subpop_size):
                perturb = self.alpha_cauchy * np.random.standard_cauchy(size=best_vector.shape)
                new_ind = best_vector + perturb
                new_subpop.append(new_ind)
            self.subpopulations[i] = new_subpop

        # После burst-мутации обнуляем кумулятивные данные
        for i in range(self.hidden_size):
            self.cum_fitness[i].fill(0.0)
            self.count_trials[i].fill(0)

    def adapt_structure(self, env: gym.Env, n_episodes: int = 1):
        """
        Алгоритм 7.3 с возможностью удалить несколько подпопуляций за один проход.
        Если ни одна не удалена — добавляется новая.
        """
        print("=== ADAPT STRUCTURE ===")

        removed_any = True
        while removed_any:
            removed_any = False
            current_best = self._compute_global_best_fitness()

            # Копируем существующие подпопуляции для безопасного перебора
            old_subpops = [list(sp) for sp in self.subpopulations]
            old_hidden = self.hidden_size

            for i in range(old_hidden):
                # Временная модель без i-й подпопуляции
                tmp_pops = [old_subpops[k] for k in range(old_hidden) if k != i]
                tmp_hidden = old_hidden - 1

                tmp = ESPPopulation(
                    self.input_size, tmp_hidden, self.output_size,
                    self.subpop_size, self.trials_per_individual,
                    self.alpha_cauchy, self.stagnation_b,
                    self.mutation_rate, self.crossover_rate
                )
                tmp.subpopulations = [[ind.copy() for ind in sp] for sp in tmp_pops]

                best_tmp = tmp._compute_global_best_fitness_from_avg(
                    tmp.evaluate(env, n_episodes=n_episodes)
                )

                if best_tmp > current_best:
                    print(f"Удаляем подпопуляцию {i}: {current_best:.3f} → {best_tmp:.3f}")
                    self.subpopulations = tmp.subpopulations
                    self.hidden_size = tmp_hidden
                    removed_any = True
                    break  # Начинаем новый проход после удаления

        if not removed_any:  # Ничего не удалили → добавляем новую подпопуляцию
            print("Ни одна подпопуляция не удалена, добавляем новую")
            self.subpopulations.append([self._random_individual()
                                        for _ in range(self.subpop_size)])
            self.hidden_size += 1

        # Сброс статистики
        self.cum_fitness = [np.zeros(self.subpop_size) for _ in range(self.hidden_size)]
        self.count_trials = [np.zeros(self.subpop_size, dtype=np.int32) for _ in range(self.hidden_size)]

    def _compute_global_best_fitness(self) -> float:
        """
        Вспомогательная функция: по текущим cum_fitness и count_trials возвращает
        глобально лучшее значение avg_fitness среди всех индивидов.
        """
        best_value = -np.inf
        for i in range(self.hidden_size):
            avg_i = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            best_i = np.max(avg_i)
            if best_i > best_value:
                best_value = best_i
        return best_value

    def _compute_global_best_fitness_from_avg(self, avg_fitness: list[np.ndarray]) -> float:
        """
        Аналогично, но по готовому списку avg_fitness.
        """
        best = -np.inf
        for arr in avg_fitness:
            val = np.max(arr)
            if val > best:
                best = val
        return best

    def get_best_network(self) -> FeedforwardNetwork:
        """
        Собираем сеть из лучших особей (по средней приспособленности) каждой подпопуляции.
        """
        best_indices = []
        for i in range(self.hidden_size):
            avg_i = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            best_idx = int(np.argmax(avg_i))
            best_indices.append(best_idx)
        return self.assemble_network(best_indices)

    def get_current_network(self) -> FeedforwardNetwork:
        """
        Сеть из «первых» особей (индекс 0 в каждой подпопуляции), для визуализации.
        """
        indices = [0] * self.hidden_size
        return self.assemble_network(indices)