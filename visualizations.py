import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.lines import Line2D


def save_network_legend(filename):
    """
    Создает улучшенную легенду для визуализации сети (с учетом цветовой схемы coolwarm и нормализации)
    """
    # Настройки оформления
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.facecolor'] = '#f8f8f8'

    # Создаем элементы легенды
    legend_elements = [
        # Узлы сети
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=22, markeredgewidth=1.5, label='Входной слой'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=22, markeredgewidth=1.5, label='Скрытый слой'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=22, markeredgewidth=1.5, label='Выходной слой'),

        # Связи (отрицательные веса)
        Line2D([0], [0], color=plt.cm.coolwarm(0.0), lw=1.5, alpha=0.8,
               label='Отрицательный вес (малый)'),
        Line2D([0], [0], color=plt.cm.coolwarm(0.0), lw=5.5, alpha=0.8,
               label='Отрицательный вес (большой)'),

        # Связи (положительные веса)
        Line2D([0], [0], color=plt.cm.coolwarm(1.0), lw=1.5, alpha=0.8,
               label='Положительный вес (малый)'),
        Line2D([0], [0], color=plt.cm.coolwarm(1.0), lw=5.5, alpha=0.8,
               label='Положительный вес (большой)'),

        # Пояснение масштабирования
        Line2D([0], [0], marker='', color='w', label='Толщина и насыщенность:'),
        Line2D([0], [0], marker='', color='w', label='величина веса (|w|)'),
        Line2D([0], [0], marker='', color='w', label='Цвет: знак веса'),
    ]

    # Создаем фигуру
    fig = plt.figure(figsize=(10, 7), dpi=120)
    ax = fig.add_subplot(111)

    # Создаем легенду с группировкой
    legend = ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=2,
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='#333333',
        title='Легенда нейронной сети\n(нормализация по 95-му перцентилю)',
        title_fontsize=16
    )

    # Стилизация
    legend.get_frame().set_boxstyle('Round, pad=0.5, rounding_size=0.3')
    legend.get_frame().set_linewidth(2)
    plt.setp(legend.get_title(), color='#2a2a2a', fontweight='bold')

    # Цветной фон для групп
    for i, text in enumerate(legend.get_texts()):
        if i in [0, 1, 2]:  # Узлы
            text.set_color('#1a3657')
        elif i in [3, 4]:  # Отрицательные веса
            text.set_color('#1f77b4')
        elif i in [5, 6]:  # Положительные веса
            text.set_color('#d62728')
        elif i > 6:  # Пояснения
            text.set_color('#555555')
            text.set_fontstyle('italic')

    # Сохранение
    ax.axis('off')
    plt.tight_layout(pad=3.0)
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)

def visualize_network(network, filename):
    """
    Визуализация структуры сети (input-hidden-output) с нормализацией
    цвета и толщины рёбер: масштабируем не по глобальному максимуму,
    а по 95-му перцентилю |w|.  Выбросы клипуются и лишь «насыщают» цвет.
    """

    # ---- 1. Собираем все веса ----
    all_w = np.concatenate([
        network.weights_input_hidden.flatten(),
        network.weights_hidden_output.flatten()
    ])
    if all_w.size == 0:
        return

    # ── НОВОЕ: робастный «максимум» ──────────────────────────────────
    robust_max = np.percentile(np.abs(all_w), 95)  # 95-й перцентиль
    w_max = max(robust_max, 1e-8)                  # защита от нуля

    # ---- 2. Расставляем узлы ----
    in_x, in_y  = [0.0] * network.input_size,  np.linspace(0.1, 0.9, network.input_size)
    hid_x, hid_y = [0.5] * network.hidden_size, np.linspace(0.05, 0.95, network.hidden_size)
    out_x, out_y = [1.0] * network.output_size, np.linspace(0.3, 0.7, network.output_size)

    cmap = cm.get_cmap('coolwarm')
    plt.figure(figsize=(8, 6))

    # ---- 3. Связи input → hidden ----
    for i, (x0, y0) in enumerate(zip(in_x, in_y)):
        for j, (x1, y1) in enumerate(zip(hid_x, hid_y)):
            w = network.weights_input_hidden[j, i]
            w_clip = np.clip(w, -w_max, w_max)           # клипуем выбросы
            norm_signed = (w_clip + w_max) / (2 * w_max)
            norm_abs = abs(w_clip) / w_max
            color = cmap(norm_signed)
            lw = 0.5 + 4.5 * norm_abs
            plt.plot([x0, x1], [y0, y1], color=color, alpha=0.8, linewidth=lw)

    # ---- 4. Связи hidden → output ----
    for i, (x0, y0) in enumerate(zip(hid_x, hid_y)):
        for j, (x1, y1) in enumerate(zip(out_x, out_y)):
            w = network.weights_hidden_output[j, i]
            w_clip = np.clip(w, -w_max, w_max)
            norm_signed = (w_clip + w_max) / (2 * w_max)
            norm_abs = abs(w_clip) / w_max
            color = cmap(norm_signed)
            lw = 0.5 + 4.5 * norm_abs
            plt.plot([x0, x1], [y0, y1], color=color, alpha=0.8, linewidth=lw)

    # ---- 5. Узлы ----
    plt.scatter(in_x,  in_y,  s=200, color='blue',   edgecolors='black', zorder=3)
    plt.scatter(hid_x, hid_y, s=200, color='orange', edgecolors='black', zorder=3)
    plt.scatter(out_x, out_y, s=200, color='green',  edgecolors='black', zorder=3)

    for i in range(network.input_size):
        plt.text(in_x[i] - 0.03,  in_y[i],  str(i), fontsize=9, ha='right',  va='center', color='white')
    for i in range(network.hidden_size):
        plt.text(hid_x[i], hid_y[i], str(i), fontsize=9, ha='center', va='center', color='black')
    for i in range(network.output_size):
        plt.text(out_x[i] + 0.03, out_y[i], str(i), fontsize=9, ha='left',   va='center', color='black')

    plt.axis('off')
    plt.title('Network Structure (95-percentile scaling)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_metric(metric_history, ylabel, filename):
    plt.figure(figsize=(6, 4))
    # Если seaborn импортирован, можно так:
    sns.lineplot(x=np.arange(len(metric_history)), y=metric_history, marker="o")
    # Или, без seaborn:
    # plt.plot(np.arange(len(metric_history)), metric_history, marker="o", linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over epochs")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()