import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

def visualize_network(network, filename):
    """
    Визуализация структуры сети (input-hidden-output) с улучшениями.
    """
    plt.figure(figsize=(8, 6))
    in_x, in_y = [0]*network.input_size, np.linspace(0, 1, network.input_size)
    hid_x, hid_y = [0.5]*network.hidden_size, np.linspace(0, 1, network.hidden_size)
    out_x, out_y = [1]*network.output_size, np.linspace(0.3, 0.7, network.output_size)

    colormap = cm.get_cmap('coolwarm')

    for i, (x0, y0) in enumerate(zip(in_x, in_y)):
        for j, (x1, y1) in enumerate(zip(hid_x, hid_y)):
            w = network.weights_input_hidden[j, i]
            color = colormap(abs(w))
            plt.plot([x0, x1], [y0, y1], color=color, alpha=0.7, lw=abs(w) * 3 + 1)

    for i, (x0, y0) in enumerate(zip(hid_x, hid_y)):
        for j, (x1, y1) in enumerate(zip(out_x, out_y)):
            w = network.weights_hidden_output[j, i]
            color = colormap(abs(w))
            plt.plot([x0, x1], [y0, y1], color=color, alpha=0.7, lw=abs(w) * 3 + 1)

    plt.scatter(in_x, in_y, s=300, label='Input', color='blue', edgecolors='black', linewidths=1)
    plt.scatter(hid_x, hid_y, s=300, label='Hidden', color='orange', edgecolors='black', linewidths=1)
    plt.scatter(out_x, out_y, s=300, label='Output', color='green', edgecolors='black', linewidths=1)

    for i, txt in enumerate(range(network.input_size)):
        plt.text(in_x[i] - 0.05, in_y[i], str(txt), fontsize=10, ha='center', color='white')
    for i, txt in enumerate(range(network.hidden_size)):
        plt.text(hid_x[i] + 0.05, hid_y[i], str(txt), fontsize=10, ha='center', color='black')
    for i, txt in enumerate(range(network.output_size)):
        plt.text(out_x[i] + 0.05, out_y[i], str(txt), fontsize=10, ha='center', color='black')

    plt.axis('off')
    plt.title('Network Structure with Enhanced Visualization')
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()


def plot_metric(metric_history, ylabel, filename):
    plt.figure()
    sns.lineplot(x=np.arange(len(metric_history)), y=metric_history)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over epochs")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
