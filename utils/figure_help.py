import os
import matplotlib.pyplot as plt

def plot_metric_curve(metric_values, metric_name, save_dir, filename=None, highlight_best=False):
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or f'{metric_name.lower()}_curve.png'

    plt.figure()
    epochs = range(1, len(metric_values) + 1)
    plt.plot(epochs, metric_values, label=metric_name)

    if highlight_best:
        best_epoch = int(metric_values.index(max(metric_values))) + 1
        best_value = max(metric_values)
        plt.scatter(best_epoch, best_value, color='red', label=f'Best: {best_value:.4f} @ Epoch {best_epoch}')
        plt.annotate(f'{best_value:.4f}', xy=(best_epoch, best_value), xytext=(best_epoch, best_value + 0.01),
                     ha='center', fontsize=9, color='red')

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_all_metrics_subplots(loss_history, acc_history, f1_history, auc_history, save_dir, filename='metrics_overview.png'):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(loss_history) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Training Metrics Overview', fontsize=16)

    def plot_with_labels(ax, values, title, color, highlight_best=True):
        ax.plot(epochs, values, label=title, color=color)

        for x, y in zip(epochs, values):
            ax.text(x, y + 0.002, f'{y:.4f}', ha='center', va='bottom', fontsize=8)

        if highlight_best:
            best_idx = int(values.index(max(values)))
            best_x = epochs[best_idx]
            best_y = values[best_idx]
            ax.scatter(best_x, best_y, color='red', zorder=5)
            ax.annotate(f'Best: {best_y:.4f} @ {best_x}',
                        xy=(best_x, best_y),
                        xytext=(best_x, best_y + 0.015),
                        fontsize=9, color='red',
                        ha='center', arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.grid(True)

    plot_with_labels(axs[0, 0], loss_history, 'Loss', 'blue', highlight_best=False)
    plot_with_labels(axs[0, 1], acc_history, 'Accuracy', 'green')
    plot_with_labels(axs[1, 0], f1_history, 'F1 Score', 'orange')
    plot_with_labels(axs[1, 1], auc_history, 'AUC', 'purple')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


import os
def save_metrics_to_file(acc_history, f1_history, auc_history, far_history, frr_history, eer_history, picture_dir, filename='result.txt'):
    """
    将模型评估指标（准确率、F1、AUC、FAR、FRR、EER）保存到指定的文件中。

    :param acc_history: 准确率的历史列表
    :param f1_history: F1 分数的历史列表
    :param auc_history: AUC 的历史列表
    :param far_history: FAR 的历史列表
    :param frr_history: FRR 的历史列表
    :param eer_history: EER 的历史列表
    :param picture_dir: 保存文件的目录路径
    :param filename: 结果文件名，默认为 'result.txt'
    """
    # 确保保存目录存在
    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)

    # 定义文件路径
    result_file_path = os.path.join(picture_dir, filename)

    # 打开文件并写入数据
    with open(result_file_path, 'a') as result_file:
        # 如果文件为空，写入标题行
        if os.stat(result_file_path).st_size == 0:
            result_file.write('Epoch\tAccuracy\tF1 Score\tAUC\tFAR\tFRR\tEER\n')

        # 写入每一轮的结果
        for epoch in range(len(acc_history)):
            result_file.write(f'{epoch + 1}\t{acc_history[epoch]:.4f}\t{f1_history[epoch]:.4f}\t{auc_history[epoch]:.4f}\t{far_history[epoch]:.4f}\t{frr_history[epoch]:.4f}\t{eer_history[epoch]:.4f}\n')

    print(f"Metrics saved to {result_file_path}")

