import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def plot_training_curves(train_losses, val_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 使用插值方法平滑曲线
    smooth_epochs = np.linspace(1, len(train_losses), 500)  # 创建更多的点进行插值
    train_loss_interp = interp1d(epochs, train_losses, kind='cubic')
    val_loss_interp = interp1d(epochs, val_losses, kind='cubic')
    accuracy_interp = interp1d(epochs, accuracies, kind='cubic')

    # 插值生成的平滑曲线
    smooth_train_losses = train_loss_interp(smooth_epochs)
    smooth_val_losses = val_loss_interp(smooth_epochs)
    smooth_accuracies = accuracy_interp(smooth_epochs)

    # 设置图片尺寸和DPI
    plt.figure(figsize=(10, 6), dpi=350)
    plt.plot(smooth_epochs, smooth_train_losses, label='Train Loss', linestyle='-', color='b')
    plt.plot(smooth_epochs, smooth_val_losses, label='Validation Loss', linestyle='-', color='r')
    plt.scatter(epochs, train_losses, color='b', zorder=5)  # 在曲线上添加训练集损失点
    plt.scatter(epochs, val_losses, color='r', zorder=5)  # 在曲线上添加验证集损失点
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)  # 确保x轴是整数
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    plt.show()

    # 设置图片尺寸和DPI
    plt.figure(figsize=(10, 6), dpi=350)
    plt.plot(smooth_epochs, smooth_accuracies, label='Accuracy', linestyle='-', color='g')
    plt.scatter(epochs, accuracies, color='g', zorder=5)  # 在曲线上添加准确率点
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)  # 确保x轴是整数
    plt.legend()
    plt.title('Accuracy Curve')
    plt.grid(True)
    plt.show()
