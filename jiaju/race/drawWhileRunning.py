# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
#
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']


def draw_file_figure(filename, savepath, figurename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    accuracy = data[:, 1]
    privacy = data[:, 2]
    storage = data[:, 3]
    reward = data[:, 4]
    plt.figure()
    # draw accuracy
    plt.subplot(2, 2, 1)
    plt.plot(x, accuracy)
    plt.xlabel("episode")
    plt.ylabel("accuracy")

    # draw privacy
    plt.subplot(2, 2, 2)
    plt.plot(x, privacy)
    plt.xlabel("episode")
    plt.ylabel("privacy")

    # draw storage
    plt.subplot(2, 2, 3)
    plt.plot(x, storage)
    plt.xlabel("episode")
    plt.ylabel("storage")

    # draw reward
    plt.subplot(2, 2, 4)
    plt.plot(x, reward)
    plt.xlabel("episode")
    plt.ylabel("reward")

    plt.savefig(os.path.join(savepath, figurename))


def draw_while_running(file_root, file_name, save_path, save_name, title_name, x_label, y_label,
                       label_list):
    """
    Args:
        file_root:
        file_name:
        save_path:
        save_name:
        title_name:
        x_label: x杞寸殑鍚嶅瓧
        y_label: y杞寸殑鍚嶅瓧
        label_list:

    Returns:

    """
    plt.figure()
    file = os.path.join(file_root, file_name)
    data = np.loadtxt(file)
    x_axis = data[:, 0]
    plt.title(title_name)
    # print(len(y_label))
    for i in range(1, len(label_list)):
        # print(i)
        plt.plot(x_axis, data[:, i], color=color_list[i % 10], label=label_list[i])
    plt.legend()  # 鏄剧ず鍥句緥

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()

# draw_while_running('results', 'data.txt', './', 'test.svg', 'code_test', 'epoch', 'reward',
#                    ['episode', 'acc', 's', 'p', 'reward'])
