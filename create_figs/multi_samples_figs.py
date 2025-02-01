from typing import List

import matplotlib
import numpy as np
from matplotlib.legend_handler import HandlerLine2D

from create_figs.run_result_object import RunResultsObject
from matplotlib import pyplot as plt
from settings import HOST_NAME
from utils import model_utils
from utils.plotting_utils import create_acc_plot, create_plot, create_acc_box_plot, get_xlim_ylim

print(f"Running {__file__} on {HOST_NAME}")

suffix_with_acc = 'model_stats.db'
suffix_summary = 'model_stats_summary.db'


class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        xx = 0.7 * height
        return super(SymHandler, self).create_artists(legend, orig_handle, xdescent, xx, width, height, fontsize,
                                                      trans)


def analyze_between_samples_num(results_list: List[RunResultsObject],
                                samples_num=False,
                                boxplot=False,
                                test_acc=False,
                                legend=True,
                                prefix=''):
    # for each path, get the summary db path and the db path with the accuracy
    for result_obj in results_list:
        print(result_obj)
        result_obj.get_multi_samples_res()

    x_label = '# samples'

    test_acc_ylim, xlim, y_lim = get_xlim_ylim(result_obj)

    if samples_num:
        y_label = r'-log$_2 (P_W($Train Err.=0$))$'
        create_plot(run_list=results_list, x_axis="samples_num", y_axis="models_ratio_mean",
                    title='Probability to sample 100% train accuracy', x_label=x_label,
                    y_label=y_label, linear_y=True, xlim=xlim, ylim=y_lim, prefix=prefix)

    if test_acc:
        y_label = 'Test accuracy'
        create_acc_plot(run_list=results_list, title='Mean test accuracy',
                        x_label=x_label, y_label=y_label, xlim=xlim, ylim=test_acc_ylim, linear_y=False,
                        legend=legend, prefix=prefix)

    if boxplot:
        y_label = 'Test accuracy'
        create_acc_box_plot(run_list=results_list, x_label=x_label, y_label=y_label, xlim=xlim,
                            ylim=(0.3, 1), legend=legend, prefix=prefix)


def get_width_comp_legend(dataset, seed):
    handles, labels = get_fake_legend(get_config_list_width_comp_all_samples(dataset, seed))

    plt.figure()
    legend = plt.legend()
    h, l = legend.axes.get_legend_handles_labels()

    # Move the first handle to the middle
    handles.insert(5, h[0])
    labels.insert(5, l[0])

    # Delete the figure
    plt.close()
    return handles, labels


def get_width_comp_legend_seperate_fig(dataset, seed):
    arch = 'resnet'
    handles, labels = get_fake_legend(get_config_list_width_comp_all_samples(dataset, seed, arch=arch))

    plt.figure()
    if arch == 'lenet':
        columns = 7
        narrow_sgd_handle = handles[labels.index('SGD, width=$\\frac{1}{6}^*$')]
    else:
        columns = 6
        narrow_sgd_handle = handles[labels.index('SGD, width=$\\frac{1}{6}$')]
    narrow_sgd_color = narrow_sgd_handle.get_color()
    plt.plot([], [], 'k--', label='Theoretical')
    plt.plot([], [], ' ', label='G&C, width:')
    plt.plot([], [], '--', color=narrow_sgd_color, label='Narrow SGD')
    plt.plot([], [], ' ', label='SGD, width:')

    legend = plt.legend()
    h, l = legend.axes.get_legend_handles_labels()
    plt.close()

    # Move the first handle to the middle
    handles.insert(0, h[0])
    labels.insert(0, l[0])
    handles.insert(1, h[1])
    labels.insert(1, l[1])
    handles.insert(columns, h[2])
    labels.insert(columns, l[2])
    handles.insert(columns + 1, h[3])
    labels.insert(columns + 1, l[3])
    labels = [model_utils.split('=')[-1] for label in labels]
    new_order = np.arange(len(handles))
    h_reordered = [handles[i] for i in new_order]
    l_reordered = [labels[i] for i in new_order]

    h_reordered = np.array(h_reordered, dtype="object").reshape((2, columns)).T.reshape(-1).tolist()
    l_reordered = np.array(l_reordered, dtype="object").reshape((2, columns)).T.reshape(-1).tolist()

    legend_fig, legend_ax = plt.subplots(figsize=(15, 1.5))
    legend_ax.axis('off')
    plt.legend(h_reordered, l_reordered, loc='center', ncol=columns, columnspacing=1, frameon=False,
               labelspacing=0.05, handler_map={matplotlib.lines.Line2D: SymHandler()}, handleheight=2.5,
               fontsize=25,
               handlelength=1.5)

    legend_fig.savefig(f'figures/legend_widths_{arch}.png', pad_inches=0.1, dpi=600)
    plt.show()
    # Delete the figure
    plt.close()
    return handles, labels


def get_depth_comp_legend_seperate_fig(dataset, seed):
    arch = 'lenet'
    handles, labels = get_fake_legend(get_depth_comp_all_samples_config_list(dataset, seed, arch=arch))

    plt.figure()

    plt.plot([], [], 'k--', label='Theoretical')
    if arch == 'lenet':
        plt.plot([], [], ' ', label='G&C:')
        plt.plot([], [], ' ', label=' ', markersize=0, linewidth=0)
        plt.plot([], [], ' ', label='SGD:')
    elif arch == 'mlp':
        plt.plot([], [], ' ', label='G&C, layers:')
        plt.plot([], [], ' ', label=' ', markersize=0, linewidth=0)
        plt.plot([], [], ' ', label='SGD, layers:')
    else:
        plt.plot([], [], ' ', label='G&C, blocks:')
        plt.plot([], [], ' ', label=' ', markersize=0, linewidth=0)
        plt.plot([], [], ' ', label='SGD, blocks:')

    legend = plt.legend()
    h, l = legend.axes.get_legend_handles_labels()
    plt.close()

    if arch == 'mlp':
        columns = 6
    elif arch == 'lenet':
        columns = 6
    elif arch == 'resnet':
        columns = 5
    else:
        raise ValueError(f'arch {arch} not supported')

    # Move the first handle to the middle
    labels = [model_utils.split(',')[-1] for label in labels]

    if arch == 'lenet':
        for ind, label in enumerate(labels):
            if label == ' c p p f ':
                labels[ind] = '1c-1f'
            elif label == ' c p c p f ':
                labels[ind] = '2c-1f'
            elif label == ' c p c p f f ':
                labels[ind] = '2c-2f'
            elif label == ' c p c p f f f ':
                labels[ind] = '2c-3f'
    elif arch == 'resnet':
        for ind, label in enumerate(labels):
            labels[ind] = str(int(label) + 1)

    handles.insert(0, h[0])
    labels.insert(0, l[0])
    handles.insert(1, h[1])
    labels.insert(1, l[1])
    handles.insert(columns, h[2])
    labels.insert(columns, l[2])
    handles.insert(columns + 1, h[3])
    labels.insert(columns + 1, l[3])

    new_order = np.arange(len(handles))
    h_reordered = [handles[i] for i in new_order]
    l_reordered = [labels[i] for i in new_order]

    h_reordered = np.array(h_reordered, dtype="object").reshape((2, columns)).T.reshape(-1).tolist()
    l_reordered = np.array(l_reordered, dtype="object").reshape((2, columns)).T.reshape(-1).tolist()

    legend_fig, legend_ax = plt.subplots(figsize=(15, 1.5))
    legend_ax.axis('off')
    plt.legend(h_reordered, l_reordered, loc='center', ncol=columns, columnspacing=1, frameon=False,
               labelspacing=0.05, handler_map={matplotlib.lines.Line2D: SymHandler()}, handleheight=2.5,
               fontsize=25,
               handlelength=1.5)
    # save fig in high resolution
    legend_fig.savefig(f'figures/legend_depth_{arch}.png', pad_inches=0.1, dpi=600)
    plt.show()
    plt.close()
    return handles, labels


def get_fake_legend(results_list: List[RunResultsObject]):
    plt.figure()
    for result_obj in results_list:
        plt.plot([], [], marker='|', markersize=20, color=result_obj.color(), label=result_obj.get_label(),
                 markeredgewidth=5)

    legend = plt.legend()
    handles, labels = legend.axes.get_legend_handles_labels()

    # Delete the figure
    plt.close()

    return handles, labels


def get_config_list_width_comp_all_samples(dataset, seed, fig_name=None, boxplot=False, arch='lenet', prefix=''):
    """
    Get config object list for width comparison with all samples (x-axis are number of samples)
    """
    permutation_seed = [200, 201, 202, 203]

    width_list = [(0.165, 0.04), 0.165, 0.33, 0.66, 1] if arch == 'lenet' else [0.165, 0.33, 0.66, 1]
    if arch == 'lenet':
        depth = [(0, 1, 2)]
    elif arch == 'mlp':
        depth = [(0, 1, 2, 3, 4)]
    elif arch == 'resnet':
        depth = [(1, 2, 3)]
    else:
        raise ValueError(f'arch {arch} not supported')

    width_comparison_dict = {
        'algorithm': ['guess', 'sgd'],
        'dataset': [dataset],
        'seed': [seed],
        'arch': [arch],
        'fig_name': [fig_name],
        'permutation_seed': permutation_seed,
        'initialization': ['uniform', 'kaiming'],
        'lr': [0.01],
        'depth': depth,
        'prefix': [prefix],
        'width': width_list

        if not boxplot else [
            (0.165, 0.04),
            1,
        ],
    }

    width_comp_list = RunResultsObject.generate_configs_list(width_comparison_dict, comparison_variable_name='width')
    return width_comp_list


def get_depth_comp_all_samples_config_list(dataset, seed, fig_name=None, arch='lenet', prefix=''):
    permutation_seed = [200, 201, 202, 203]

    if arch == 'lenet':
        conv_layers = [(0,), (0, 1), (0, 1), (0, 1)]
        depth_options = [(2,), (2,), (1, 2), (0, 1, 2)]
    elif arch == 'mlp':
        conv_layers = [(0, 1), (0, 1), (0, 1), (0, 1)]  # arbitrary doesn't matter
        depth_options = [(3, 4), (2, 3, 4), (1, 2, 3, 4), (0, 1, 2, 3, 4)]
    elif arch == 'resnet':
        conv_layers = [(0, 1), (0, 1), (0, 1)]  # arbitrary doesn't matter
        depth_options = [(3,), (2, 3), (1, 2, 3)]
    else:
        raise ValueError(f'arch {arch} not supported')

    depth_comparison_dict = {
        'algorithm': ['guess', 'sgd'],
        'dataset': [dataset],
        'seed': [seed],
        'arch': [arch],
        'fig_name': [fig_name],
        'permutation_seed': permutation_seed,
        'initialization': ['uniform', 'kaiming'],
        'lr': [0.01],
        'width': [0.33],
        'conv_layers': conv_layers,
        'depth': depth_options,
        'prefix': [prefix]
    }

    depth_comp_list = RunResultsObject.generate_configs_list(depth_comparison_dict, comparison_variable_name='depth')
    return depth_comp_list


def create_fig_depth(dataset, seed, fig_name, arch, prefix=''):
    # ===========================================================================
    # Effects of Depth on SGD and GNC
    # ===========================================================================
    analyze_between_samples_num(
        get_depth_comp_all_samples_config_list(dataset=dataset, seed=seed, fig_name=fig_name, arch=arch, prefix=prefix),
        test_acc=True,
        samples_num=True, legend=False, prefix='depth_')


def create_fig_width(dataset, seed, fig_name, arch, prefix=''):
    # ===========================================================================
    # Effects of Widhts on SGD and GNC
    # ===========================================================================
    analyze_between_samples_num(
        get_config_list_width_comp_all_samples(dataset=dataset, seed=seed, fig_name=fig_name, arch=arch, prefix=prefix),
        test_acc=True,
        samples_num=True, legend=False)


def create_fig_width_boxplot(dataset, seed, fig_name, arch):
    # ===========================================================================
    # Effects of Widhts on SGD and GNC - plot boxplot
    # ===========================================================================
    analyze_between_samples_num(
        get_config_list_width_comp_all_samples(dataset=dataset, seed=seed, fig_name=fig_name, arch=arch, boxplot=True),
        boxplot=True, legend=False)


def run_multi_samples_figs(fig_name):
    # ================================================================================================================
    # ===================================== widths - figure 3  =======================================================
    # ================================================================================================================
    if fig_name == 'fig3_lenet_mnist':
        create_fig_width(dataset='mnist', seed=202, arch='lenet', fig_name='fig3_lenet_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig3_mlp_mnist':
        create_fig_width(dataset='mnist', seed=202, arch='mlp', fig_name='fig3_mlp_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig3_resnet_mnist':
        create_fig_width(dataset='mnist', seed=202, arch='resnet', fig_name='fig3_resnet_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig3_lenet_cifar':
        create_fig_width(dataset='cifar', seed=201, arch='lenet', fig_name='fig3_lenet_cifar')  # CIFAR Bird vs Ship
    elif fig_name == 'fig3_mlp_cifar':
        create_fig_width(dataset='cifar', seed=201, arch='mlp', fig_name='fig3_mlp_cifar')  # CIFAR Bird vs Ship
    elif fig_name == 'fig3_resnet_cifar':
        create_fig_width(dataset='cifar', seed=201, arch='resnet', fig_name='fig3_resnet_cifar')  # CIFAR Bird vs Ship
    # ================================================================================================================
    # ===================================== depths - figure 4 ========================================================
    # ================================================================================================================
    elif fig_name == 'fig4_lenet_mnist':
        create_fig_depth(dataset='mnist', seed=202, arch='lenet', fig_name='fig4_lenet_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig4_mlp_mnist':
        create_fig_depth(dataset='mnist', seed=202, arch='mlp', fig_name='fig4_mlp_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig4_resnet_mnist':
        create_fig_depth(dataset='mnist', seed=202, arch='resnet', fig_name='fig4_resnet_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig4_lenet_cifar':
        create_fig_depth(dataset='cifar', seed=201, arch='lenet', fig_name='fig4_lenet_cifar')  # CIFAR Bird vs Ship
    elif fig_name == 'fig4_mlp_cifar':
        create_fig_depth(dataset='cifar', seed=201, arch='mlp', fig_name='fig4_mlp_cifar')  # CIFAR Bird vs Ship
    elif fig_name == 'fig4_resnet_cifar':
        create_fig_depth(dataset='cifar', seed=201, arch='resnet', fig_name='fig4_resnet_cifar')  # CIFAR Bird vs Ship
    # ================================================================================================================
    # ============================== widths appendix figure 19 - lenet MNIST =========================================
    # ================================================================================================================
    elif fig_name == 'fig19_lenet_mnist_2vs5':
        create_fig_width(dataset='mnist', seed=200, arch='lenet', fig_name='fig19_lenet_mnist_2vs5')  # MNIST 2 vs 5
        create_fig_width_boxplot(dataset='mnist', seed=200, arch='lenet',
                                 fig_name='fig19_lenet_mnist_2vs5')  # MNIST 2 vs 5
    elif fig_name == 'fig19_lenet_mnist_2vs8':
        create_fig_width(dataset='mnist', seed=201, arch='lenet', fig_name='fig19_lenet_mnist_2vs8')  # MNIST 2 vs 8
        create_fig_width_boxplot(dataset='mnist', seed=201, arch='lenet',
                                 fig_name='fig19_lenet_mnist_2vs8')  # MNIST 2 vs 8
    elif fig_name == 'fig19_lenet_mnist_0vs7':
        create_fig_width(dataset='mnist', seed=202, arch='lenet', fig_name='fig3_lenet_mnist')  # MNIST 0 vs 7
        create_fig_width_boxplot(dataset='mnist', seed=202, arch='lenet', fig_name='fig3_lenet_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig19_lenet_mnist_0vs8':
        create_fig_width(dataset='mnist', seed=203, arch='lenet', fig_name='fig19_lenet_mnist_0vs8')  # MNIST 0 vs 8
        create_fig_width_boxplot(dataset='mnist', seed=203, arch='lenet',
                                 fig_name='fig19_lenet_mnist_0vs8')  # MNIST 0 vs 8
    elif fig_name == 'fig19_lenet_mnist_3vs5':
        create_fig_width(dataset='mnist', seed=204, arch='lenet', fig_name='fig19_lenet_mnist_3vs5')  # MNIST 3 vs 5
        create_fig_width_boxplot(dataset='mnist', seed=204, arch='lenet',
                                 fig_name='fig19_lenet_mnist_3vs5')  # MNIST 3 vs 5
    # ================================================================================================================
    # ============================== widths appendix figure 20 - lenet cifar =========================================
    # ================================================================================================================
    elif fig_name == 'fig20_lenet_cifar_bird_vs_ship':
        create_fig_width(dataset='cifar', seed=201, arch='lenet', fig_name='fig3_lenet_cifar')  # CIFAR Bird vs Ship
        create_fig_width_boxplot(dataset='cifar', seed=201, arch='lenet',
                                 fig_name='fig3_lenet_cifar')  # CIFAR Bird vs Ship
    elif fig_name == 'fig20_lenet_cifar_deer_vs_truck':
        create_fig_width(dataset='cifar', seed=219, arch='lenet',
                         fig_name='fig20_lenet_cifar_deer_vs_truck')  # CIFAR Deer vs Truck
        create_fig_width_boxplot(dataset='cifar', seed=219, arch='lenet',
                                 fig_name='fig20_lenet_cifar_deer_vs_truck')  # CIFAR Deer vs Truck
    elif fig_name == 'fig20_lenet_cifar_plane_vs_frog':
        create_fig_width(dataset='cifar', seed=224, arch='lenet',
                         fig_name='fig20_lenet_cifar_plane_vs_frog')  # CIFAR Plane vs Frog
        create_fig_width_boxplot(dataset='cifar', seed=224, arch='lenet',
                                 fig_name='fig20_lenet_cifar_plane_vs_frog')  # CIFAR Plane vs Frog
    elif fig_name == 'fig20_lenet_cifar_car_vs_ship':
        create_fig_width(dataset='cifar', seed=230, arch='lenet',
                         fig_name='fig20_lenet_cifar_car_vs_ship')  # CIFAR Car vs  Ship
        create_fig_width_boxplot(dataset='cifar', seed=230, arch='lenet',
                                 fig_name='fig20_lenet_cifar_car_vs_ship')  # CIFAR Car vs  Ship
    elif fig_name == 'fig20_lenet_cifar_horse_vs_truck':
        create_fig_width(dataset='cifar', seed=243, arch='lenet',
                         fig_name='fig20_lenet_cifar_horse_vs_truck')  # CIFAR Horse vs Truck
        create_fig_width_boxplot(dataset='cifar', seed=243, arch='lenet',
                                 fig_name='fig20_lenet_cifar_horse_vs_truck')  # CIFAR Horse vs Truck
    # ================================================================================================================
    # =========================== widths  appendix figure 21 - mlp ===================================================
    # ================================================================================================================
    elif fig_name == 'fig21_mlp_mnist_3vs5':
        create_fig_width(dataset='mnist', seed=204, arch='mlp', fig_name='fig21_mlp_mnist_3vs5')  # MNIST 3 vs 5
    elif fig_name == 'fig21_mlp_cifar_plane_vs_frog':
        create_fig_width(dataset='cifar', seed=224, arch='mlp',
                         fig_name='fig21_mlp_cifar_plane_vs_frog')  # CIFAR Plane vs Frog
    # ================================================================================================================
    # ============================== width appendix figure 22 - lenet - 128 samples ==================================
    # ================================================================================================================
    elif fig_name == 'fig22_lenet_mnist':
        create_fig_width(dataset='mnist', seed=202, arch='lenet', fig_name='fig22_lenet_mnist',
                         prefix='long')  # MNIST 0 vs 7
    # ================================================================================================================
    # ==================================== depths appendix figure 23 - lenet =========================================
    # ================================================================================================================
    elif fig_name == 'fig23_lenet_mnist_0vs7':
        create_fig_depth(dataset='mnist', seed=202, arch='lenet', fig_name='fig4_lenet_mnist')  # MNIST 0 vs 7
    elif fig_name == 'fig23_lenet_mnist_0vs8':
        create_fig_depth(dataset='mnist', seed=203, arch='lenet', fig_name='fig23_lenet_mnist_0vs8')  # MNIST 0 vs 8
    elif fig_name == 'fig23_lenet_mnist_3vs5':
        create_fig_depth(dataset='mnist', seed=204, arch='lenet', fig_name='fig23_lenet_mnist_3vs5')  # MNIST 3 vs 5
    elif fig_name == 'fig23_lenet_cifar_bird_vs_ship':
        create_fig_depth(dataset='cifar', seed=201, arch='lenet', fig_name='fig4_lenet_cifar')  # CIFAR Bird vs Ship
    elif fig_name == 'fig23_lenet_cifar_deer_vs_truck':
        create_fig_depth(dataset='cifar', seed=219, arch='lenet',
                         fig_name='fig23_lenet_cifar_deer_vs_truck')  # CIFAR Deer vs Truck
    elif fig_name == 'fig23_lenet_cifar_plane_vs_frog':
        create_fig_depth(dataset='cifar', seed=224, arch='lenet',
                         fig_name='fig23_lenet_cifar_plane_vs_frog')  # CIFAR Plane vs Frog
    # ================================================================================================================
    # ==================================== depths appendix figure 24 - mlp =========================================
    # ================================================================================================================
    elif fig_name == 'fig24_mlp_mnist_3vs5':
        create_fig_depth(dataset='mnist', seed=204, arch='mlp', fig_name='fig24_mlp_mnist_3vs5')  # MNIST 3 vs 5
    elif fig_name == 'fig24_mlp_cifar_plane_vs_frog':
        create_fig_depth(dataset='cifar', seed=224, arch='mlp',
                         fig_name='fig24_mlp_cifar_plane_vs_frog')  # CIFAR Plane vs Frog
    # ================================================================================================================
    # ============================== depth appendix figure 25 - lenet - 128 samples ==================================
    # ================================================================================================================
    elif fig_name == 'fig25_lenet_mnist':
        create_fig_depth(dataset='mnist', seed=202, arch='lenet', fig_name='fig25_lenet_mnist',
                         prefix='long')  # MNIST 0 vs 7
    # ================================================================================================================
    else:
        raise ValueError(f'fig_name {fig_name} not supported')


if __name__ == '__main__':
    run_multi_samples_figs(fig_name='fig3_lenet_mnist')
