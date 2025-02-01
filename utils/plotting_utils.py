import itertools
import os
import warnings

import numpy as np
from matplotlib import colors as mcolors, colors
from matplotlib.patches import Rectangle

from create_figs.plotting_setting import plt
from settings import FIGURES_DIR


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


label_to_title_dict = {
    'test_acc': 'Test accuracy',
    'train_acc': 'Train accuracy',
    'normalized_logits': 'Normalized logits',
    'zero_ratio': 'Zero ratio',
    'train_loss': 'Train loss',
    'test_loss': 'Test loss',
    'train_loss_normalize': 'Train loss (normalized)',
    'train_loss_normalize_grad_input': 'Train loss (normalized)',
    'test_loss_normalize': 'Test loss (normalized)',
    'weights_grad': 'Gradient norm',
    'weight_norm': 'Weight norm',
    'max_norm_per_model': 'Lipschitz constant'
}


def scatter_plot_2d_hist(results_list, x_axis, y_axis, x_lim, y_lim, prefix='',
                         plot_all_in_one=False, max_num_values=2000, samples_num=16, plot_rectangles=False,
                         plot_all_titles=False, fig_dir=None):
    N = len(results_list)

    # Prepare the figure if we want to plot all the runs in one figure
    if plot_all_in_one:
        plt.subplots(figsize=(6, 5))
        all_y_values = []
        all_x_values = []

    # Loop over all the runs
    for i, result_obj in enumerate(results_list):
        # Create a new figure if we want to plot each run in a different figure
        if not plot_all_in_one:
            plt.subplots(figsize=(6, 5))

        # Extract x and y values from the list of dictionaries
        x_values, y_values, y_mean = extract_data_for_hist_2d(result_obj, samples_num, max_num_values, x_axis, y_axis)

        # Aggregate the values of all the runs for future mean calculation
        if plot_all_in_one:
            all_y_values.extend(y_values)
            all_x_values.extend(x_values)
        else:
            all_y_values = y_values
            all_x_values = x_values

        # Actual 2d histogram plot
        plt.hist2d(x_values, y_values, bins=60, cmap=result_obj.cmap(), range=[x_lim, y_lim],
                   weights=np.ones_like(x_values) / len(x_values),
                   norm=mcolors.LogNorm(vmin=1e-4, vmax=1e-1))

        if plot_rectangles:
            # plot a box around loss bins 0.35-0.4 (x) and 0.4-1 (y)
            rectangle = Rectangle((0.35, 0.405), 0.045, 0.59, edgecolor='black', linewidth=4, fill=False)
            plt.gca().add_patch(rectangle)

        # Mean plotting
        plot_means_and_legends_hist_2d(all_x_values, all_y_values, plot_all_in_one, result_obj, x_lim, y_lim, y_mean, i,
                                       N)

        adjust_legend_size_and_not_all_in_one_legends(plot_all_in_one, plot_all_titles, prefix, result_obj, samples_num,
                                                      x_axis, y_axis, x_lim, y_lim, fig_dir)

    if plot_all_in_one:
        plot_all_in_one_legends(N, prefix, result_obj, x_axis, x_lim, y_axis, y_lim, fig_dir)


def adjust_legend_size_and_not_all_in_one_legends(plot_all_in_one, plot_all_titles, prefix, result_obj, samples_num,
                                                  x_axis, y_axis, x_lim, y_lim, fig_dir=None):
    if plot_all_titles:
        plt.title(f"num of samples is: {samples_num}")
        plt.subplots_adjust(top=0.9, bottom=0.14, left=0.1, right=0.98)
    else:
        plt.subplots_adjust(top=0.97, bottom=0.14, left=0.15, right=0.98)
    rearange_ticks_for_hist_2d(x_lim, y_lim, arrange=not plot_all_in_one)
    if not plot_all_in_one:
        legend = plt.legend(loc='lower left', handlelength=1.2, markerscale=1., handletextpad=0.1)
        plt.gca().add_artist(legend)
        save_and_show_hist_2d(prefix, result_obj, x_axis, y_axis, fig_dir)


def plot_all_in_one_legends(N, prefix, result_obj, x_axis, x_lim, y_axis, y_lim, fig_dir=None):
    if N != 4:
        warnings.warn(f'legend only works if number of runs is 4 but it is {N}')
    # assert N == 4  # epoch 1, 3, 5, 60
    # Create a dummy glot for spacing the bottom of the legend
    plt.scatter([], [], s=0, label=f"   ")
    handles, labels = plt.gca().get_legend_handles_labels()  # 1, 3, 5, mean, 60, dummy
    # Place the dummy entry in the middle of the legend
    handles.insert(2, handles.pop(-1))  # 1, 3, dummy, 5, mean, 60
    labels.insert(2, labels.pop(-1))  # 1, 3, dummy, 5, mean, 60
    # Pull out the mean label entry to be the last legend entry
    handles.append(handles.pop(-2))  # 1, 3, dummy, 5, 60, mean
    labels.append(labels.pop(-2))  # 1, 3, dummy, 5, 60, mean
    # Create one legend with two columns and one legend with one column
    labels[-2] = f'{labels[-2]}             '
    legend = plt.legend(handles[:-1], labels[:-1], loc='lower left', handlelength=1.2, markerscale=1.,
                        handletextpad=0.1, ncol=2, columnspacing=1.5)
    plt.gca().add_artist(legend)
    legend = plt.legend([handles[-1]], [labels[-1]], loc='lower left', handlelength=1.2, markerscale=1.,
                        handletextpad=0.1, frameon=False)
    plt.gca().add_artist(legend)
    rearange_ticks_for_hist_2d(x_lim, y_lim, arrange=True)
    save_and_show_hist_2d(prefix, result_obj, x_axis, y_axis, fig_dir)


def plot_means_and_legends_hist_2d(all_x_values, all_y_values, plot_all_in_one, result_obj, x_lim, y_lim, y_mean, i, N):
    # Only plot the mean for the last run if we want to plot all the runs in one figure
    if (not plot_all_in_one) or (i == N - 1):
        # Calculate the mean for each bin
        means, xedges_new = calculate_means_for_hist_2d(x_lim, all_x_values, y_lim, all_y_values)
        # Plot the mean test accuracy for each bin
        plt.scatter(xedges_new, means, color='black', marker='o', s=30, zorder=10,
                    label='Mean test accuracy (per loss bin)')

    # Plot the y-mean on the graph
    color = result_obj.color()
    if plot_all_in_one:
        # Plot fake lines for the legend
        plt.plot([], [], color=color, linestyle=':', linewidth=4, label=result_obj.get_label())
    else:
        reduced_color = list(color)
        reduced_color[-1] = 0.5
        plt.axhline(y=y_mean, color=reduced_color, linestyle=':', linewidth=4,
                    label=f'Test accuracy mean (total) {y_mean:.3f}')


def calculate_means_for_hist_2d(x_lim, x_values, y_lim, y_values):
    _, xedges, _ = np.histogram2d(x_values, y_values, bins=40, range=[x_lim, y_lim],
                                  weights=np.ones_like(x_values) / len(x_values))
    # Calculate mean for each x-value bin
    y_values_arr = np.array(y_values)
    x_bin_indices = np.digitize(x_values, xedges)
    means = [np.mean(y_values_arr[x_bin_indices == i]) for i in range(1, len(xedges)) if
             np.sum(x_bin_indices == i) > 0]
    xedges_new = [xedges[i] for i in range(1, len(xedges)) if np.sum(x_bin_indices == i) > 0]
    return means, xedges_new


def save_and_show_hist_2d(prefix, result_obj, x_axis, y_axis, fig_dir=None):
    # create the figure directory if it does not exist
    if fig_dir is not None:
        fig_dir = FIGURES_DIR / fig_dir
    else:
        fig_dir = FIGURES_DIR
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_name = f"{prefix}2d_hist_{x_axis}_{y_axis}_{result_obj.get_default_str()}.png"
    fig_name = fig_name.replace('/', '_')
    print(fig_dir / fig_name)
    plt.savefig(fig_dir / fig_name, dpi=300)
    plt.show()


def rearange_ticks_for_hist_2d(x_lim, y_lim, arrange=True):
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    if arrange:
        step = 0.1
        x_marks = np.arange(x_lim[0], x_lim[1] + step, step)
        y_marks = np.arange(y_lim[0], y_lim[1] + step, step)
        plt.xticks(x_marks, minor=True)
        plt.xticks(x_marks[::2])
        plt.yticks(y_marks, minor=True)
        plt.yticks(y_marks[::2])
        plt.grid(which='both')


def extract_data_for_hist_2d(result_obj, samples_num, max_num_values, x_axis, y_axis):
    data_dict = result_obj.single_samples_num_results
    results_obj = data_dict[samples_num]
    # Define the bins range and edges
    bin_edges = [i / 100 for i in range(20, 75, 5)]
    # Create dictionaries to store x and y values in each bin
    bins_data = {i: {x_axis: [], y_axis: []} for i in bin_edges}
    # Group 'x' and 'y' values in each bin
    for data in results_obj.values():
        if not isinstance(data, dict):
            continue
        x_value = data[x_axis]
        y_value = data[y_axis]
        for bin_start in bin_edges:
            if bin_start <= x_value < bin_start + 0.05:
                bins_data[bin_start][x_axis].append(x_value)
                bins_data[bin_start][y_axis].append(y_value)
                break
    # Extract x and y values from the list of dictionaries
    if max_num_values is not None:
        x_values = [data[x_axis] for data in results_obj.values() if isinstance(data, dict)][:max_num_values]
        y_values = [data[y_axis] for data in results_obj.values() if isinstance(data, dict)][:max_num_values]
    else:
        x_values = [data[x_axis] for data in results_obj.values()]
        y_values = [data[y_axis] for data in results_obj.values()]

    y_mean = np.mean(np.array(y_values))
    return x_values, y_values, y_mean


def create_acc_plot(run_list, title, x_label, y_label, xlim=None, ylim=None, linear_y=False, legend=None, prefix='',
                    plot_narrow_sgd=True):
    plt.figure(figsize=(8, 4.5))

    for i, run in enumerate(run_list):
        x_values = run.multi_samples_res.samples_num
        y_values = run.multi_samples_res.test_acc_mean
        y_err = run.multi_samples_res.test_acc_std

        if plot_narrow_sgd and run.algorithm == 'sgd':
            if run.arch == 'lenet' and run.width == (0.165, 0.04) or run.arch != 'lenet' and run.width == 0.165:
                # plot y_line in the color of the algorithm with value equal to the last value of the algorithm
                plt.axhline(y=y_values[-1], color=run.color(), linestyle='--')

        plt.errorbar(x_values, y_values, yerr=y_err, fmt='-o', linestyle=None, capsize=0, capthick=3,
                     label=run.get_label(), color=run.color())

    # Create the graph
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # add x_ticks on even number starting from x_lim[0] to x_lim[1]
    if xlim[1] - xlim[0] > 50:
        x_ticks = np.arange(xlim[0], xlim[1], 20)
    else:
        x_ticks = np.arange(xlim[0], xlim[1], 4)
    plt.xticks(x_ticks)

    # Show the graph
    plt.grid(True)

    plt.subplots_adjust(top=0.97, bottom=0.21, left=0.15, right=0.99)

    fig_folder = FIGURES_DIR / run.fig_name
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # Save the graph as svg
    save_path = fig_folder / f"{prefix}acc_plot_{run.dataset}_{run.arch}.png"
    print(f"Saving figure to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()


def create_plot(run_list, x_axis, y_axis, title, x_label, y_label, linear_y=False, xlim=None,
                ylim=None, legend=True, prefix=''):
    # Create the graph
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))

    for run in run_list:
        # Extract the x and y values from the list of dictionaries
        x_values = run.multi_samples_res.__getattribute__(x_axis)
        y_values = run.multi_samples_res.__getattribute__(y_axis)
        y_err = run.multi_samples_res.__getattribute__(y_axis.replace("mean", "std"))

        plt.plot(x_values, y_values, marker='o', linestyle='-', color=run.color())

        plt.errorbar(x_values, y_values, yerr=y_err, fmt='-o', marker='o', linestyle=None, capsize=0, capthick=3,
                     label=run.get_label(), color=run.color())

    # Create the x values for the default line
    x_min = min(x_values) if xlim is None else xlim[0]
    x_max = max(x_values) if xlim is None else xlim[1]
    linear_x = np.linspace(x_min, x_max, 100)

    if linear_y:
        plt.plot(linear_x, linear_x, linestyle='--', color='black', label='Random classifier of independent samples')
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label, fontsize=22)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # Set the y-axis ticks to display every second number
    y_ticks = np.arange(ax1.get_ylim()[0], ax1.get_ylim()[1] + 1, 4)
    y_ticklabels = [f'{int(val)}' for val in y_ticks]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticklabels)
    # add x_ticks on even number starting from x_lim[0] to x_lim[1]
    if xlim[1] - xlim[0] > 50:
        x_ticks = np.arange(xlim[0], xlim[1], 20)
    else:
        x_ticks = np.arange(xlim[0], xlim[1], 4)

    plt.xticks(x_ticks)

    plt.subplots_adjust(top=0.97, bottom=0.21, left=0.15, right=0.99)

    # Show the graph
    plt.grid(True)

    fig_folder = FIGURES_DIR / run.fig_name
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # Save the graph as svg
    save_path = fig_folder / f"{prefix}num_models_plot_{run.dataset}_{run.arch}.png"
    print(f"Saving figure to {save_path}")
    plt.savefig(save_path, dpi=300)

    plt.show()


def create_acc_box_plot(run_list, x_label, y_label, xlim=None, ylim=None, legend=True, prefix=''):
    # fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))

    bp_list = []
    all_models_num = []
    # Aggregate the data from each configuration run
    for i, run in enumerate(run_list):
        # acc_data contains a list of lists of tuples, each list is for a seed, each tuple is (num_samples, acc).
        # the number of tuples in each list is the number of models for this seed
        acc_data = run.multi_samples_res.acc_aux

        # choose only the first 26 number of samples
        acc_data = [list(filter(lambda x: x[0] <= xlim[1], acc)) for acc in acc_data]

        # For each seed, remove models if there are enough models for this sample num
        max_num_models = 100

        # Get the number of models for sample num
        acc_data_grouped_per_seed = [itertools.groupby(acc, lambda x: x[0]) for acc in acc_data]
        acc_data_grouped_per_seed = [[list(group)[:max_num_models] for key, group in acc_grouped] for acc_grouped
                                     in
                                     acc_data_grouped_per_seed]
        all_models_num.extend([len(group) for acc_grouped in
                               acc_data_grouped_per_seed for group in acc_grouped])

        # concatenate all the data from the different seeds
        acc_data = list(itertools.chain.from_iterable(itertools.chain.from_iterable(acc_data_grouped_per_seed)))
        # sort the list according to the number of samples
        acc_data.sort(key=lambda x: x[0])
        # Group the data by the number of samples, the first element of each tuple
        acc_grouped_by_samples = itertools.groupby(acc_data, lambda x: x[0])
        acc_grouped_by_samples_array = []
        num_samples_array = []
        for key, group in acc_grouped_by_samples:
            num_samples_array.append(key)
            # Get the accuracy values for this number of samples
            acc_grouped_by_samples_array.append(np.array([val[1] for val in group]))

        # Get the minimum number of models within all number of samples
        min_num_models = min([len(acc) for acc in acc_grouped_by_samples_array])

        # Get the accuracy values for the minimum number of models
        acc_grouped_by_samples_array = np.array([acc[:min_num_models] for acc in acc_grouped_by_samples_array])
        num_samples_array = np.array(num_samples_array)

        # Calculate positions for each algorithm with 4 samples in between
        step = 2
        positions = num_samples_array[1::step]  # take only the even positions

        positions = positions + i * 0.7 - 1.05

        bp = plt.boxplot(acc_grouped_by_samples_array[1::step].T, notch=False, positions=positions, widths=0.5,
                         showfliers=True,
                         showmeans=True,
                         meanline=True,
                         whis=(0, 100),
                         meanprops={'color': 'black'},
                         medianprops={'color': run.color()},
                         boxprops={'color': run.color()}, whiskerprops={'color': run.color()},
                         capprops={'color': run.color()}, patch_artist=True)
        bp_list.append(bp)
        plt.setp(bp['boxes'], color=run.color())
        plt.setp(bp['whiskers'], color=run.color())

    print(f"{min(all_models_num)=}")
    print(f"{max(all_models_num)=}")
    print(f"{len(all_models_num)=}")

    plt.xticks(num_samples_array[1::step], num_samples_array[1::step])
    plt.xlim(xlim[0], xlim[1] + 2)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.subplots_adjust(top=0.97, bottom=0.21, left=0.15, right=0.99)
    if legend:
        plt.legend(loc='lower right', labels=[run.get_label() for run in run_list],
                   handles=[bp["boxes"][0] for bp in bp_list], ncol=2, columnspacing=1)

    # add grid
    plt.grid(True)

    # Save the graph as png
    fig_folder = FIGURES_DIR / run.fig_name
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    save_path = fig_folder / f"{prefix}acc_box_plot_{run.dataset}_{run.arch}.png"
    print(f"Saving figure to {save_path}")
    plt.savefig(save_path, dpi=300)

    plt.show()


def plot_normalized_loss_per_epoch(run_list, element='normalized_logits'):
    y_lim = None

    plt.figure(figsize=(6, 5))
    for run in run_list:
        assert len(run.single_samples_num_results) == 1
        for num_samples, models_dict in run.single_samples_num_results.items():
            data = [(key, model_dict[element]) for key, model_dict in models_dict.items() if
                    isinstance(model_dict, dict)]
            model_ids, data = zip(*data)
            data = np.array(data)  # shape: (num_models, num_epochs/num_epoch+1, [num_samples])

            if element == 'weights_grad':
                data = data[..., None]  # shape: (num_models, num_epochs/num_epoch+1, 1)

            data = data.transpose(1, 0, 2)  # shape: (num_epochs/num_epoch+1, num_models, num_samples/1)
            epochs_num = data.shape[0]

            total_mean = data.mean(axis=(1, 2))  # shape: (num_epochs/num_epoch+1, 1)
            x_axis = np.arange(epochs_num)

            plt.plot(x_axis, total_mean, label=run.get_label(), color=run.color())

    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel(label_to_title_dict[element])
    plt.yscale('log')
    plt.ylim(y_lim)
    plt.grid(True)
    plt.subplots_adjust(left=0.19, bottom=0.15, right=0.99, top=0.97)
    plt.savefig(FIGURES_DIR / f"{element}_per_epoch.png", dpi=300)
    plt.show()


def get_xlim_ylim(result_obj):
    if 'mnist' in result_obj.dataset:
        if 'long' in result_obj.expname:
            xlim = (2, 128)
            test_acc_ylim = (0.9, 1)
            y_lim = (0, 26)
        else:
            xlim = (2, 32)
            test_acc_ylim = (0.6, 1)
            if result_obj.seed == 202:
                y_lim = (0, 20)
            else:
                y_lim = (0, xlim[1])
    elif 'cifar' in result_obj.dataset:
        xlim = (2, 24)
        y_lim = (0, xlim[1])
        if result_obj.seed == 224:
            test_acc_ylim = (0.5, 0.9)
        else:
            test_acc_ylim = (0.5, 0.85)

    else:
        raise ValueError(f"dataset {result_obj.dataset} not supported")
    return test_acc_ylim, xlim, y_lim
