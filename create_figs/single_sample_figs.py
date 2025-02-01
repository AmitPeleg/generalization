import argparse
import os
import sys
from copy import copy
from copy import deepcopy

import yaml
from fastargs import get_current_config, set_current_config
from torch import nn

import settings
from create_figs.run_result_object import RunResultsObject
from utils.datasets_utils import get_dataset
from utils.inference_utils import load_models
from utils.plotting_utils import scatter_plot_2d_hist, plot_normalized_loss_per_epoch
from utils.train_utils import calculate_loss_acc, calculate_max_norm_grad_wrt_input_per_model


def create_list_of_configs(config_dir_name):
    config_dir = os.path.join(settings.CONFIG_ROOT, config_dir_name)
    config_list = []
    for i, file in enumerate(os.listdir(config_dir)):
        if file.endswith('.yaml'):
            config_file = os.path.join(config_dir, file)
            config_list.append(config_file)
    return config_list


def create_basic_fig(fig_name, num_samples=16, using_lipschitz_norm=True, lipschitz_norm_from_all_train=False):
    # ===========================================================================
    # Basic scatter plot configuration
    # ===========================================================================
    plot_all_in_one = False
    prefix = ''
    comparison_variable = None
    num_samples = [num_samples]
    config_list = create_list_of_configs(fig_name)
    return config_list, lipschitz_norm_from_all_train, prefix, plot_all_in_one, using_lipschitz_norm, num_samples, comparison_variable, fig_name


def create_fig_7():
    # ===========================================================================
    # Effects of Initialization on SGD - MNIST 0 vs 7 - 16 samples
    # ===========================================================================
    using_lipschitz_norm = False
    lipschitz_norm_from_all_train = False
    plot_all_in_one = True
    prefix = 'normalized_logits'
    fig_name = 'fig7'
    num_samples = [16]
    config_list = create_list_of_configs(fig_name)
    comparison_variable = 'initialization'
    return config_list, lipschitz_norm_from_all_train, prefix, plot_all_in_one, using_lipschitz_norm, num_samples, comparison_variable, fig_name


def create_fig_12():
    # ===========================================================================
    # Kaiming initialization with small number of epochs is similar to uniform [-1,1] - MNIST 0 vs 7 - 16 samples
    # ===========================================================================
    using_lipschitz_norm = True
    lipschitz_norm_from_all_train = False
    plot_all_in_one = True
    prefix = 'epochs'
    fig_name = 'fig12'
    num_samples = [16]
    config_list = create_list_of_configs(fig_name)
    comparison_variable = 'epochs'
    return config_list, lipschitz_norm_from_all_train, prefix, plot_all_in_one, using_lipschitz_norm, num_samples, comparison_variable, fig_name


def load_settings(fig_name):
    if fig_name == 'fig1':
        # Effects of Initialization on SGD and GNC - MNIST 0 vs 7 - 16 samples
        return create_basic_fig(fig_name='fig1', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig2_lipschitz_norm':
        # Effects of Width on SGD and GNC - MNIST 0 vs 7 - 16 samples - lipschitz norm
        return create_basic_fig(fig_name='fig2', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig2_weights_norm':
        # Effects of Width on SGD and GNC - MNIST 0 vs 7 - 16 samples - weight norm
        return create_basic_fig(fig_name='fig2', num_samples=16, using_lipschitz_norm=False,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig5':
        # Effects of Depth on SGD and GNC - MNIST 0 vs 7 - 16 samples
        return create_basic_fig(fig_name='fig5', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig7':
        return create_fig_7()
    elif fig_name == 'fig8':
        # Effects of Initialization on SGD and GNC - MNIST 3 vs 5 - 16 samples
        return create_basic_fig(fig_name='fig8', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig9':
        # Effects of Initialization on SGD and GNC - cifar bird vs ship and deer vs truck - 16 samples
        return create_basic_fig(fig_name='fig9', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig10_4samples':
        # Effects of Initialization on SGD and GNC - MNIST 0 vs 7 - 4 samples
        return create_basic_fig(fig_name='fig10_4samples', num_samples=4, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig10_32samples':
        # Effects of Initialization on SGD and GNC - MNIST 0 vs 7 - 32 samples
        return create_basic_fig(fig_name='fig10_32samples', num_samples=32, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig11':
        # Effects of Initialization on SGD and GNC - MNIST 0 vs 7 - 16 samples - weights norm
        return create_basic_fig(fig_name='fig1', num_samples=16, using_lipschitz_norm=False,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig12':
        return create_fig_12()
    elif fig_name == 'fig14_weights_norm':
        # Effects of Width on SGD and GNC - MNIST 0 vs 7 - 4 samples - weights norm
        return create_basic_fig(fig_name='fig14', num_samples=4, using_lipschitz_norm=False,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig14_lipschitz_norm':
        # Effects of Width on SGD and GNC - MNIST 0 vs 7 - 4 samples - lipschitz norm
        return create_basic_fig(fig_name='fig14', num_samples=4, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig15_weights_norm':
        # Effects of Width on SGD and GNC - MNIST 0 vs 7 - 32 samples - weights norm
        return create_basic_fig(fig_name='fig15', num_samples=32, using_lipschitz_norm=False,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig15_lipschitz_norm':
        # Effects of Width on SGD and GNC - MNIST 0 vs 7 - 32 samples - lipschitz norm
        return create_basic_fig(fig_name='fig15', num_samples=32, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig16_weights_norm':
        # Effects of Width on SGD and GNC - MNIST 3 vs 5 - 16 samples - weights norm
        return create_basic_fig(fig_name='fig16', num_samples=16, using_lipschitz_norm=False,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig16_lipschitz_norm':
        # Effects of Width on SGD and GNC - MNIST 3 vs 5 - 16 samples - lipschitz norm
        return create_basic_fig(fig_name='fig16', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig17_weights_norm':
        # Effects of Width on SGD and GNC - cifar deer vs truck - 16 samples - weights norm
        return create_basic_fig(fig_name='fig17', num_samples=16, using_lipschitz_norm=False,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig17_lipschitz_norm':
        # Effects of Width on SGD and GNC - cifar deer vs truck - 16 samples - lipschitz norm
        return create_basic_fig(fig_name='fig17', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig18_weights_norm':
        # Effects of Width on SGD and GNC - cifar bird vs ship - 16 samples - weights norm
        return create_basic_fig(fig_name='fig18', num_samples=16, using_lipschitz_norm=False,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig18_lipschitz_norm':
        # Effects of Width on SGD and GNC - cifar bird vs ship - 16 samples - lipschitz norm
        return create_basic_fig(fig_name='fig18', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig26_4samples':
        # Effects of Depth on SGD and GNC - MNIST 0 vs 7 - 4 samples
        return create_basic_fig(fig_name='fig26_4samples', num_samples=4, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig26_32samples':
        # Effects of Depth on SGD and GNC - MNIST 0 vs 7 - 32 samples
        return create_basic_fig(fig_name='fig26_32samples', num_samples=32, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig27_mnist':
        # Effects of Depth on SGD and GNC - MNIST 3 vs 5 - 16 samples
        return create_basic_fig(fig_name='fig27_mnist', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    elif fig_name == 'fig27_cifar':
        # Effects of Depth on SGD and GNC - cifar bird vs ship and deer vs truck - 16 samples
        return create_basic_fig(fig_name='fig27_cifar', num_samples=16, using_lipschitz_norm=True,
                                lipschitz_norm_from_all_train=False)
    else:
        TypeError(f"Fig number {fig_name} is not supported")


def get_data_for_inference(config, cur_num_samples, lipschitz_norm_from_all_train):
    classes_seed = config['distributed.data_seed']
    if (permutation_seed := config['distributed.permutation_seed']) is None:
        permutation_seed = classes_seed
    train_data, train_labels, _, _, normalization_data, _, _ = \
        get_dataset(config=config,
                    num_samples=cur_num_samples,
                    classes_seed=classes_seed,
                    permutation_seed=permutation_seed)
    if lipschitz_norm_from_all_train:
        normalization_data, *_ = \
            get_dataset(config=config,
                        num_samples=20_000,
                        # there are ~5_000 samples per class, this is passed as batch size to the dataloader,
                        # so it returns all the data in one batch
                        classes_seed=classes_seed,
                        permutation_seed=permutation_seed,
                        device='cpu')

    return train_data, train_labels, normalization_data


def run_single_sample_figs(fig_name):
    config_list, lipschitz_norm_from_all_train, prefix, plot_all_in_one, using_lipschitz_norm, num_samples, comparison_variable, config_fig_name = load_settings(
        fig_name)
    # ===========================================================================
    # Main
    # ===========================================================================
    old_sys_argv = sys.argv
    default_config = copy(get_current_config())
    run_config_list = []

    for config_file in config_list:
        # Parse the arguments
        config, config_dict, config_ns = get_config(config_file, default_config, old_sys_argv)

        # Get the number of samples
        if num_samples is None:
            num_samples = [int(v) for v in config_dict['distributed']['num_samples'].split(",")]

        for cur_num_samples in num_samples:
            run_config_obj = RunResultsObject.run_config_obj_from_config_ns(config_ns,
                                                                            fig_name=config_fig_name,
                                                                            comparison_variable=comparison_variable,
                                                                            num_samples=cur_num_samples,
                                                                            prefix=prefix)

            print(f"{run_config_obj}:link")

            # Aggregate the results
            run_config_obj.get_single_samples_results(False)

            if using_lipschitz_norm:
                x_axis = 'train_loss_normalize_grad_input'
                plot_rectangles = False

                # Update the loss normalization
                update_loss_normalization(config, cur_num_samples, lipschitz_norm_from_all_train, run_config_obj)
            else:
                x_axis = 'train_loss_normalize'
                if config_fig_name == 'fig1':
                    plot_rectangles = False
                else:
                    plot_rectangles = True

            if plot_all_in_one:
                run_config_list.append(run_config_obj)
            else:
                scatter_plot_2d_hist([run_config_obj],
                                     x_axis=x_axis,
                                     y_axis='test_acc',
                                     x_lim=(0., 0.7),
                                     y_lim=(0.4, 1.),
                                     prefix=prefix,
                                     samples_num=cur_num_samples,
                                     plot_rectangles=plot_rectangles,
                                     fig_dir=fig_name)

    if plot_all_in_one:
        if fig_name == 'fig7':
            plot_normalized_loss_per_epoch(run_config_list, element='weights_grad')
        elif fig_name == 'fig12':
            scatter_plot_2d_hist(run_config_list,
                                 x_axis=x_axis,
                                 y_axis='test_acc',
                                 x_lim=(0., 0.7),
                                 y_lim=(0.4, 1.),
                                 prefix=prefix,
                                 plot_all_in_one=True,
                                 max_num_values=500,
                                 fig_dir=fig_name)
        else:
            raise ValueError(f"fig_name {fig_name} is not supported")


def get_config(config_file, default_config, old_sys_argv):
    sys.argv = old_sys_argv.copy()
    sys.argv += ['-C', config_file]
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    # reset fastargs config with deepcopy
    set_current_config(deepcopy(default_config))
    parser = argparse.ArgumentParser()
    config = get_current_config()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config_ns = config.get()
    return config, config_dict, config_ns


def update_loss_normalization(config, cur_num_samples, lipschitz_norm_from_all_train, run_config_obj):
    # Load the models
    models, model_ids = load_models(config, cur_num_samples)
    # Get the data
    train_data, train_labels, normalization_data = get_data_for_inference(config, cur_num_samples,
                                                                          lipschitz_norm_from_all_train)

    loss_func = nn.CrossEntropyLoss(reduction='none')
    # Loop over the models and calculate the max norm per model
    # Calculate the maximmum per model norm of the gradients of the logit differences with respect to the input
    if lipschitz_norm_from_all_train:
        model_batch_size = 2
    else:
        model_batch_size = 16
    max_norm_per_model = calculate_max_norm_grad_wrt_input_per_model(normalization_data.cuda(),
                                                                     models, data_batch_size=None,
                                                                     model_batch_size=model_batch_size)
    normalization_data.cpu()
    # Calculate the normalizaed train loss
    normalized_train_loss, _ = calculate_loss_acc(train_data, train_labels, models, loss_func,
                                                  logits_normalization=max_norm_per_model[None, :, None])
    # Adding the new normalized loss
    for id, train_loss, max_norm in zip(model_ids, normalized_train_loss, max_norm_per_model):
        run_config_obj.single_samples_num_results[cur_num_samples][id][
            'train_loss_normalize_grad_input'] = train_loss.detach().cpu().numpy()
        run_config_obj.single_samples_num_results[cur_num_samples][id][
            'max_norm_per_model'] = max_norm.detach().cpu().numpy().item()


if __name__ == '__main__':
    run_single_sample_figs('fig12')
