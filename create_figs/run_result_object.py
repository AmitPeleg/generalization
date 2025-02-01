import argparse
import itertools
import sys
from copy import copy, deepcopy
from typing import Optional, List, Union, Sequence

import numpy as np
from fastargs import get_current_config, set_current_config
from matplotlib import pyplot as plt

from settings import CONFIG_ROOT, OUTPUT_ROOT, suffix_with_acc, suffix_summary, table_dict, CustomPath
from utils.plotting_utils import truncate_colormap
from utils.run_result_utils import load_numpy_arrays_from_folder, create_dict_from_rows, get_stats_per_sample, \
    combine_two_keys
from utils.sql import get_model_stats_short_summary, get_model_stats_summary, get_model_stats_acc, \
    get_model_stats_full_with_id


class BetweenSamplesNumResultsSummary:
    def __init__(self):
        self.res = None
        self.r = None
        self.acc_aux = None
        self.acc_stats_mean_std = None
        self.attempts = []
        self.curr_samples_dict = {}

        # self.test_acc = None
        self.test_acc_mean = None
        self.test_acc_std = None
        self.models_ratio_mean = None
        self.models_ratio_std = None
        self.samples_num = None


class RunResultsObject:
    def __init__(self, **kwargs):
        self.arch: Optional[str] = "lenet"  # network name
        self.algorithm: str = ''  # algorithm name [sgd, guess]
        self.dataset: str = ''  # dataset name [mnist, cifar10, cifar100] and seed
        self.initialization: str = ''  # initializations of the network
        self.seed: int  # seed for the run
        self.permutation_seed: Optional[int] = None  # permutation seed for the run
        self.epochs: Union[int, float] = 60  # number of epochs to run
        self.lr: float = 0.01  # learning rate
        self.num_samples: Optional[int] = None  # number of samples to run
        self.width: float = 0.5  # width of the network
        self.depth: Sequence = [0, 1, 2]  # depth of the network
        self.conv_layers: Sequence = [0, 1]  # number of conv layers
        self.comparison_variable: Optional[str] = None
        self.prefix: Optional[str] = None
        self.algorithm_initialization: Optional[tuple] = None
        self.dataset_seed: Optional[tuple] = None
        self.desc: Optional[str] = None
        self.fig_name: Optional[str] = None
        self.__dict__.update(kwargs)

        # Extract tupled params
        self.algorithm = self.algorithm_initialization[0]
        self.initialization = self.algorithm_initialization[1]

        self.dataset = self.dataset_seed[0].replace('cifar10', 'cifar')
        self.seed = int(self.dataset_seed[1])

        if 'conv_layers_depth' in kwargs:
            self.conv_layers = kwargs['conv_layers_depth'][0]
            self.depth = kwargs['conv_layers_depth'][1]

        # Set the permutation seed if it is not set
        if self.permutation_seed is None:
            self.permutation_seed = self.seed

        # Data structures for holding the results
        self.multi_samples_res = BetweenSamplesNumResultsSummary()
        self.single_samples_num_results = None

    def __str__(self):
        return f"{self.config_path():link}"

    def comparison_variable_value(self):
        if self.comparison_variable is not None:
            return self.__getattribute__(self.comparison_variable)
        else:
            return None

    def get_label(self):
        if self.algorithm == 'sgd':
            alg_name = 'SGD'
        elif self.algorithm == 'guess':
            alg_name = 'G&C'
        else:
            alg_name = self.algorithm

        comparison_val = self.comparison_variable_value()
        if comparison_val is None:
            return f"{alg_name}"

        if self.comparison_variable == 'width' and isinstance(comparison_val, (float, tuple)):
            label, _, _ = self.get_width_str()
            return f"{alg_name}, {self.comparison_variable}={label}"
        elif self.comparison_variable == 'depth' and self.arch == 'lenet':
            label = self.get_depth_str()
            return f"{alg_name}, {label}"
        elif self.comparison_variable == 'initialization':
            if comparison_val == 'uniform':
                comparison_val = 'Uniform [-1, 1]'
            elif comparison_val == 'uniform02':
                comparison_val = 'Uniform [-0.2, 0.2]'
            elif comparison_val == 'kaiming':
                comparison_val = 'Kaiming'
            elif comparison_val == 'kaiming_gaussian':
                comparison_val = 'Kaiming Gaussian'
            return f"{comparison_val}"
        elif self.comparison_variable == 'epochs':
            return f"{comparison_val} epochs"
        else:
            return f"{alg_name}, {self.comparison_variable}={comparison_val}"

    def cmap(self):
        if self.algorithm in ['sgd', 'sgd_all_models']:
            cmap = plt.get_cmap('Reds')
            # scale the color map according to the comparison variable
            k, v = self.comparison_variable, self.comparison_variable_value()
            if v is not None:
                if k == 'epochs':
                    if v == 1:
                        cmap = truncate_colormap(cmap, 0.1, 0.3)
                    elif v == 3:
                        cmap = truncate_colormap(cmap, 0.3, 0.5)
                    elif v == 5:
                        cmap = truncate_colormap(cmap, 0.5, 0.7)
                    elif v == 60:
                        cmap = truncate_colormap(cmap, 0.7, 1)
        elif self.algorithm in ['guess', 'guess_all_models']:
            cmap = plt.get_cmap('Blues')
        elif self.algorithm in ['gd']:
            cmap = plt.get_cmap('Oranges')
        else:
            raise ValueError(f"algorithm {self.algorithm} is not supported")
        return cmap

    def color(self):
        cmap = self.cmap()
        k = self.comparison_variable
        v = self.comparison_variable_value()

        if k == 'initialization':
            if self.initialization == 'kaiming':
                v = 0.7
            elif self.initialization == 'kaiming_gaussian':
                v = 0.5
            elif self.initialization == 'uniform02':
                v = 0.3
            elif self.initialization == 'uniform':
                v = 0.1
            else:
                raise ValueError(f"initialization {v} is not supported")
        elif k == 'epochs':
            v = 1
        elif k == 'depth':
            v, _ = self.get_depth_str()
        elif k == 'width':
            _, v, _ = self.get_width_str()
        elif k is None:
            v = 0.5
        else:
            raise ValueError(f"comparison variable {k} is not supported")

        color = v if self.width == 1 else cmap(v)
        return color

    def get_default_str(self):
        if self.num_samples is not None:
            num_samples_str = f"{self.num_samples}samples"
        else:
            num_samples_str = 'multi_samples'

        if 'lenet' in self.arch:
            arch_str = 'lenet_'
        else:
            arch_str = f"{self.arch}_"

        dataset_str = f"{self.dataset}_{self.seed}_{self.permutation_seed}_"

        prefix_str = f'{self.prefix}_' if (self.prefix is not None and self.prefix != '') else ''

        alg_str = self.get_alg_str()

        _, _, width_str = self.get_width_str()

        _, depth_str = self.get_depth_str()

        path_str = f'{prefix_str}{dataset_str}{arch_str}{alg_str}{width_str}{depth_str}{num_samples_str}'

        return path_str

    def get_depth_str(self):

        if isinstance(self.depth, (Sequence,)):
            if self.arch == 'lenet' or self.arch == 'lenet_more_layers':
                if tuple(self.depth) == (0, 1, 2):  # lenet standard, 3 layers
                    depth_str = ''
                    v = int(200)
                elif tuple(self.depth) == (1, 2):  # lenet 2 layers
                    depth_str = f'2c-2f_'
                    v = int(150)
                elif tuple(self.depth) == (2,):  # lenet 1 layers
                    if isinstance(self.conv_layers, (Sequence,)):
                        if tuple(self.conv_layers) == (0, 1):  # lenet standard, 2 conv
                            depth_str = f'2c-1f_'
                            v = int(100)
                        elif tuple(self.conv_layers) == (0,):  # lenet 1 conv
                            depth_str = f'1c-1f_'
                            v = int(50)
                        else:
                            raise ValueError(f"conv {self.conv_layers} is not supported")
                    else:
                        raise ValueError(f"conv {self.conv_layers} is not supported")
                else:
                    raise ValueError(f"depth {self.depth} is not supported for lenet")
            elif self.arch == 'mlp':
                if tuple(self.depth) == (0, 1, 2, 3, 4):  # mlp standard
                    depth_str = ''
                    v = int(200)
                elif tuple(self.depth) == (1, 2, 3, 4):  # mlp 3 hidden layers
                    depth_str = '3layers_'
                    v = int(150)
                elif tuple(self.depth) == (2, 3, 4):  # mlp 2 hidden layers
                    depth_str = '2layers_'
                    v = int(100)
                elif tuple(self.depth) == (3, 4):  # mlp 1 hidden layers
                    depth_str = '1layers_'
                    v = int(50)
                else:
                    raise ValueError(f"depth {self.depth} is not supported for mlp")
            elif self.arch == 'resnet':
                if tuple(self.depth) == (1, 2, 3):  # resnet standard
                    depth_str = ''
                    v = int(200)
                elif tuple(self.depth) == (2, 3):  # resnet 2 layers
                    depth_str = '2blocks_'
                    v = int(150)
                elif tuple(self.depth) == (3,):  # resnet 1 layers
                    depth_str = '1blocks_'
                    v = int(100)
                else:
                    raise ValueError(f"depth {self.depth} is not supported for resnet")
            else:
                raise ValueError(f"arch {self.arch} is not supported")
        elif isinstance(self.depth, (int, float)):
            if self.arch == 'lenet' or self.arch == 'lenet_more_layers':
                if self.depth == 2:  # lenet 1 layers
                    if isinstance(self.conv_layers, (int, float)):
                        if self.conv_layers == 0:  # lenet 1 conv
                            depth_str = f'1c-1f_'
                            v = int(50)
                        else:
                            raise ValueError(f"conv {self.conv_layers} is not supported")
                    elif isinstance(self.conv_layers, (Sequence,)):
                        if tuple(self.conv_layers) == (0, 1):  # lenet standard, 2 conv
                            depth_str = f'2c-1f_'
                            v = int(100)
                    else:
                        raise ValueError(f"conv {self.conv_layers} is not supported")
                else:
                    raise ValueError(f"depth {self.depth} is not supported for lenet")
        else:
            raise ValueError(f"depth {self.depth} is not supported")

        return v, depth_str

    def get_width_str(self):
        if isinstance(self.width, (float, int)) or self.width in ['1/6', '2/6', '3/6', '4/6', '1']:
            if self.width in [1 / 6, 0.165, '1/6']:
                width_str = f'width0165_'
                v = int(150)
                label = r'$\frac{1}{6}$'
            elif self.width in [2 / 6, 0.33, '2/6']:
                width_str = f'width033_'
                v = int(200)
                label = r'$\frac{2}{6}$'
            elif self.width in [3 / 6, 0.5, '3/6']:
                width_str = f'width05_'
                v = int(225)
                label = r'$\frac{3}{6}$'
            elif self.width in [4 / 6, 0.66, '4/6']:
                width_str = f'width066_'
                v = int(250)
                label = r'$\frac{4}{6}$'
            elif self.width in [1, '1']:
                width_str = f'width1_'
                v = '#300b09' if self.algorithm == 'sgd' else '#091c30'
                label = r'$1$'
            else:
                width_str = str(self.width).replace('.', '')
                width_str = f'width{width_str}_'
        else:
            if self.width in [(0.165, 0.04), (1 / 6, 1 / 6, 1 / 24, 1 / 24), ['1/6', '1/6', '1/24', '1/24'],
                              [1 / 6, 1 / 6, 1 / 24, 1 / 24]] and self.arch == 'lenet':
                width_str = f'width0165x2_01x2_'
                v = int(100)
                label = r'$\frac{1}{6}^*$'
            else:
                raise ValueError(f"width {self.width} is not supported")
        return label, v, width_str

    def get_alg_str(self):
        if self.algorithm in ['sgd', 'gd']:
            lr_str = str(self.lr).split('.')[1]
            lr_str = f'lr{lr_str}_'
            if isinstance(self.epochs, int):
                epochs_str = f"epoch{self.epochs}_"
            elif isinstance(self.epochs, float):
                # remove the dot from the float
                epochs_str = str(self.epochs).replace('.', '')
                epochs_str = f"epoch{epochs_str}_"
            else:
                raise NotImplementedError(f"epochs {self.epochs} is not supported")
        elif self.algorithm == 'guess':
            lr_str = ''
            epochs_str = ''
        else:
            raise NotImplementedError(f"algorithm {self.algorithm} is not supported")
        alg_str = f'{self.algorithm}_{self.initialization}_{lr_str}{epochs_str}'
        return alg_str

    @property
    def table_name(self):
        # table_name = self.dataset.split('_')[0]
        dataset = self.dataset.split('_')[0]
        table_name = table_dict.get(dataset, None)
        if table_name is None:
            raise ValueError(f"table_name {table_name} is not supported")

        return table_name

    @property
    def config_dir(self):
        assert hasattr(self, 'fig_name'), f"fig_name is not defined in {self}"
        return CONFIG_ROOT / self.fig_name

    @property
    def expname(self):
        return self.get_default_str()

    def config_path(self, supress_assert=False):
        expname = self.expname
        config_path = self.config_dir / f"{expname}.yaml"
        if not supress_assert:
            assert config_path.exists(), f"config file {config_path} does not exist"
        return config_path

    def __eq__(self, other):
        # check that all the attributes are the same
        return self.__dict__ == other.__dict__

    def equal_except_permutation_seed(self, other):
        # check that all the attributes are the same except the permutation seed
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()
        self_dict.pop('permutation_seed')
        self_dict.pop('multi_samples_res')
        other_dict.pop('permutation_seed')
        other_dict.pop('multi_samples_res')
        return self_dict == other_dict

    @classmethod
    def generate_configs_list(cls, comparison_dict: dict, comparison_variable_name: str, desc: str = None):
        premutation_seed_list = comparison_dict.pop('permutation_seed', None)
        params_dict_list = cls.generate_param_prod_list(comparison_dict)
        return cls.dict_list_to_obj(params_dict_list, premutation_seed_list, comparison_variable_name, desc)

    @classmethod
    def generate_param_prod_list(cls, comparison_variables_dict):
        # Rearranging some of the keys
        combine_two_keys(comparison_variables_dict, 'algorithm', 'initialization')
        combine_two_keys(comparison_variables_dict, 'dataset', 'seed')
        if 'depth' in comparison_variables_dict and 'conv_layers' in comparison_variables_dict:
            combine_two_keys(comparison_variables_dict, 'conv_layers', 'depth')

        return [dict(zip(comparison_variables_dict.keys(), combination)) for combination in
                itertools.product(*comparison_variables_dict.values())]

    @classmethod
    def dict_list_to_obj(cls, params_dict_list, permutation_seed_list, comparison_variable, desc=None):
        if permutation_seed_list is None:
            # Create a list of dictionaries
            run_results_list = [cls(**item, comparison_variable=comparison_variable, desc=desc) for item in
                                params_dict_list]
        else:
            # Create the list of dictionaries grouped by the permutation seed
            # Create the product of each seed and the result list and group the results by the seed
            result_list_grouped = []
            for params_dict in params_dict_list:
                result_list_grouped.append(
                    [dict(**params_dict, permutation_seed=permutation_seed) for permutation_seed in
                     permutation_seed_list])

            run_results_list = []
            for group in result_list_grouped:
                run_results_list.append(
                    RunResultsGroup(
                        [cls(**item, comparison_variable=comparison_variable, desc=desc) for item in group]))

        return run_results_list

    @property
    def results_dir(self):
        results_dir = OUTPUT_ROOT / self.table_name / self.expname
        assert results_dir.exists(), f"results dir {results_dir.absolute()} does not exist"
        return results_dir

    @property
    def summary_path(self):
        summary_path = self.results_dir / suffix_summary
        assert summary_path.exists(), f"summary file {summary_path} does not exist"
        return summary_path

    @property
    def acc_path(self):
        acc_path = self.results_dir / suffix_with_acc
        assert acc_path.exists(), f"acc file {acc_path} does not exist"
        return acc_path

    def get_multi_samples_res(self):
        self.multi_samples_res.res = get_model_stats_short_summary(self.summary_path, verbose=False)
        self.multi_samples_res.r = get_model_stats_summary(self.acc_path, verbose=False)
        self.multi_samples_res.acc_aux = get_model_stats_acc(self.acc_path)  # List[Tuple(samples_num, acc)]
        self.multi_samples_res.acc_stats_mean_std = get_stats_per_sample(
            self.multi_samples_res.acc_aux)

        # res is total summary and r is per model summary where r[i] corresponds to res[i] in number of samples
        # res[i][0]: num_training_samples, res[i][1]: tested_model_count, res[i][2]: perfect_model_count
        # r[i][0]:num_training_samples', r[i][1]:train_loss, r[i][2]:train_loss_weight_normalization, r[i][3]:train_acc,
        # r[i][4]:test_loss, r[i][5]:test_acc, r[i][6]:num_of_models

        for i in range(len(self.multi_samples_res.res)):
            self.multi_samples_res.curr_samples_dict["num_of_samples"] = self.multi_samples_res.res[i][0]
            if self.multi_samples_res.res[i][2] > 0:
                self.multi_samples_res.curr_samples_dict["num_of_models_to_find_good_model"] = \
                    self.multi_samples_res.res[i][1] / self.multi_samples_res.res[i][2]
                self.multi_samples_res.curr_samples_dict["test_acc"] = self.multi_samples_res.r[i][5]
            else:
                self.multi_samples_res.curr_samples_dict["num_of_models_to_find_good_model"] = np.nan
                self.multi_samples_res.curr_samples_dict["test_acc"] = np.nan

            self.multi_samples_res.attempts.append(self.multi_samples_res.curr_samples_dict.copy())

        acc_stats_mean_std = self.multi_samples_res.acc_stats_mean_std
        num_samples = np.array(list(acc_stats_mean_std.keys()))

        mean_std_array = np.array(list(acc_stats_mean_std.values()))  # (num_of_samples, 2)

        model_ratio_mean = np.array(
            [a['num_of_models_to_find_good_model'] for a in self.multi_samples_res.attempts])

        self.multi_samples_res.test_acc_mean = mean_std_array[:, 0]
        self.multi_samples_res.test_acc_std = mean_std_array[:, 1]
        self.multi_samples_res.models_ratio_mean = model_ratio_mean
        self.multi_samples_res.models_ratio_std = np.zeros_like(model_ratio_mean)
        self.multi_samples_res.samples_num = num_samples

    def get_single_samples_results(self, add_num_of_test_predictions):
        model_stats_path = self.acc_path
        res = get_model_stats_full_with_id(model_stats_path, add_num_of_test_predictions)
        data_dict = create_dict_from_rows(res, add_num_of_test_predictions=add_num_of_test_predictions)
        data_dict = load_numpy_arrays_from_folder(folder_path=self.results_dir,
                                                  data_dict=data_dict)

        for sample_num, models_dict in data_dict.items():
            for model_id, model_dict in models_dict.items():
                if model_id == 'test_labels':
                    continue
                predictions = model_dict['test_pred'].argmax(axis=1)
                model_dict['num_of_0_predictions'] = np.sum(predictions == 0).item()
                model_dict['num_of_1_predictions'] = np.sum(predictions == 1).item()

        self.single_samples_num_results = data_dict

    @classmethod
    def run_configuration_obj_from_config_path(cls, config_path, desc=None, comparison_variable=None, prefix=None):
        old_sys_argv = sys.argv
        default_config = copy(get_current_config())
        sys.argv = old_sys_argv.copy()
        sys.argv += ['-C', config_path]
        # reset fastargs config with deepcopy
        set_current_config(deepcopy(default_config))
        parser = argparse.ArgumentParser()
        config = get_current_config()
        config.augment_argparse(parser)
        config.collect_argparse_args(parser)
        config_ns = config.get()

        config_obj = cls.run_config_obj_from_config_ns(config_ns, comparison_variable=comparison_variable, desc=desc,
                                                       prefix=prefix)

        # reset fastargs config with deepcopy
        set_current_config(deepcopy(default_config))
        sys.argv = old_sys_argv.copy()

        return config_obj

    @classmethod
    def run_config_obj_from_config_ns(cls, config_ns, fig_name=None, comparison_variable=None, desc=None,
                                      num_samples=None, prefix=None):
        # convert the string of num_samples to a list of ints
        num_samples_list = [int(num_samples) for num_samples in config_ns.distributed.num_samples.split(',')]
        if num_samples is None and len(num_samples_list) == 1:
            num_samples = num_samples_list[0]
        if hasattr(config_ns.distributed, 'permutation_seed'):
            permutation_seed = config_ns.distributed.permutation_seed
        else:
            permutation_seed = config_ns.distributed.data_seed

        if hasattr(config_ns.distributed, 'model.lenet.rem_layers'):
            rem_layers = config_ns.distributed.model.lenet.rem_layers
        else:
            rem_layers = None

        new_obj = cls(algorithm_initialization=(config_ns.optimizer.name.lower(), config_ns.model.init),
                      arch=config_ns.model.arch,
                      dataset_seed=(config_ns.dataset.name, config_ns.distributed.data_seed),
                      permutation_seed=permutation_seed,
                      epochs=config_ns.optimizer.epochs,
                      lr=config_ns.optimizer.lr,
                      width=config_ns.model.lenet.width,
                      rem_layers=rem_layers,
                      depth=config_ns.model.lenet.fc_layers,
                      conv_layers=config_ns.model.lenet.conv_layers,
                      pooling_layers=config_ns.model.lenet.pooling_layers,
                      num_samples=num_samples,
                      comparison_variable=comparison_variable,
                      desc=desc,
                      fig_name=fig_name,
                      prefix=prefix,
                      )
        assert new_obj.config_path(supress_assert=True).stem == CustomPath(config_ns.output.folder).stem, \
            f"The input --config-file was \n" \
            f"{CustomPath(config_ns.output.folder):link}\n" \
            f"The generated config path is\n" \
            f"{new_obj.config_path(supress_assert=True):link}"
        assert new_obj.config_path(
            supress_assert=True).exists(), f"config file {new_obj.config_path(supress_assert=True)} does not exist"

        return new_obj


# class that hold the results of different permutation seeds with the same config in groups
class RunResultsGroup(RunResultsObject):
    def __init__(self, run_results_list: List[RunResultsObject]):
        # check that all the configs are the same except the permutation seed
        assert all(
            [run_results_list[0].equal_except_permutation_seed(run_results) for run_results in run_results_list[1:]]), \
            "not all the configs are the same except the permutation seed"

        config = run_results_list[0].__dict__.copy()
        super().__init__(**config)
        self.run_results_list = run_results_list
        self.multi_samples_res = BetweenSamplesNumResultsSummary()
        self.single_samples_num_results = None

    def get_multi_samples_res(self):
        for run_results in self.run_results_list:
            run_results.get_multi_samples_res()

        # get all the different permutation seeds results for a specific seed
        different_permseed_runs = [run_results.multi_samples_res.attempts for run_results in self.run_results_list]
        # get the number of different samples of the shorterst run (in case of different number of samples beteween runs)
        minimum_num_runs_per_permseed = min([len(run) for run in different_permseed_runs])

        num_samples_list = []
        test_acc_list = []
        models_ratio_list = []
        for run in different_permseed_runs:
            run_tmp = run[:minimum_num_runs_per_permseed]
            # get the number of samples for each of the permutation seeds
            num_samples_list.append([r['num_of_samples'] for r in run_tmp])
            # get the test accuracy for each of the permutation seeds
            test_acc_list.append([r['test_acc'] for r in run_tmp])
            # get the number of attempts to get a model with 100% train accuracy
            models_ratio_list.append([r['num_of_models_to_find_good_model'] for r in run_tmp])

        assert all([num_samples_list[0] == num_samples for num_samples in num_samples_list[1:]]), \
            "not all the runs have the same number of samples"

        self.multi_samples_res.samples_num = num_samples_list[0]

        test_acc_array = np.array(test_acc_list)  # (num_of_runs, num_of_samples)
        self.multi_samples_res.test_acc_mean = np.mean(test_acc_array, axis=0)  # (num_of_samples,)
        self.multi_samples_res.test_acc_std = np.std(test_acc_array, axis=0)  # (num_of_samples,)

        models_ratio_array = np.array(models_ratio_list)  # (num_of_runs, num_of_samples)
        self.multi_samples_res.models_ratio_mean = np.mean(np.log2(models_ratio_array), axis=0)  # (num_of_samples,)
        self.multi_samples_res.models_ratio_std = np.std(np.log2(models_ratio_array), axis=0)  # (num_of_samples,)

        self.multi_samples_res.acc_aux = [run.multi_samples_res.acc_aux for run in
                                          self.run_results_list]  # List[List[Tuple(samples_num, acc)]] number of permutation seeds

    def get_single_samples_results(self, add_num_of_test_predictions):
        assert len(self.run_results_list) == 1, "get_single_samples_results should be called only for one seed"
        for run_results in self.run_results_list:
            run_results.get_single_samples_results(add_num_of_test_predictions)

        self.single_samples_num_results = self.run_results_list[0].single_samples_num_results
