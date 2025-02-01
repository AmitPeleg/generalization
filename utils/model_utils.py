from collections import namedtuple

import numpy as np
from torch import nn

"""
https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
Based on https://github.com/davidcpage/cifar10-fast/blob/master/experiments.ipynb
"""
from functools import partial

import torch


def get_model(config, model_count, device):
    arch = config['model.arch']
    if arch == "mlp":
        model = MLPModels(output_dim=config['dataset.mnistcifar.num_classes'],
                          width_factor=config['model.lenet.width'],
                          model_count=model_count,
                          dataset=config['dataset.name'],
                          pooling_layers=config['model.lenet.pooling_layers'],
                          fc_layers=config['model.lenet.fc_layers'],
                          feature_dim=config['model.lenet.feature_dim']).to(device)
    elif arch == "lenet_more_layers":
        model = ModifiableLeNetModels(output_dim=config['dataset.mnistcifar.num_classes'],
                                      width_factor=config['model.lenet.width'],
                                      model_count=model_count,
                                      dataset=config['dataset.name'],
                                      conv_layers=config['model.lenet.conv_layers'],
                                      kernel_size=config['model.lenet.kernel_size'],
                                      pooling_layers=config['model.lenet.pooling_layers'],
                                      fc_layers=config['model.lenet.fc_layers'],
                                      feature_dim=config['model.lenet.feature_dim']).to(device)
    elif arch == "lenet":
        model = LeNetModels(output_dim=config['dataset.mnistcifar.num_classes'],
                            width_factor=config['model.lenet.width'],
                            model_count=model_count,
                            dataset=config['dataset.name'],
                            feature_dim=config['model.lenet.feature_dim']).to(device)
    elif arch == "resnet4":
        momentum = 1 if config["optimizer.name"] == 'guess' else 0.1
        model = Resnet4(output_dim=config['dataset.mnistcifar.num_classes'],
                        width_factor=config['model.lenet.width'],
                        model_count=model_count,
                        dataset=config['dataset.name'],
                        rem_layers=config['model.lenet.rem_layers'],
                        feature_dim=config['model.lenet.feature_dim'],
                        momentum=momentum).to(device)
    else:
        raise ValueError(f"Unknown model {arch}")

    return model


class Repeat(nn.Module):
    def __init__(self, num_repeats):
        super().__init__()
        self.num_repeats = num_repeats

    def forward(self, data):
        data = data.repeat(1, self.num_repeats, 1, 1)
        return data


class LeNetModels(nn.Module):
    def __init__(self, output_dim, width_factor, model_count, dataset, feature_dim=None, kernel_size=5):
        super(LeNetModels, self).__init__()
        self.model_count = model_count
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        if isinstance(width_factor, float):
            width_factor = (width_factor,) * 4  # 4 is the number of layers - 1
        self.width_factor = width_factor
        self.dataset = dataset
        self.repeat = Repeat(model_count)

        self.feature_dim = self.get_feature_dim(feature_dim, width_factor)

        if dataset == "cifar10":
            self.conv1_in_channels = 3
        elif dataset == "mnist":
            self.conv1_in_channels = 1
        else:
            raise ValueError(f"dataset {dataset} not supported")

        self.conv1 = None
        self.conv2 = None
        self.conv_layers_sequential = None
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.fc_layers_sequential = None
        self.relu_fc1 = None
        self.relu_fc2 = None

        self.build_original_lenet_arch(model_count, output_dim, width_factor)

    def get_feature_dim(self, feature_dim, width_factor):
        if feature_dim is None:
            return int(84 * width_factor[3])

        return feature_dim

    def build_original_lenet_arch(self, model_count, output_dim, width_factor):
        # Conv1
        self.conv1 = nn.Conv2d(self.conv1_in_channels * model_count,
                               int(6 * width_factor[0]) * model_count,
                               kernel_size=self.kernel_size,
                               groups=model_count
                               )
        # Conv2
        self.conv2 = nn.Conv2d(int(6 * width_factor[0]) * model_count,
                               int(16 * width_factor[1]) * model_count,
                               kernel_size=self.kernel_size,
                               groups=model_count)
        # Convolutions layers
        self.conv_layers_sequential = nn.Sequential(
            self.conv1, nn.ReLU(), nn.MaxPool2d(2),
            self.conv2, nn.ReLU(), nn.MaxPool2d(2)
        )
        # calculate spatial dimension after convolutions
        spatial_dim = self.calc_fc1_spatial_dim(conv_layers=(0, 1), pooling_layers=(0, 1))
        # FC1
        self.fc1 = nn.Conv2d(int(16 * width_factor[1]) * (spatial_dim ** 2) * model_count,
                             int(120 * width_factor[2]) * model_count,
                             1,
                             groups=model_count)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Conv2d(int(120 * width_factor[2]) * model_count,
                             int(self.feature_dim * model_count),
                             1,
                             groups=model_count)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Conv2d(int(self.feature_dim * model_count),
                             output_dim * model_count,
                             1,
                             groups=model_count)
        # fc layers
        self.fc_layers_sequential = nn.Sequential(
            self.fc1, self.relu_fc1,
            self.fc2, self.relu_fc2,
            self.fc3
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        # Manipulate the state_dict in case that fc_layers_sequential is not in the state_dict
        if not any(['fc_layers_sequential' in name for name in state_dict.keys()]):
            # Copy the fc layers into fc_layers_sequential
            # In the lenet case, there are 3 fc layers fc1, fc2, fc3
            # The sequential layers are 0, 2, 4, because of the relu layers
            for i in [0, 1, 2]:
                # Copy the weight and bias of the fc layer into fc_layers_sequential
                state_dict[f'fc_layers_sequential.{i * 2}.weight'] = state_dict[f'fc{i + 1}.weight']
                state_dict[f'fc_layers_sequential.{i * 2}.bias'] = state_dict[f'fc{i + 1}.bias']
        super().load_state_dict(state_dict, strict)

    def calc_fc1_spatial_dim(self, conv_layers, pooling_layers):
        if self.dataset == "cifar10":
            spatial_dim = 32
        elif self.dataset == "mnist":
            spatial_dim = 28
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        last_conv_layer = conv_layers[-1]
        last_pooling_layer = pooling_layers[-1]
        last_layer = max(last_conv_layer, last_pooling_layer) + 1

        for layer in range(last_layer):
            if layer in conv_layers:
                spatial_dim -= self.kernel_size - 1
            if layer in pooling_layers:
                spatial_dim /= 2
        return int(spatial_dim)

    def forward(self, x):
        # the forward takes in x shaped [# of examples, input_dim, H, W]
        # outputs [# of examples, model_count, logit_count]
        x = self.repeat(x)

        out = self.conv_layers_sequential(x)

        out = out.reshape(out.size(0), -1, 1, 1)
        out = self.fc_layers_sequential(out)
        out = out.view(out.size(0), self.model_count, self.output_dim)  # (batch_size, model_count, output_dim)
        return out

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def get_model_subsets(self, idx):
        if isinstance(idx, slice):
            current_model_count = self.model_count
            new_model_count = len(range(*idx.indices(current_model_count)))
        else:
            new_model_count = len(idx)
        # Generate a new model with the same parameters with the correct class
        new_model = type(self)(**self.get_kwargs_for_new_model(new_model_count))
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model.to(self.device)

    def get_kwargs_for_new_model(self, new_model_count):
        return {
            'output_dim': self.output_dim,
            'width_factor': self.width_factor,
            'model_count': new_model_count,
            'dataset': self.dataset,
            'feature_dim': self.feature_dim,
            'kernel_size': self.kernel_size,
        }

    @torch.no_grad()
    def reinitialize(self, mult=1, zero_bias=True):
        for name, para in self.named_parameters():
            if 'bn' in name:
                if 'bias' in name:
                    torch.nn.init.uniform_(para.data, a=-0.05, b=0.05)
                elif 'weight' in name:
                    torch.nn.init.uniform_(para.data, a=0.9, b=1.1)
                else:
                    raise ValueError(f"Unknown parameter {name}")
            else:
                if zero_bias and 'bias' in name:
                    torch.nn.init.zeros_(para.data)
                else:
                    torch.nn.init.uniform_(para.data, a=-mult, b=mult)

    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        weight_dict = {}
        for name, para in self.state_dict().items():
            original_shape = para.shape
            if len(original_shape) == 0:
                weight_dict[name] = para
            else:
                para_reshaped = para.reshape(self.model_count, -1, *original_shape[2:])
                para_selected = para_reshaped[idx]
                para_selected = para_selected.reshape(-1, *original_shape[1:])
                weight_dict[name] = para_selected.clone().detach().cpu()

        return weight_dict

    def __getitem__(self, item):
        return self.get_model_subsets(item)

    @torch.no_grad()
    def get_weights_by_idx_for_save(self, idx):
        weight_dict = {}
        for name, para in self.state_dict().items():
            original_shape = para.shape
            if len(original_shape) == 0:
                weight_dict[name] = np.full((self.model_count, 1), para.item())
            else:
                para_reshaped = para.reshape(self.model_count, -1, *original_shape[1:])
                para_selected = para_reshaped[idx]
                weight_dict[name] = para_selected.clone().detach().cpu().numpy()

        return weight_dict

    @torch.no_grad()
    def forward_normalize(self, x):
        x = self.forward(x)
        cum_norm = 1
        # for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
        for name, weight in self.named_parameters():
            if 'bias' in name:
                continue
            if 'bn' in name:
                continue
            cur_weight = weight
            original_shape = cur_weight.shape
            cur_weight = cur_weight.view(self.model_count, -1, *original_shape[2:])
            cur_norm = cur_weight.norm(p=2, dim=(1, 2, 3), keepdim=True) / 3
            cum_norm *= cur_norm.view(self.model_count, -1)
        x /= cum_norm
        return x

    def forward_normalize_mult(self, x):
        x = self.forward(x)  # (examples, model_count, output_dim)
        cum_norm = 1
        for name, weight in self.named_parameters():
            if 'bias' in name:
                continue
            if 'bn' in name:
                continue
            cur_weight = weight
            original_shape = cur_weight.shape
            cur_weight = cur_weight.view(self.model_count, -1, *original_shape[2:])
            cur_norm = cur_weight.norm(p=2, dim=(1, 2, 3), keepdim=True) / 3
            cum_norm *= cur_norm.view(self.model_count, -1)
        return x, x / cum_norm


class ModifiableLeNetModels(LeNetModels):
    def __init__(self, output_dim, width_factor, model_count, dataset, feature_dim=None, conv_layers=(0, 1),
                 kernel_size=5, pooling_layers=(0, 1), fc_layers=(0, 1, 2)):
        super().__init__(output_dim, width_factor, model_count, dataset, feature_dim, kernel_size=kernel_size)
        """total_conv_layers: The total number of convolutional layers in the model. The first two layers are the 
        ones in the default LeNet model. kernel_size: The kernel size for all convolutional layers, default in the 
        LeNet model is 5. pooling_layers: A tuple of the indices of the convolutional layers to add max pooling 
        after. The default in LeNet is in the first two layers (conv1 and conv2)."""

        self.kernel_size = kernel_size

        self.pooling_layers = pooling_layers if isinstance(pooling_layers, tuple) else (pooling_layers,)
        self.conv_layers = conv_layers if isinstance(conv_layers, tuple) else (conv_layers,)
        self.fc_layers = fc_layers if isinstance(fc_layers, tuple) else (fc_layers,)

        if isinstance(width_factor, float):
            self.width_factor = (width_factor,) * (len(self.conv_layers) + len(self.fc_layers))  # number of layers

        self.build_conv_layers()

        self.build_fc_layers()

    def get_feature_dim(self, feature_dim, width_factor):
        if feature_dim is None:
            return int(84 * width_factor[-1])

        return feature_dim

    def build_original_lenet_arch(self, model_count, output_dim, width_factor):
        pass

    def build_conv_layers(self):
        add_conv_modules_list = self.create_additional_conv_layers()
        add_layers_list = []
        last_conv_layer = self.conv_layers[-1]
        last_pooling_layer = self.pooling_layers[-1]
        last_layer = max(last_conv_layer, last_pooling_layer) + 1
        for i in range(last_layer):
            if i in self.conv_layers:
                add_layers_list.append(add_conv_modules_list.pop(0))
                add_layers_list.append(nn.ReLU())
            if i in self.pooling_layers:
                add_layers_list.append(nn.MaxPool2d(2))
        self.conv_layers_sequential = nn.Sequential(*add_layers_list)
        # After creating sequential, remove the additional layers from the model (were only created in the first
        # place to use the basic LeNet model )

    def build_fc_layers(self):
        spatial_dim = self.calc_fc1_spatial_dim(self.conv_layers, self.pooling_layers)
        # Change the number of input channels to the first fully connected layer, based on the number of convolutional
        last_conv_out_channels = self.num_layers_fn(self.conv_layers[-1] + 1)

        # Change fc layers
        if self.fc_layers == (0, 1, 2):
            # 3 fc layers - default
            raise NotImplementedError("Should go to lenet")

        elif self.fc_layers == (1, 2):
            self.fc2 = nn.Conv2d(last_conv_out_channels * (spatial_dim ** 2),
                                 int(self.feature_dim * self.model_count),
                                 1,
                                 groups=self.model_count)
            self.fc3 = nn.Conv2d(int(self.feature_dim * self.model_count),
                                 self.output_dim * self.model_count,
                                 1,
                                 groups=self.model_count)
            self.fc_layers_sequential = nn.Sequential(self.fc2, nn.ReLU(), self.fc3)
        elif self.fc_layers in [(2,)]:
            # Change the number of output channels of fc1 to be the same as the number of input channels of fc2
            self.fc3 = nn.Conv2d(last_conv_out_channels * (spatial_dim ** 2),
                                 self.output_dim * self.model_count,
                                 1,
                                 groups=self.model_count)
            self.fc_layers_sequential = nn.Sequential(self.fc3, )
        else:
            raise NotImplementedError(f"fc_layers {self.fc_layers} not supported")

    def num_layers_fn(self, i):
        if i == 0:
            return self.conv1_in_channels * self.model_count
        if i == 1:
            return int(6 * self.width_factor[0]) * self.model_count
        return int(16 * (i - 1) * self.width_factor[1]) * self.model_count

    def create_additional_conv_layers(self):
        if self.conv_layers == (-1,):  # no additional layers
            return []
        add_layers_list = []
        for i in self.conv_layers:
            add_layers_list.append(nn.Conv2d(in_channels=self.num_layers_fn(i),
                                             out_channels=self.num_layers_fn(i + 1),
                                             kernel_size=self.kernel_size,
                                             groups=self.model_count))
        return add_layers_list

    def get_kwargs_for_new_model(self, new_model_count):
        kwargs = super().get_kwargs_for_new_model(new_model_count)
        kwargs['conv_layers'] = self.conv_layers
        kwargs['pooling_layers'] = self.pooling_layers
        kwargs['fc_layers'] = self.fc_layers
        return kwargs

    def load_state_dict(self, state_dict, strict: bool = True):
        # Manipulate the state_dict in case that fc_layers_sequential is not in the state_dict
        if not any(['fc_layers_sequential' in name for name in state_dict.keys()]):
            raise NotImplementedError()
        super().load_state_dict(state_dict, strict)


class MLPModels(ModifiableLeNetModels):
    def __init__(self, output_dim, width_factor, model_count, dataset, feature_dim=None, pooling_layers=(),
                 fc_layers=(0, 1, 2)):
        super().__init__(output_dim, width_factor, model_count, dataset, feature_dim=feature_dim, conv_layers=(),
                         kernel_size=5, pooling_layers=pooling_layers, fc_layers=fc_layers)

    def get_kwargs_for_new_model(self, new_model_count):
        return {
            'output_dim': self.output_dim,
            'width_factor': self.width_factor,
            'model_count': new_model_count,
            'dataset': self.dataset,
            'fc_layers': self.fc_layers,
            'pooling_layers': self.pooling_layers
        }

    def calc_fc1_spatial_dim(self, conv_layers, pooling_layers):
        if self.dataset == "cifar10":
            spatial_dim = 32
        elif self.dataset == "mnist":
            spatial_dim = 28
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        for layer in pooling_layers:
            spatial_dim /= 2
        return int(spatial_dim)

    def calc_fc1_num_channels(self):
        if self.dataset == "cifar10":
            num_channels = 3
        elif self.dataset == "mnist":
            num_channels = 1
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")
        return num_channels

    def build_conv_layers(self):
        layers = []

        for i in self.pooling_layers:
            layers.append(nn.MaxPool2d(2))
            # layers.append(nn.AvgPool2d(2))
        self.conv_layers_sequential = nn.Sequential(*layers)

    def build_fc_layers(self):
        spatial_dim = self.calc_fc1_spatial_dim(self.conv_layers, self.pooling_layers)
        # Change the number of input channels to the first fully connected layer, based on the number of convolutional
        input_channels = self.calc_fc1_num_channels()

        layers_channels_list = [120, 60, 30, 12]

        layers = []

        if len(self.fc_layers) == 1:
            layers.append(nn.Conv2d(input_channels * (spatial_dim ** 2) * self.model_count,
                                    self.output_dim * self.model_count,
                                    1,
                                    groups=self.model_count))
            self.fc_layers_sequential = nn.Sequential(*layers)
            return

        layers += [nn.Conv2d(input_channels * (spatial_dim ** 2) * self.model_count,
                             int(layers_channels_list[self.fc_layers[0]] * self.width_factor[0]) * self.model_count,
                             1,
                             groups=self.model_count),
                   nn.ReLU()]

        for i, layer_idx in enumerate(self.fc_layers[1:-1]):
            layers.append(
                nn.Conv2d(int(layers_channels_list[layer_idx - 1] * self.width_factor[i - 1]) * self.model_count,
                          int(layers_channels_list[layer_idx] * self.width_factor[i]) * self.model_count,
                          1,
                          groups=self.model_count))
            layers.append(nn.ReLU())

        layers.append(
            nn.Conv2d(int(layers_channels_list[self.fc_layers[-1] - 1] * self.width_factor[-1]) * self.model_count,
                      self.output_dim * self.model_count,
                      1,
                      groups=self.model_count))

        self.fc_layers_sequential = nn.Sequential(*layers)


#####################
# Resnet4
#####################
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            self.weight.data.fill_(weight_init)
        if bias_init is not None:
            self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze


sep = '/'

# Resnet4 definitions
batch_norm = partial(BatchNorm, weight_init=None, bias_init=None, momentum=0.1)


class Resnet4(ModifiableLeNetModels):
    def __init__(self, output_dim, width_factor, model_count, dataset, rem_layers=(), feature_dim=None, momentum=0.1):
        if isinstance(width_factor, (float, int)):
            width_factor = (width_factor,) * 4  # 4 is the number of layers - 1
        super().__init__(output_dim, width_factor, model_count, dataset, feature_dim, conv_layers=(), kernel_size=1,
                         pooling_layers=(), fc_layers=())
        self.momentum = momentum
        channels = {'prep': int(6 * width_factor[0]),
                    'layer1': int(12 * width_factor[1]),
                    'layer2': int(24 * width_factor[2]),
                    'layer3': int(48 * width_factor[3])}
        n = net(dataset=dataset, channels=channels, model_count=model_count, output_dim=output_dim,
                rem_layers=rem_layers, momentum=momentum)
        self.graph = build_graph(n)
        for path, (val, _) in self.graph.items():
            setattr(self, path.replace('/', '_'), val)
        self.model_count = model_count
        self.rem_layers = rem_layers

    def nodes(self):
        return (node for node, _ in self.graph.values())

    def forward(self, data):
        data = self.repeat(data)
        inputs = {'input': data}
        outputs = dict(inputs)

        for k, (node, ins) in self.graph.items():
            # only compute nodes that are not supplied as inputs.
            if k not in outputs:
                outputs[k] = node(*[outputs[x] for x in ins])
        out = outputs['logits']
        out = out.view(out.size(0), self.model_count, self.output_dim)  # (batch_size, model_count, output_dim)
        return out

    def build_conv_layers(self):
        pass

    def build_fc_layers(self):
        pass

    def get_kwargs_for_new_model(self, new_model_count):
        return {
            'output_dim': self.output_dim,
            'width_factor': self.width_factor,
            'model_count': new_model_count,
            'dataset': self.dataset,
            'feature_dim': self.feature_dim,
            'rem_layers': self.rem_layers,
            'momentum': self.momentum
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        # Bypassing manipulation of the parent class load_state_dict method
        nn.Module.load_state_dict(self, state_dict, strict)


class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x


class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)


class Add(namedtuple('Add', [])):
    def __call__(self, x, y): return x + y


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


def split(path):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]


def res_block(c_in, c_out, stride, model_count, **kw):
    block = {
        'bn1': batch_norm(c_in * model_count, **kw),
        'relu1': nn.ReLU(True),
        'branch': {
            'conv1': nn.Conv2d(c_in * model_count, c_out * model_count, kernel_size=3, stride=stride, padding=1,
                               bias=False, groups=model_count),
            'bn2': batch_norm(c_out * model_count, **kw),
            'relu2': nn.ReLU(True),
            'conv2': nn.Conv2d(c_out * model_count, c_out * model_count, kernel_size=3, stride=1, padding=1, bias=False,
                               groups=model_count),
        }
    }
    projection = (stride != 1) or (c_in != c_out)
    if projection:
        block['conv3'] = (
            nn.Conv2d(c_in * model_count, c_out * model_count, kernel_size=1, stride=stride, padding=0, bias=False,
                      groups=model_count), ['relu1'])
    block['add'] = (Add(), [('conv3' if projection else 'relu1'), 'branch/conv2'])
    return block


def union(*dicts):
    return {k: v for d in dicts for (k, v) in d.items()}


def DAWN_net(model_count, c=64, output_dim=10, block=res_block, prep_bn_relu=False, concat_pool=True, **kw):
    if isinstance(c, int):
        c = [c, 2 * c, 4 * c, 4 * c]

    classifier_pool = {
        'in': Identity(),
        'maxpool': nn.MaxPool2d(4),
        'avgpool': (nn.AvgPool2d(4), ['in']),
        'concat': (Concat(), ['maxpool', 'avgpool']),
    } if concat_pool else {'pool': nn.MaxPool2d(4)}

    return {
        'input': (None, []),
        'prep': union({'conv': nn.Conv2d(3 * model_count, c[0] * model_count, kernel_size=3, stride=1, padding=1,
                                         bias=False, groups=model_count)},
                      {'bn': batch_norm(c[0] * model_count, **kw), 'relu': nn.ReLU(True)} if prep_bn_relu else {}),
        'layer1': {
            'block0': block(c[0], c[0], 1, model_count, **kw),
            'block1': block(c[0], c[0], 1, model_count, **kw),
        },
        'layer2': {
            'block0': block(c[0], c[1], 2, model_count, **kw),
            'block1': block(c[1], c[1], 1, model_count, **kw),
        },
        'layer3': {
            'block0': block(c[1], c[2], 2, model_count, **kw),
            'block1': block(c[2], c[2], 1, model_count, **kw),
        },
        'layer4': {
            'block0': block(c[2], c[3], 2, model_count, **kw),
            'block1': block(c[3], c[3], 1, model_count, **kw),
        },
        'final': union(classifier_pool, {
            'flatten': Flatten(),
            'linear': nn.Conv2d(in_channels=2 * c[3] * model_count if concat_pool else c[3] * model_count,
                                out_channels=output_dim * model_count,
                                bias=True,
                                kernel_size=1,
                                groups=model_count),
        }),
        'logits': Identity(),
    }


def conv_bn(c_in, c_out, model_count, bn_weight_init=1.0, momentum=0.1, **kw):
    return {
        'conv': nn.Conv2d(c_in * model_count, c_out * model_count, kernel_size=3, stride=1, padding=1, bias=False,
                          groups=model_count),
        'bn': batch_norm(c_out * model_count, weight_init=bn_weight_init, momentum=momentum, **kw),
        # 'bn': Identity(),
        'relu': nn.ReLU(True)
    }


def basic_net(dataset, model_count, output_dim, channels, weight, pool, momentum=0.1, **kw):
    input_dim, last_pool = get_input_dim_last_pool(dataset)

    return {
        'input': (None, []),
        'prep': conv_bn(input_dim, channels['prep'], model_count, momentum=momentum, **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], model_count, momentum=momentum, **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], model_count, momentum=momentum, **kw),
                       pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], model_count, momentum=momentum, **kw),
                       pool=pool),
        'pool': nn.MaxPool2d(last_pool),
        'flatten': Flatten(),
        'linear': nn.Conv2d(channels['layer3'] * model_count, output_dim * model_count, bias=False, kernel_size=(1, 1),
                            groups=model_count),
        'logits': Mul(weight),
    }


def get_input_dim_last_pool(dataset):
    if dataset == 'cifar10':
        input_dim = 3
        last_pool = 4
    elif dataset == 'mnist':
        input_dim = 1
        last_pool = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return input_dim, last_pool


def last_layer_correction(last_channel_layers, model_count, dataset, rem_layers):
    if dataset == 'cifar10':
        last_layer_coeff = 4 ** len(rem_layers)
    elif dataset == 'mnist':
        if len(rem_layers) == 1:
            last_layer_coeff = 9
        elif len(rem_layers) == 2:
            last_layer_coeff = 49
        else:
            raise ValueError(f"Unknown number of layers to remove: {len(rem_layers)}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return last_channel_layers * model_count * last_layer_coeff


def net(dataset, model_count, output_dim, channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(),
        res_layers=('layer1', 'layer3'), rem_layers=(), momentum=0.1, **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    residual = lambda c, **kw: {'in': Identity(), 'res1': conv_bn(c, c, model_count, momentum=momentum, **kw),
                                'res2': conv_bn(c, c, model_count, momentum=momentum, **kw),
                                'add': (Add(), ['in', 'res2/relu'])}
    n = basic_net(dataset, model_count, output_dim, channels, weight, pool, momentum=momentum, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], model_count, momentum=momentum, **kw)

    # remove layers
    for layer in rem_layers:
        del n[layer]
    if len(rem_layers) > 0:
        last_layer = list(filter(lambda x: x.startswith('layer'), n.keys()))[-1]
        input_dim_last_layer = last_layer_correction(channels[last_layer], model_count, dataset, rem_layers)
        n['linear'] = nn.Conv2d(input_dim_last_layer, output_dim * model_count, bias=False, kernel_size=(1, 1),
                                groups=model_count)
        # n['linear'] = nn.Conv2d(channels[last_layer]*model_count*last_layer_pool**len(rem_layers), output_dim*model_count, bias=False, kernel_size=(1,1), groups=model_count)
    if 'layer1' in rem_layers:
        if 'layer2' in rem_layers:
            n['layer3'] = dict(conv_bn(channels['prep'], channels['layer3'], model_count, momentum=momentum, **kw),
                               pool=pool)
        else:
            n['layer2'] = dict(conv_bn(channels['prep'], channels['layer2'], model_count, momentum=momentum, **kw),
                               pool=pool)

    return n


def normpath(path):
    # simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == '..':
            parts.pop()
        elif p.startswith(sep):
            parts = [p]
        else:
            parts.append(p)
    return sep.join(parts)


def has_inputs(node):
    return type(node) is tuple


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


def pipeline(net):
    return [(sep.join(path), (node if has_inputs(node) else (node, [-1]))) for (path, node) in path_iter(net)]


def build_graph(net):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: normpath(sep.join((path, '..', rel_path))) if isinstance(rel_path,
                                                                                                         str) else \
        flattened[idx + rel_path][0]
    return {path: (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]]) for idx, (path, node) in
            enumerate(flattened)}


def new_models_extractor(config, num_perfect_model):
    if config['model.arch'] == "mlp":
        kwargs = {"output_dim": config['dataset.mnistcifar.num_classes'],
                  "width_factor": config['model.lenet.width'],
                  "model_count": num_perfect_model,
                  "dataset": config['dataset.name'],
                  "pooling_layers": config['model.lenet.pooling_layers'],
                  "fc_layers": config['model.lenet.fc_layers'],
                  }
        new_models = MLPModels(**kwargs)
    elif config['model.arch'] == "lenet":
        kwargs = {"output_dim": config['dataset.mnistcifar.num_classes'],
                  "width_factor": config['model.lenet.width'],
                  "model_count": num_perfect_model,
                  "dataset": config['dataset.name'],
                  "feature_dim": config['model.lenet.feature_dim']}
        new_models = LeNetModels(**kwargs)
    elif config['model.arch'] == "lenet_more_layers":
        kwargs = {"output_dim": config['dataset.mnistcifar.num_classes'],
                  "width_factor": config['model.lenet.width'],
                  "model_count": num_perfect_model,
                  "dataset": config['dataset.name'],
                  "feature_dim": config['model.lenet.feature_dim'],
                  "kernel_size": config['model.lenet.kernel_size'],
                  "pooling_layers": config['model.lenet.pooling_layers'],
                  "conv_layers": config['model.lenet.conv_layers'],
                  "fc_layers": config['model.lenet.fc_layers']}
        new_models = ModifiableLeNetModels(**kwargs)
    elif config['model.arch'] == "resnet4":
        momentum = 1 if config["optimizer.name"] == 'guess' else 0.1
        kwargs = {"output_dim": config['dataset.mnistcifar.num_classes'],
                  "width_factor": config['model.lenet.width'],
                  "model_count": num_perfect_model,
                  "dataset": config['dataset.name'],
                  "rem_layers": config['model.lenet.rem_layers'],
                  "feature_dim": config['model.lenet.feature_dim'],
                  "momentum": momentum}
        new_models = Resnet4(**kwargs)
    else:
        raise ValueError(f"model arch {config['model.arch']} not recognized")
    return new_models
