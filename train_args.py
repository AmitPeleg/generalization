from fastargs import Section, Param
from fastargs.validation import OneOf, Or, Checker


class Fraction:
    """
    A new type for fractions in the shape of float or a/b.
    """

    def __init__(self, value):
        if isinstance(value, str):
            if '/' in value:
                value = value.split('/')
                self.value = float(value[0]) / float(value[1])
            else:
                self.value = float(value)
        else:
            self.value = float(value)

    def __str__(self):
        return str(self.value)


class SequenceChecker(Checker):
    def __init__(self, length=None, dtype=float):
        self.len = length
        self.dtype = dtype

    def check(self, value):
        if isinstance(value, str):
            value = tuple([self.dtype(idx) for idx in value.split(',')])
        else:
            value = tuple([self.dtype(idx) for idx in value])

        if self.len is not None:
            assert len(value) == self.len

        return value

    def help(self):
        return "a sequence"


class FractionChecker(Checker):
    """
    A checker for fractions in the shape of float or a/b.
    """

    def check(self, value):
        # if it is list check for every value in the list
        if isinstance(value, list):
            for i, val in enumerate(value):
                if isinstance(val, str):
                    if '/' in val:
                        val = val.split('/')
                        value[i] = float(val[0]) / float(val[1])
                    else:
                        value[i] = float(val)
        elif isinstance(value, str):
            if '/' in value:
                value = value.split('/')
                value = float(value[0]) / float(value[1])
            else:
                value = float(value)
        else:
            value = float(value)
        return value

    def help(self):
        return "a fraction"


Section("dataset", "Dataset parameters").params(name=Param(str, OneOf(("mnist", "cifar10")), default="mnist"))

Section("dataset.mnistcifar", "Dataset parameters for mnist/cifar").params(num_classes=Param(int))

Section("model", "Model architecture parameters").params(
    arch=Param(str, OneOf(("mlp", "lenet", "lenet_more_layers", "resnet4")), default="lenet"),
    model_count_times_batch_size=Param(int, default=20000 * 16),
    init=Param(str, OneOf(("uniform", "uniform02", "kaiming", "kaiming_gaussian")), default="uniform"),
    zero_bias=Param(bool, default=True)
)

Section("model.lenet", "Model architecture parameters").params(
    width=Param(Or(FractionChecker(), SequenceChecker(length=4, dtype=Fraction))),
    feature_dim=Param(float),
    kernel_size=Param(int, default=5),
    pooling_layers=Param(Or(int, SequenceChecker(dtype=int)), default=(0, 1)),
    conv_layers=Param(Or(int, SequenceChecker(dtype=int)), default=(0, 1)),
    fc_layers=Param(Or(int, SequenceChecker(dtype=int)), default=(0, 1, 2)),
    rem_layers=Param(SequenceChecker(dtype=str), default=()),
)
Section("model.mlp", "Model architecture parameters").enable_if(lambda cfg: cfg['model.arch'] == 'mlp').params(
    hidden_units=Param(int),
    layers=Param(int)
)

Section("optimizer").params(
    name=Param(str, OneOf(["SGD", "guess", "GD"]), default='guess'),
    lr=Param(float, desc='learning rate', default=0.01),
    momentum=Param(float, desc='momentum', default=0),
    epochs=Param(int, desc='number of epochs to optimize  for', default=60),
    batch_size=Param(int, desc='number of samples in the batch', default=2),
    scheduler=Param(str, desc='whether to use a scheduler', default=None)
)

Section("distributed").params(
    num_samples=Param(str, default="2,4"),
    training_seed=Param(int, default=None,
                        desc='If there is no training seed, then the training seed increments with every run'),
    data_seed=Param(int, default=None,
                    desc='If there is no data seed, then the data seed increment with every run'),
    permutation_seed=Param(int, default=None,
                           desc='If there is no permutation seed, then permutation seed equals to the data_seed'),
)

Section("output", "arguments associated with output").params(
    successful_model_count=Param(int, default=1),
    save_predictions=Param(int, default=False),
    save_weights=Param(int, default=False),
    save_normalized_loss_per_epoch=Param(int, default=False),
    folder=Param(str, default='debug'),
    load_models_from_previous_num_of_samples=Param(bool, default=False),
)
