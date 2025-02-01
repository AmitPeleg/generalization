import argparse
import os
import sys
from functools import partial

import torch
from fastargs import get_current_config
from torch import nn

import train_args
from utils.model_utils import new_models_extractor
from utils.sql import *

dummy_obj = train_args.Fraction(0.5)  # in order that the import of train_args will not be removed


def train_sgd(
        train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, batch_size, epochs,
        per_sample_normalized_logits=None):
    for epoch in range(epochs):
        if per_sample_normalized_logits is not None:
            with SetBNSetRunningStats(model, new_mode=False, name="calculating normalized logits"):
                per_sample_normalized_logits["logits"].append(
                    calculate_normalized_logits(train_data, train_labels, model))
                print(
                    f"before epoch {epoch} - normalized logits difference mean: {per_sample_normalized_logits['logits'][epoch].mean().cpu().detach().item(): 0.4f}")
        idx_list = torch.randperm(len(train_data))

        for st_idx in range(0, len(train_data), batch_size):
            idx = idx_list[st_idx:min(st_idx + batch_size, len(train_data))]
            train_loss, train_acc = calculate_loss_acc(train_data[idx], train_labels[idx], model, loss_func)

            optimizer.zero_grad()
            train_loss.sum().backward()
            if per_sample_normalized_logits is not None:
                per_sample_normalized_logits = compute_grad_norm(model, epoch, per_sample_normalized_logits, st_idx,
                                                                 st_idx + batch_size >= len(train_data))
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                with SetBNSetRunningStats(model, new_mode=False, name="calculating train loss"):
                    train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize,
                                                               loss_func)
                with SetTrainModelMode(model, new_mode=False, name="calculating validation loss"):
                    test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
                train_loss = train_loss[~train_loss.isnan()]
                test_loss = test_loss[~test_loss.isnan()]
                print(
                    f"epoch {epoch} -  train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}, train_loss: {train_loss.mean().cpu().detach().item(): 0.3f}")
                print(
                    f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.3f}")

    if per_sample_normalized_logits is not None:
        with SetBNSetRunningStats(model, new_mode=False, name="calculating normalized logits"):
            per_sample_normalized_logits["logits"].append(
                calculate_normalized_logits(train_data, train_labels, model))
    optimizer.zero_grad()


def train_gd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model, loss_func)
        optimizer.zero_grad()
        train_loss.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if epoch % (epochs // 100 + 1) == 0:
                train_loss, train_acc = calculate_loss_acc(train_data, train_labels, model.forward_normalize, loss_func)
                test_loss, test_acc = calculate_loss_acc(test_data, test_labels, model.forward_normalize, loss_func)
                print(
                    f"epoch {epoch} - train_loss: {train_loss.mean().cpu().detach().item(): 0.4f}, train_acc: {train_acc.mean().cpu().detach().item(): 0.2f}")
                print(
                    f"epoch {epoch} - test acc: {test_acc.mean().item(): 0.2f}, test loss: {test_loss.mean().item(): 0.2f}")
    optimizer.zero_grad()


def initialize_model(model, init_type="uniform", zero_bias=True):
    if init_type == "uniform":
        model.reinitialize(mult=1, zero_bias=zero_bias)
    elif init_type == "uniform02":
        model.reinitialize(mult=0.2, zero_bias=zero_bias)
    elif init_type == "kaiming":
        initialization_fn = partial(reinitialize_kaiming, zero_bias=zero_bias, gaussian=False)
        model.apply(initialization_fn)
    elif init_type == "kaiming_gaussian":
        initialization_fn = partial(reinitialize_kaiming, zero_bias=zero_bias, gaussian=True)
        model.apply(initialization_fn)
    else:
        raise ValueError(f"init type {init_type} not recognized")
    return model


def get_batch_size_model_count(config, cur_num_samples):
    if config['optimizer.name'] == "SGD":
        cur_batch_size = min(cur_num_samples // 2, config['optimizer.batch_size'])
        cur_model_count = config['model.model_count_times_batch_size'] // cur_batch_size
    elif config['optimizer.name'] in ["GD", "guess"]:
        cur_batch_size = None
        cur_model_count = config['model.model_count_times_batch_size'] // cur_num_samples
    else:
        raise ValueError(f"optimizer name {config['optimizer.name']} not recognized")

    return cur_batch_size, cur_model_count


def load_models_from_previous_num_of_samples(config, db_path_summary, previous_num_samples, device):
    # check how many perfect models were trained and how many models needed for them
    con = sq.connect(db_path_summary)
    con.execute("BEGIN EXCLUSIVE")
    rows = con.execute(get_model_stats_short_summary_sql_script()).fetchall()
    model_cnt_dict = defaultdict(int)
    for row in rows:
        model_cnt_dict[(row[0], 0)] = row[1]
        model_cnt_dict[(row[0], 1)] = row[2]

    num_all_runs = model_cnt_dict[(previous_num_samples, 0)]
    num_successful_runs = model_cnt_dict[(previous_num_samples, 1)]
    con.close()

    # load the weights of the perfect models
    weights = []
    num_of_loaded_runs = 0
    folder_path = config['output.folder']
    for foldername in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, foldername)):
            continue
        step = foldername
        if step != 'models':
            continue
        for filename in os.listdir(os.path.join(folder_path, foldername)):
            if filename.endswith('.npy'):
                sample_num = int(filename.split('_')[-2])
                if sample_num != previous_num_samples:
                    continue
                filepath = os.path.join(folder_path, foldername, filename)
                # if step == 'models' and load_weights:
                weights.append(np.load(filepath, allow_pickle=True).item())
                num_of_loaded_runs += 1
                if num_of_loaded_runs == num_successful_runs:
                    print(
                        f"loaded {num_of_loaded_runs} runs which is the amount of successful runs for num_sample {previous_num_samples}")
                    break

    # convert the weights to the correct format
    weights_dict = {}
    for key in weights[0].keys():
        param = torch.from_numpy(np.stack([w[key] for w in weights]))
        original_shape = param.shape
        weights_dict[key] = param.reshape(-1, *original_shape[2:])

    # create new models with the same weights
    new_models = new_models_extractor(config=config, num_perfect_model=num_of_loaded_runs)
    new_models.load_state_dict(weights_dict)
    new_models.to(device)
    print(
        f"successfully loaded {num_of_loaded_runs} runs which is the amount of successful runs for num_sample {previous_num_samples}")
    return new_models, num_all_runs


def print_model_summary(model):
    bias_param = (np.array([p.numel() for n, p in model.named_parameters() if
                            p.requires_grad and 'bias' in n]) / model.model_count).astype(int)
    weight_param = (np.array([p.numel() for n, p in model.named_parameters() if
                              p.requires_grad and 'weight' in n]) / model.model_count).astype(int)
    print(f"{'=' * 20} model summary {'=' * 20}")
    print(
        f"parameters per model: {bias_param.sum() + weight_param.sum()}. per layer: {bias_param} bias, {weight_param} weights.")
    print(f"{'=' * 55} \n")


def creating_folders_for_save(config, cur_num_samples, test_all_labels):
    if config['output.save_weights']:
        os.makedirs(os.path.join(config['output.folder'], "models"), exist_ok=True)
    if config['output.save_normalized_loss_per_epoch']:
        os.makedirs(os.path.join(config['output.folder'], "normalized_logits"), exist_ok=True)
        os.makedirs(os.path.join(config['output.folder'], "weights_grad"), exist_ok=True)
    if config['output.save_predictions']:
        os.makedirs(os.path.join(config['output.folder'], "test_labels"), exist_ok=True)
        os.makedirs(os.path.join(config['output.folder'], "train_pred"), exist_ok=True)
        os.makedirs(os.path.join(config['output.folder'], "test_pred"), exist_ok=True)

        test_labels_id = secrets.token_hex(8)
        np.save(f"{config['output.folder']}/test_labels/test_labels_{cur_num_samples}_{test_labels_id}.npy",
                test_all_labels.detach().cpu().numpy())
    else:
        test_labels_id = None
    return test_labels_id


def get_models_zero_training_error(config, model_result, perfect_model_idxs, num_perfect_model, device):
    perfect_model_weights = model_result.get_weights_by_idx(perfect_model_idxs)
    perfect_model_weights_for_save = model_result.get_weights_by_idx_for_save(perfect_model_idxs)

    new_models = new_models_extractor(config=config, num_perfect_model=num_perfect_model)

    new_models.load_state_dict(perfect_model_weights)
    new_models.to(device)
    new_models.eval()

    return new_models, perfect_model_weights_for_save


def saving_results_as_np_arrays(config, aligned_idx, cur_num_samples, model_id, i, test_pred, train_pred,
                                perfect_model_weights_for_save, per_sample_normalized_logits):
    if config['output.save_predictions']:
        np.save(
            f"{config['output.folder']}/train_pred/{aligned_idx}_train_pred_{cur_num_samples}_{model_id}.npy",
            train_pred[:, aligned_idx, :])
        np.save(f"{config['output.folder']}/test_pred/{i}_test_pred_{cur_num_samples}_{model_id}.npy",
                test_pred[:, i, :])
    if config['output.save_weights']:
        # tmp = get_jth_elements(perfect_model_weights_for_save, i)
        np.save(f"{config['output.folder']}/models/model_{cur_num_samples}_{model_id}.npy",
                get_jth_elements(perfect_model_weights_for_save, i))
    if config['output.save_normalized_loss_per_epoch']:
        np.save(
            f"{config['output.folder']}/normalized_logits/normalized_logits_{cur_num_samples}_{model_id}.npy",
            per_sample_normalized_logits["logits"][aligned_idx, :, :])
        np.save(f"{config['output.folder']}/weights_grad/weights_grad_{cur_num_samples}_{model_id}.npy",
                per_sample_normalized_logits["weights"][aligned_idx, :])


def get_config_and_stats_tables():
    parser = argparse.ArgumentParser()

    config = get_current_config()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.summary(target=sys.stdout)
    config_ns = config.get()

    os.makedirs(config['output.folder'], exist_ok=True)
    db_path_summary = os.path.join(config['output.folder'], "model_stats_summary.db")
    create_short_table(db_path_summary)
    db_path = os.path.join(config['output.folder'], "model_stats.db")
    create_model_stats_table(db_path)

    return config, config_ns, db_path, db_path_summary


def find_perfect_models(config, model_count, successful_model_count, train_acc):
    perfect_model_idxs = torch.isclose(train_acc, torch.ones_like(train_acc))
    perfect_model_idxs_true = torch.where(perfect_model_idxs)[0]
    num_perfect_model = perfect_model_idxs.sum().detach().cpu().item()
    # if the number of perfect models is now more than the target model count, we will save only part of them up to the target model count (config['output.successful_model_count'])
    if num_perfect_model + successful_model_count > config['output.successful_model_count']:
        num_perfect_model_before_truncation = num_perfect_model
        num_perfect_model = config['output.successful_model_count'] - successful_model_count
        # get the indices of the perfect models
        perfect_model_idxs_true = perfect_model_idxs_true[:num_perfect_model]
        perfect_model_idxs[perfect_model_idxs_true[-1] + 1:] = False
        model_count = (model_count * num_perfect_model) // num_perfect_model_before_truncation

    return perfect_model_idxs, perfect_model_idxs_true, num_perfect_model, model_count


def train(config, db_path_summary, train_data, train_labels, test_data, test_labels, model, loss_func, optimizer,
          scheduler, batch_size, num_samples, cur_num_samples, cur_model_count, num_runs_current_num_samples, device):
    name = config['optimizer.name']
    epochs = config['optimizer.epochs']

    # for saving the norm of the gradients for figure number 7 in the paper
    per_sample_normalized_logits = None
    if config['output.save_normalized_loss_per_epoch']:
        per_sample_normalized_logits = {"logits": [], "weights": []}

    if name == "SGD":
        train_sgd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler,
                  batch_size=batch_size, epochs=epochs,
                  per_sample_normalized_logits=per_sample_normalized_logits)
    elif name == "GD":
        train_gd(train_data, train_labels, test_data, test_labels, model, loss_func, optimizer, scheduler, epochs)
    elif name == "guess":
        # if using resnet with batch norm, we need to run the model to update the running stats
        bn_layers = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)]
        if len(bn_layers) > 0:
            with torch.no_grad():
                with SetBNSetRunningStats(model, new_mode=True, name="guess train"):
                    model.forward(train_data)
        # if we don't run resnet we can use the weights that led to zero training error for the previous number of samples
        elif num_runs_current_num_samples == 1 and cur_num_samples > num_samples[0] and config[
            'output.load_models_from_previous_num_of_samples']:
            previous_num_samples = num_samples[num_samples.index(cur_num_samples) - 1]
            model, cur_model_count = load_models_from_previous_num_of_samples(config, db_path_summary,
                                                                              previous_num_samples, device)
    else:
        raise ValueError(f"optimizer name {name} not recognized")

    # for saving the normalized loss per epoch and the weights for figure number in the paper
    if per_sample_normalized_logits is not None:
        per_sample_normalized_logits["logits"] = torch.stack(per_sample_normalized_logits["logits"],
                                                             dim=0).permute(2, 0, 1).cpu().detach().numpy()
        per_sample_normalized_logits["weights"] = torch.stack(per_sample_normalized_logits["weights"],
                                                              dim=0).permute(1, 0).cpu().detach().numpy()

    return model, per_sample_normalized_logits, cur_model_count


def calculate_loss_acc_w_norm(data, labels, model, loss_func, batch_size=None, classes=None):
    if batch_size is None:
        # pred, pred_norm, norm_weights = model(data)  # pred.shape = (# of examples, # model counts , output_dim)
        pred, pred_norm = model(data)  # pred.shape = (# of examples, # model counts , output_dim)
    else:
        pred = []
        pred_norm = []
        for i in range(0, len(data), batch_size):
            pred_cur, pred_norm_cur = model(data[i:min(i + batch_size, len(data))])
            pred.append(pred_cur)  # pred_cur.shape = (# batchsize, # model counts , output_dim)
            pred_norm.append(pred_norm_cur)  # pred_norm_cur.shape = (# of batchsize, # model counts , output_dim)
        pred = torch.cat(pred, dim=0)  # pred.shape = (# of examples, # model counts , output_dim)
        pred_norm = torch.cat(pred_norm, dim=0)  # pred_norm.shape = (# of examples, # model counts , output_dim)
    n, m, o = pred.shape
    loss = loss_func(pred.view(n * m, o), labels.repeat_interleave(m)).view(n, m).mean(dim=0)
    loss_norm = loss_func(pred_norm.view(n * m, o), labels.repeat_interleave(m)).view(n, m).mean(dim=0)
    tmp = (pred.view(n * m, o).argmax(dim=1) == labels.repeat_interleave(m)).view(n, m).float()
    acc = tmp.mean(dim=0)
    acc_splited = torch.stack([tmp[labels == label].mean(dim=0) for label in classes], dim=1)
    return loss, loss_norm, acc, acc_splited, pred.detach().cpu().numpy()


def get_optimizer_and_scheduler(config, model):
    if config['optimizer.name'] in ["SGD", "GD", "guess"]:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer.lr'],
                                    momentum=config['optimizer.momentum'])
    else:
        raise ValueError(f"Optimizer {config['optimizer.name']} is not supported")

    if config['optimizer.scheduler'] is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999999], gamma=0.2)
    else:
        raise NotImplementedError("Scheduler is not implemented")
    return optimizer, scheduler


def calculate_normalized_logits(data, labels, model, batch_size=None):
    grad_input_norm, logits_subtract = calculate_norm_grad_wrt_input(data, model, batch_size)
    # convert labels from 0,1 to -1,1
    labels = labels * 2 - 1

    logits_subtract_from_correct = (labels[:, None] * logits_subtract)
    return logits_subtract_from_correct / grad_input_norm


def get_jth_elements(dictionary, j):
    return {key: values[j] for key, values in dictionary.items()}


def calculate_loss_acc(data, labels, model, loss_func, batch_size=None, weights=None, logits_normalization=None):
    """
    @param: data [b, c, h, w]
    @param: labels [b]
    @param: logits_normalization [{n,1}, {m,1}, {o,1}]

    Returns
    -------

    """
    pred = infer_model(data, model, batch_size)  # (# of examples, # model counts , output_dim)
    n, m, o = pred.shape
    if logits_normalization is not None:
        pred = pred / logits_normalization
    loss_term = loss_func(pred.view(n * m, o), labels.repeat_interleave(m)).view(n,
                                                                                 m)  # (# of examples, # model counts )

    # Average / weighted average over the models
    if weights is None:
        loss = loss_term.mean(dim=0)
    else:
        loss = (loss_term * weights[:, None]).sum(dim=0)

    acc = (pred.argmax(dim=-1) == labels[..., None]).float().mean(dim=0)
    return loss, acc


def calculate_norm_grad_wrt_input(data, model, batch_size=None):
    # Register a hook for getting the repeated data
    repeated_data_list, hook_handle = register_hook_for_getting_data_repeated(model)

    # Infer the model
    data_with_grad = data.clone().detach().requires_grad_(True)
    logits = infer_model(data_with_grad, model, batch_size)  # pred.shape = (# of examples, # model counts , output_dim)

    # Calculate the gradient of the logits with respect to the input
    logits_diff = logits[..., 1] - logits[..., 0]
    optimizer = torch.optim.SGD([data_with_grad], lr=0.1)
    optimizer.zero_grad()
    logits_diff.sum().backward()
    grad_wrt_input = repeated_data_list[0].grad
    if data.shape[-1] == 32:  # cifar10
        num_models = logits_diff.shape[1]
        grad_wrt_input = grad_wrt_input.reshape(grad_wrt_input.shape[0], num_models, 3, 32, 32)
        grad_wrt_input_norm = torch.linalg.norm(grad_wrt_input, dim=(2, 3, 4))
    else:  # mnist
        grad_wrt_input_norm = grad_wrt_input.norm(dim=(2, 3))

    # Clean up
    optimizer.zero_grad()
    hook_handle.remove()

    return grad_wrt_input_norm, logits_diff


def compute_grad_norm(model, epoch, per_sample_normalized_logits, idx, last_batch):
    num_models = model.model_count
    grads_from_layers = []
    if idx == 0:
        per_sample_normalized_logits["weights"].append([])
    for name, para in model.named_parameters():
        if 'bias' not in name:
            grads = para.grad.view(num_models, -1)
            grads_from_layers.append(grads)

    # concatenate all the grads from all the layers together and apply norm on them
    grads_from_layers = torch.cat(grads_from_layers, dim=1)
    per_sample_normalized_logits["weights"][epoch].append(torch.linalg.norm(grads_from_layers, dim=1))
    # in the last batch store the average across all the batches
    if last_batch:
        per_sample_normalized_logits["weights"][epoch] = torch.vstack(
            per_sample_normalized_logits["weights"][epoch]).mean(axis=0)
    return per_sample_normalized_logits


def calculate_max_norm_grad_wrt_input_per_model(data, models, data_batch_size=None, model_batch_size=None):
    if model_batch_size is None:
        model_batch_size = models.model_count

    max_norm_per_model = []

    for model_i in range(0, models.model_count, model_batch_size):
        subset_models = models[model_i:model_i + model_batch_size]
        # Calculate the norm of the gradients of the logit differences with respect to the input
        subset_grad_wrt_input_norm, _ = calculate_norm_grad_wrt_input(data,
                                                                      subset_models,
                                                                      batch_size=data_batch_size)
        # Calculate the maximum norm along the samples axis
        subset_max_norm_per_model = subset_grad_wrt_input_norm.max(axis=0)[0].detach()
        max_norm_per_model.append(subset_max_norm_per_model)

    max_norm_per_model = torch.cat(max_norm_per_model, dim=0)
    return max_norm_per_model


def register_hook_for_getting_data_repeated(model):
    repeated_data_list = []

    def log_repeated_data_hook(module, input, output):
        if output.requires_grad:
            output.retain_grad()
            repeated_data_list.append(output)

    hook_handle = model.repeat.register_forward_hook(log_repeated_data_hook)

    return repeated_data_list, hook_handle


def infer_model(data, model, batch_size):
    if batch_size is None:
        pred = model(data)  # pred.shape = (# of examples, # model counts , output_dim)
    else:
        pred = []
        for i in range(0, len(data), batch_size):
            pred_cur = model(data[i:min(i + batch_size, len(data))])
            pred.append(pred_cur)
        pred = torch.cat(pred, dim=0)
    return pred


class SetBNSetRunningStats:
    def __init__(self, model, new_mode, name=""):
        self.model = model
        self.new_mode = new_mode
        self.name = name

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = self.new_mode
        # print(f"[{self.name}] entering - setting BN track_running_stats to {self.new_mode}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = not self.new_mode
        # print(f"[{self.name}] exiting - setting BN track_running_stats to {not self.new_mode}")


class SetTrainModelMode:
    def __init__(self, model, new_mode, name=""):
        self.model = model
        self.new_mode = new_mode
        self.name = name

    def __enter__(self):
        self.model.train(self.new_mode)
        # print(f"[{self.name}] entering - setting model to {'train' if self.model.training else 'eval'}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(not self.new_mode)
        # print(f"[{self.name}] exiting - setting model to {'train' if self.model.training else 'eval'}")


@torch.no_grad()
def reinitialize_kaiming(m, zero_bias=True, gaussian=False):
    if isinstance(m, nn.Conv2d):
        if gaussian:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

        if m.bias is not None:
            if zero_bias:
                torch.nn.init.zeros_(m.bias)
            else:
                val = m.weight.max().item()
                torch.nn.init.uniform_(m.bias, a=-val, b=val)
