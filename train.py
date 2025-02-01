import json
import time

import torch
from torch import nn

from utils.datasets_utils import get_dataset
from utils.model_utils import get_model
from utils.sql import *
from utils.train_utils import initialize_model, get_batch_size_model_count, print_model_summary, \
    creating_folders_for_save, get_models_zero_training_error, saving_results_as_np_arrays, get_config_and_stats_tables, \
    find_perfect_models, train, calculate_loss_acc_w_norm, get_optimizer_and_scheduler, SetBNSetRunningStats, \
    SetTrainModelMode

if __name__ == "__main__":
    # get stats tables and configs
    config, config_ns, db_path, db_path_summary = get_config_and_stats_tables()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    loss_func = nn.CrossEntropyLoss(reduction='none')
    num_samples_list = [int(v) for v in config['distributed.num_samples'].split(",")]

    # loop over the number of samples
    for num_samples in num_samples_list:

        # num_iter_per_sample helps us understand if we need to load the training data again (computational reasons)
        # num_runs_current_num_samples, which we will define later, counts the number of runs across all gpus and helps
        # us understand if we need to load the weights of GNC from the previous number of samples
        num_iter_per_sample = 0

        # run until reaching the target model count for the current number of samples
        while True:
            num_iter_per_sample += 1

            # get the next configuration
            batch_size, model_count = get_batch_size_model_count(config, num_samples)
            next_config = get_next_config(db_path_summary=db_path_summary, num_sample=num_samples, config=config)
            run_id, classes_seed, permutation_seed, training_seed, successful_model_count, all_model_count, num_runs_current_num_samples = next_config

            # check if we reached the target model count for the current number of samples
            if successful_model_count >= config['output.successful_model_count']:
                print(
                    f"Tested more models than the tested model count:{config['output.successful_model_count']}, so ending the search")
                break

            # print the current configuration
            get_model_stats_short_summary(db_path_summary)
            print(f"training_seed:{training_seed} , classes_seed:{classes_seed}, permutation_seed:{permutation_seed}")
            print("next config:",
                  json.dumps(
                      {"num_samples": num_samples,
                       "num_runs_for_current_num_samples": num_runs_current_num_samples,
                       "already_tested_num_models": all_model_count,
                       "successful_num_models": successful_model_count,
                       "current_num_models": model_count}))

            # get the dataset
            if (num_iter_per_sample == 1) | (config['distributed.data_seed'] is None):
                train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels, classes = get_dataset(
                    config=config,
                    num_samples=num_samples,
                    classes_seed=classes_seed,
                    permutation_seed=permutation_seed)

            torch.manual_seed(training_seed)
            # initialize the model and the optimizer

            model = get_model(config=config, model_count=model_count, device=device)

            if num_iter_per_sample == 1:
                print_model_summary(model[:1])

            model = initialize_model(model, init_type=config['model.init'], zero_bias=config['model.zero_bias'])
            optimizer, scheduler = get_optimizer_and_scheduler(config, model)

            start_time = time.time()

            # train
            with SetTrainModelMode(model, new_mode=True, name="training"):
                model_result, per_sample_normalized_logits, model_count = train(config, db_path_summary, train_data,
                                                                                train_labels, test_data, test_labels,
                                                                                model, loss_func, optimizer, scheduler,
                                                                                batch_size, num_samples_list,
                                                                                num_samples, model_count,
                                                                                num_runs_current_num_samples, device)

            train_time = time.time() - start_time

            print("==============================================")
            print("finished training")
            print("==============================================")
            with torch.no_grad():
                with SetTrainModelMode(model, new_mode=True, name="training"):
                    with SetBNSetRunningStats(model, new_mode=False, name="checking 100 percent train acc"):
                        train_loss, train_loss_norm, train_acc, train_acc_split, train_pred = calculate_loss_acc_w_norm(
                            train_data, train_labels, model_result.forward_normalize_mult, loss_func,
                            batch_size=batch_size,
                            classes=classes)

                # find the number of zero training error models and update the database
                perfect_model_idxs, perfect_model_idxs_true, num_perfect_model, model_count = find_perfect_models(
                    config, model_count, successful_model_count, train_acc)

                update_short_table(db_path=db_path_summary,
                                   run_id=run_id,
                                   data_seed=classes_seed,
                                   permutation_seed=permutation_seed,
                                   training_seed=training_seed,
                                   num_training_samples=num_samples,
                                   perfect_model_count=num_perfect_model,
                                   tested_model_count=model_count,
                                   status="COMPLETE")

                print(f"Found {num_perfect_model} good models for num of samples {num_samples}")

                # since we are testing only zero training error models, we can skip the testing
                if num_perfect_model == 0:
                    continue

                new_models, perfect_model_weights_for_save = get_models_zero_training_error(config, model_result,
                                                                                            perfect_model_idxs,
                                                                                            num_perfect_model, device)

                # calculate the test loss and accuracy
                test_loss, test_loss_norm, test_acc, test_acc_split, test_pred = calculate_loss_acc_w_norm(
                    test_all_data.to(device), test_all_labels.to(device), new_models.forward_normalize_mult, loss_func,
                    batch_size=4, classes=classes)

            print("=" * 50)

            print("==============================================")
            print("finished testing, update the database")
            print("==============================================")
            test_labels_id = creating_folders_for_save(config, num_samples, test_all_labels)

            num_times_0_predicted = np.sum(test_pred[:, :, 0] >= test_pred[:, :, 1], axis=0)
            num_times_1_predicted = test_pred.shape[0] - num_times_0_predicted

            for i in range(num_perfect_model):
                model_id = secrets.token_hex(8)
                aligned_idx = perfect_model_idxs_true[i].item()

                saving_results_as_np_arrays(config, aligned_idx, num_samples, model_id, i, test_pred, train_pred,
                                            perfect_model_weights_for_save, per_sample_normalized_logits)

                update_model_stats_table(db_path=db_path,
                                         model_id=model_id,
                                         data_seed=classes_seed,
                                         training_seed=training_seed,
                                         num_training_samples=num_samples,
                                         train_loss=train_loss[aligned_idx].item(),
                                         weight_norm=-1,
                                         train_loss_normalize=train_loss_norm[aligned_idx].item(),
                                         train_acc=train_acc[aligned_idx].item(),
                                         test_loss=test_loss[i].item(),
                                         test_acc=test_acc[i].item(),
                                         test_false_positive=test_acc_split[i][0].item(),
                                         test_false_negative=test_acc_split[i][1].item(),
                                         num_of_models=1,
                                         num_times_0_predicted=num_times_0_predicted[i],
                                         num_times_1_predicted=num_times_1_predicted[i],
                                         save_path=test_labels_id,
                                         status="COMPLETE")

            get_model_stats_summary(db_path, verbose=True)

            print("==============================================")
