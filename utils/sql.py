import secrets
import sqlite3 as sq
from collections import defaultdict

import numpy as np
from tabulate import tabulate


def update_model_stats_table_sql_script(model_id, data_seed, training_seed, num_training_samples, train_loss,
                                        weight_norm, train_loss_normalize, train_acc, test_loss,
                                        test_acc, test_false_positive,
                                        test_false_negative, num_of_models, num_times_0_predicted,
                                        num_times_1_predicted, save_path, status):
    return f"""
    REPLACE INTO model_stats VALUES ( '{model_id}', {data_seed}, {training_seed}, {num_training_samples}, {train_loss}, {weight_norm}, {train_loss_normalize}, {train_acc}, {test_loss}, {test_acc}, {test_false_positive}, {test_false_negative}, '{num_of_models}', '{num_times_0_predicted}', '{num_times_1_predicted}' , '{save_path}', '{status}')"""


def update_short_table_sql_script(
        run_id, data_seed, permutation_seed, training_seed, num_training_samples, perfect_model_count,
        tested_model_count, status):
    return f"""
        REPLACE INTO model_stats_summary VALUES ( '{run_id}', {data_seed}, {permutation_seed}, {training_seed}, {num_training_samples}, {perfect_model_count}, {tested_model_count}, '{status}' )"""


def create_model_stats_table_sql_script():
    return f"""CREATE TABLE IF NOT EXISTS model_stats (
	model_id TEXT PRIMARY KEY,
    data_seed INTEGER,
    training_seed INTEGER,
    num_training_samples INTEGER,
    train_loss REAL,
    weight_norm REAL,
    train_loss_normalize REAL,
    train_acc REAL,
    test_loss REAL,
    test_acc REAL,
    test_false_positive INTEGER,
    test_false_negative INTEGER,
    num_of_models INTEGER,
    num_times_0_predicted INTEGER,
    num_times_1_predicted INTEGER,
    save_path TEXT,
    status TEXT
    );"""


def create_short_table_sql_script():
    return f"""CREATE TABLE IF NOT EXISTS model_stats_summary (
	run_id TEXT PRIMARY KEY,
    data_seed INTEGER,
    permutation_seed INTEGER,
    training_seed INTEGER,
    num_training_samples INTEGER,
    perfect_model_count INTEGER,
    tested_model_count INTEGER,
    status TEXT);"""


def create_short_table(db_path):
    con = sq.connect(db_path, isolation_level="EXCLUSIVE")
    con.execute(create_short_table_sql_script())
    con.commit()
    con.close()


def create_model_stats_table(db_path):
    con = sq.connect(db_path, isolation_level="EXCLUSIVE")
    con.execute(create_model_stats_table_sql_script())
    con.commit()
    con.close()


def update_short_table(db_path, run_id, data_seed, permutation_seed, training_seed, num_training_samples,
                       perfect_model_count,
                       tested_model_count, status):
    con = sq.connect(db_path)
    con.execute("BEGIN EXCLUSIVE")
    cur = con.cursor()
    cur.execute(
        update_short_table_sql_script(
            run_id, data_seed, permutation_seed, training_seed, num_training_samples, perfect_model_count,
            tested_model_count, status)
    )
    # update model_stats table
    con.commit()
    con.close()


def update_model_stats_table(db_path, model_id, data_seed, training_seed, num_training_samples, train_loss,
                             weight_norm, train_loss_normalize, train_acc, test_loss,
                             test_acc, test_false_positive, test_false_negative, num_of_models,
                             num_times_0_predicted, num_times_1_predicted, save_path, status):
    if train_loss == np.inf or weight_norm == np.inf or test_loss == np.inf:
        print(f"{train_loss=}, {weight_norm=}, {test_loss=} in {model_id=}")
        train_loss = -1  # if train_loss==np.inf else train_loss
        weight_norm = -1  # if weight_norm==np.inf else weight_norm
        test_loss = -1  # if (test_loss==np.inf or test_loss==np.nan) else test_loss

    con = sq.connect(db_path)
    cur = con.cursor()
    cur.execute(
        update_model_stats_table_sql_script(model_id, data_seed, training_seed, num_training_samples,
                                            train_loss, weight_norm, train_loss_normalize, train_acc,
                                            test_loss, test_acc, test_false_positive,
                                            test_false_negative, num_of_models,
                                            num_times_0_predicted, num_times_1_predicted,
                                            save_path, status)
    )
    con.commit()
    con.close()


def get_model_stats_summary_sql_script():
    return """
    SELECT
        model_stats.num_training_samples,
        AVG(model_stats.train_loss) AS avg_train_loss,
        AVG(model_stats.train_loss_normalize) AS avg_train_loss_normalize,
        AVG(model_stats.train_acc) AS avg_train_acc,
        AVG(model_stats.test_loss) AS avg_test_loss,
        AVG(model_stats.test_acc) AS avg_test_acc,
        SUM(model_stats.num_of_models) AS total_models
    FROM
        model_stats,
        (SELECT num_training_samples, AVG(test_acc) AS avg_test_acc FROM model_stats WHERE status = 'COMPLETE' GROUP BY num_training_samples) AS subquery
    WHERE 
        model_stats.status = 'COMPLETE' AND model_stats.num_training_samples = subquery.num_training_samples
    GROUP BY
        model_stats.num_training_samples
    ;"""


def get_model_stats_full_with_id_sql_script_with_num_prediction():
    return """
    SELECT
        num_training_samples,
        train_loss,
        train_loss_normalize,
        train_acc,
        test_loss,
        test_acc,
        weight_norm,
        model_id,
        num_times_0_predicted,
        num_times_1_predicted
    FROM
        model_stats
    WHERE 
        status = 'COMPLETE'
    ;"""


def get_model_stats_full_with_id_sql_script():
    return """
    SELECT
        num_training_samples,
        train_loss,
        train_loss_normalize,
        train_acc,
        test_loss,
        test_acc,
        weight_norm,
        model_id
    FROM
        model_stats
    WHERE 
        status = 'COMPLETE'
    ;"""


def get_model_stats_short_summary_sql_script():
    return """
    SELECT
        num_training_samples,
        SUM(tested_model_count),
        SUM(perfect_model_count)
    FROM
        model_stats_summary
    WHERE 
        status = 'COMPLETE'
    GROUP BY
        num_training_samples
    ;"""


def get_model_stats_full_with_id(db_path, add_num_of_test_predictions=False):
    con = sq.connect(db_path)
    if add_num_of_test_predictions:
        rows = con.execute(get_model_stats_full_with_id_sql_script_with_num_prediction()).fetchall()
    else:
        rows = con.execute(get_model_stats_full_with_id_sql_script()).fetchall()
    con.close()
    return rows


def get_model_stats_summary(db_path, verbose=True):
    con = sq.connect(db_path)
    rows = con.execute(
        get_model_stats_summary_sql_script()
    ).fetchall()
    if verbose:
        print(tabulate(rows, headers=['num_training_samples', "AVG(train_loss)", "AVG(train_loss_normalize)",
                                      "AVG(train_acc)", "AVG(test_loss)",
                                      "AVG(test_acc)", "SUM(num_of_models)"], tablefmt='psql'))
    con.close()
    return rows


def get_model_stats_short_summary(db_path, verbose=True):
    con = sq.connect(db_path)
    rows = con.execute(
        get_model_stats_short_summary_sql_script()
    ).fetchall()
    if verbose:
        print(tabulate(rows, headers=['num_training_samples', 'SUM(tested_model_count)', 'SUM(perfect_model_count)'],
                       tablefmt='psql'))
    con.close()
    return rows


def get_model_stats_acc_sql_script():
    return """
    SELECT
        num_training_samples,
        test_acc
    FROM
        model_stats
    WHERE
        status = 'COMPLETE'
    ;"""


def get_model_stats_acc(db_path):
    con = sq.connect(db_path)
    rows = con.execute(
        get_model_stats_acc_sql_script()
    ).fetchall()
    con.close()
    return rows


def get_next_config(db_path_summary, num_sample, config):
    # this function finds the data_seed, permutation_seed, training_seed, successful_model_count, all_model_count, num_runs_per_sample (on all servers, so we can later load the weights for gnc only in the first run)

    # getting the definition of the seeds from the config
    training_seed = config['distributed.training_seed']
    data_seed = config['distributed.data_seed']
    permutation_seed = config['distributed.permutation_seed']

    con = sq.connect(db_path_summary)
    con.execute("BEGIN EXCLUSIVE")
    rows = con.execute(get_model_stats_short_summary_sql_script()).fetchall()
    model_cnt_dict = defaultdict(int)
    for row in rows:
        model_cnt_dict[(row[0], 0)] = row[1]
        model_cnt_dict[(row[0], 1)] = row[2]

    # get the number of successful models and all models (also the ones that do not have zero training error) for the current num_sample
    all_model_count = model_cnt_dict[(num_sample, 0)]
    successful_model_count = model_cnt_dict[(num_sample, 1)]

    rows = con.execute("""
    SELECT
        MAX(data_seed),
        MAX(training_seed)
    FROM
        model_stats_summary
    ;""")
    data_seed_next, training_seed_next = rows.fetchone()

    # in this part if we set the seeds in the config, we will use them, otherwise we will increment them by 1, starting from 200
    if data_seed is not None:
        data_seed_next = data_seed
    else:
        data_seed_next = 200 if data_seed_next is None else data_seed_next + 1
    if training_seed is not None:
        training_seed_next = training_seed
    else:
        training_seed_next = 200 if training_seed_next is None else training_seed_next + 1
    permutation_seed_next = data_seed_next if permutation_seed is None else permutation_seed

    # to enter into the model_stats table we need to have a fake result, so we can update the table also for pending runs
    fake_res = -1
    run_id = secrets.token_hex(8)

    con.execute(
        update_short_table_sql_script(
            run_id=secrets.token_hex(8),
            data_seed=data_seed_next,
            permutation_seed=permutation_seed_next,
            training_seed=training_seed_next,
            num_training_samples=num_sample,
            perfect_model_count=fake_res,
            tested_model_count=fake_res,
            status="PENDING")
    )

    # To check how many runs do we have for the current num_sample including ones that do not finish
    query = f"SELECT COUNT(*) FROM model_stats_summary WHERE num_training_samples = {num_sample} AND status = 'PENDING'"
    ans = con.execute(query)

    # Fetch the result of the query
    num_runs_per_sample = ans.fetchone()[0]
    con.commit()
    con.close()

    return run_id, data_seed_next, permutation_seed_next, training_seed_next, successful_model_count, all_model_count, num_runs_per_sample
