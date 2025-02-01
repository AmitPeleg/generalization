import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import torch

from settings import ROOT_DIR


def run_train_script(task_id, args, stdout_path=None, stderr_path=None):
    stdout = open(stdout_path, 'w') if stdout_path is not None else subprocess.DEVNULL  # Discard stdout
    stderr = open(stderr_path, 'w') if stderr_path is not None else subprocess.PIPE

    gpu_id = GPU_QUEUE.get()

    try:
        # Run the subprocess, capturing output and errors
        print(f"Start running {task_id} on gpu {gpu_id}, args: {args}")
        result = subprocess.run(args=['python', str(ROOT_DIR / 'train.py'), *args],
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)},
                                stdout=stdout,
                                stderr=stderr
                                )
    finally:
        # Close the file handles if they were opened
        if stdout not in [subprocess.PIPE, subprocess.DEVNULL]:
            stdout.close()
        if stderr not in [subprocess.PIPE, subprocess.DEVNULL]:
            stderr.close()

    GPU_QUEUE.put(gpu_id)
    print(f"Finish running {task_id} on gpu {gpu_id}")

    if result.returncode != 0 and stderr_path is None:
        print(f"Process {task_id} with {args} failed.\n"
              f"Error: {result.stderr.decode('utf-8')}")


def run_train_from_config_dir(config_dir, stdout_dir=None, stderr_dir=None, max_parallel=None):
    if max_parallel is None:
        max_parallel = GPU_COUNT

    if stdout_dir is not None:
        stdout_dir.mkdir(parents=True, exist_ok=True)
    if stderr_dir is not None:
        stderr_dir.mkdir(parents=True, exist_ok=True)

    # Using ProcessPoolExecutor to limit the number of concurrent processes
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all the tasks
        futures = []
        for task_id, cfg in enumerate(config_dir.glob('*.yaml')):
            args = ['--config', str(cfg)]
            stdout_path = stdout_dir / f'{task_id}_{cfg.stem}_out.log' if stdout_dir is not None else None
            stderr_path = stderr_dir / f'{task_id}_{cfg.stem}_err.log' if stderr_dir is not None else None
            futures.append(executor.submit(run_train_script,
                                           task_id=task_id,
                                           args=args,
                                           stdout_path=stdout_path,
                                           stderr_path=stderr_path))

        # Wait for all tasks to complete
        for future in futures:
            _ = future.result()


parser = argparse.ArgumentParser()
parser.add_argument('fig_name', type=str)

if __name__ == '__main__':
    GPU_LIST = list(range(torch.cuda.device_count()))
    GPU_COUNT = len(GPU_LIST)

    manager = Manager()
    GPU_QUEUE = manager.Queue()

    for i in GPU_LIST:
        GPU_QUEUE.put(i)

    fig_name = parser.parse_args().fig_name
    config_fig_dir = ROOT_DIR / 'configs' / fig_name
    if not config_fig_dir.exists():
        raise ValueError(f'Config directory {config_fig_dir} does not exist')

    stdout_dir = ROOT_DIR / 'logs' / f'{fig_name}'
    # create the log directory if it does not exist
    os.makedirs(ROOT_DIR / 'logs', exist_ok=True)
    run_train_from_config_dir(config_fig_dir, stdout_dir=stdout_dir, stderr_dir=stdout_dir)
