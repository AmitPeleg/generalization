import numpy as np
import torch

from settings import ROOT_DIR
from utils.model_utils import new_models_extractor


def load_models(config, cur_num_samples):
    # Get the output directory
    out_dir = ROOT_DIR / config['output.folder']

    models_dir = out_dir / "models"

    # collect the weights of the models and the model ids into lists
    weights = []
    model_ids = []
    for npy_file in models_dir.glob("*.npy"):
        sample_num = int(npy_file.stem.split('_')[-2])
        if sample_num == cur_num_samples:
            weights.append(np.load(npy_file, allow_pickle=True).item())
            model_ids.append(npy_file.stem.split('_')[-1][:16])

    # convert the weights to the correct format
    weights_dict = {}
    for key in weights[0].keys():
        param = torch.from_numpy(np.stack([w[key] for w in weights]))
        original_shape = param.shape
        weights_dict[key] = param.reshape(-1, *original_shape[2:])

    # create new models with the same weights
    new_models = new_models_extractor(config=config, num_perfect_model=len(weights))
    new_models.load_state_dict(weights_dict)
    new_models.cuda()
    print(f"Successfully loaded {len(weights)}")
    return new_models, model_ids
