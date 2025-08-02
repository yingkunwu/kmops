import os
import torch
import tqdm
import argparse
from omegaconf import OmegaConf
from contextlib import redirect_stdout

from kmops import build_model, build_postprocessor
from dataset import build_dataset
from evaluators.inference import Inferencer
from evaluators import build_validator

torch.set_float32_matmul_precision("high")


def run(args):
    ckpt_path = os.path.join(args.wandb_folder, "weight", "best.ckpt")
    cfg_path = os.path.join(args.wandb_folder, "config.yaml")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.join(args.wandb_folder, "vis")
    os.makedirs(out_dir, exist_ok=True)

    cfg = OmegaConf.load(cfg_path)
    print(cfg)

    # setup
    model = build_model(cfg)
    postprocessor = build_postprocessor(cfg)
    loader = build_dataset(cfg, 'val')
    validator = build_validator(cfg, task="pose6d")

    # weights & output dir
    ckpt = torch.load(ckpt_path, weights_only=False)['state_dict']
    model.load_state_dict(
        {k.replace('model.', ''): v for k, v in ckpt.items()}, strict=True)

    infer = Inferencer(model, postprocessor, device)
    results = []

    for batch_samples, batch_targets in tqdm.tqdm(loader, total=len(loader)):
        outputs = infer.estimate(batch_samples, batch_targets)

        # Convert all tensors in outputs to numpy
        for output in outputs:
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    output[key] = value.cpu().numpy()
                elif isinstance(value, list):
                    output[key] = [v.cpu().numpy()
                                   if isinstance(v, torch.Tensor)
                                   else v for v in value]
        for res in outputs:
            validator.add_result(res)
        results.extend(outputs)

    result_log_path = os.path.join(args.wandb_folder, 'eval_result.log')
    validator.compute_metrics()
    with open(result_log_path, 'w') as log_file:
        with redirect_stdout(log_file):
            validator.get_result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_folder",
        type=str,
        required=True,
        help="path to wandb run folder"
    )
    args = parser.parse_args()
    run(args)
