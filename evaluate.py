import os
import torch
import tqdm
import argparse
from omegaconf import OmegaConf
from contextlib import redirect_stdout

from kmops import build_model
from dataset import build_dataset
from evaluators import build_validator

torch.set_float32_matmul_precision("high")


def inference(KMOPS, samples, targets, device):
    with torch.no_grad():
        out = KMOPS(samples.to(device))
    out = KMOPS.postprocess(out, targets["ori_shape"])

    results = []  # will hold per-sample dicts
    for i in range(len(targets['kpts_3d'])):

        scores = out['score'][i]
        labels = out['label'][i]
        k2d_l = out['kpts_l'][i]
        k2d_r = out['kpts_r'][i]
        conf = out['conf'][i]

        # decode & project bboxes and axes
        PL = targets['proj_matrix_l'][i].to(torch.float32)
        PR = targets['proj_matrix_r'][i].to(torch.float32)
        baseline = targets['baseline'][i].to(torch.float32)

        # get predicted pose
        k3d = KMOPS.reprojection(PL, PR, baseline, k2d_l, k2d_r)
        preds = KMOPS.posefitting(PL, PR, k3d, conf)

        gt_k3d = targets['kpts_3d'][i]
        gts = KMOPS.posefitting(PL, PR, gt_k3d)

        results.append({
            'pred_kpts_l': k2d_l,
            'pred_kpts_r': k2d_r,
            'pred_conf': conf,
            'pred_scores': scores,
            'pred_boxes_l': preds['pt_l'],
            'pred_boxes_r': preds['pt_r'],
            'pred_RTs': preds['pose'],
            'pred_scales': preds['scale'],
            "pred_class_ids": labels.long(),
            'gt_boxes_l': gts['pt_l'],
            'gt_boxes_r': gts['pt_r'],
            'gt_RTs': gts['pose'],
            "gt_scales": gts['scale'],
            "gt_class_ids": targets['labels'][i],
        })

    return results


def run(args):
    ckpt_path = os.path.join(args.wandb_folder, "weight", "last.ckpt")
    cfg_path = os.path.join(args.wandb_folder, "config.yaml")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.join(args.wandb_folder, "vis")
    os.makedirs(out_dir, exist_ok=True)

    cfg = OmegaConf.load(cfg_path)
    print(cfg)

    # setup
    KMOPS = build_model(cfg)
    loader = build_dataset(cfg, 'val')
    validator = build_validator(cfg, task="pose6d")

    # weights & output dir
    ckpt = torch.load(ckpt_path, weights_only=False)['state_dict']
    KMOPS.load_state_dict(
        {k.replace('model.', ''): v for k, v in ckpt.items()}, strict=True)
    KMOPS = KMOPS.eval().to(device)

    for batch_samples, batch_targets in tqdm.tqdm(loader, total=len(loader)):
        outputs = inference(KMOPS, batch_samples, batch_targets, device)

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
