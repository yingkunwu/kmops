import os
import torch
import tqdm
import argparse
from omegaconf import OmegaConf
from contextlib import redirect_stdout

from kmops import build_model
from dataset import build_dataset
from evaluators import build_validator
from utils.vis import MEAN, STD, normalize_batch, visualize_with_gt

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
        preds = KMOPS.posefitting(k3d, conf)

        gt_k3d = targets['kpts_3d'][i]
        gts = KMOPS.posefitting(gt_k3d)

        results.append({
            'pred_kpts_l': k2d_l,
            'pred_kpts_r': k2d_r,
            'pred_conf': conf,
            'pred_scores': scores,
            'pred_RTs': preds['poses'],
            'pred_scales': preds['scales'],
            "pred_class_ids": labels.long(),
            'pred_box3ds': preds['box3ds'],
            'pred_ax3ds': preds['ax3ds'],  # for visualization
            'gt_RTs': gts['poses'],
            "gt_scales": gts['scales'],
            "gt_class_ids": targets['labels'][i],
            'gt_box3ds': gts['box3ds'],
            'gt_ax3ds': gts['ax3ds']  # for visualization
        })

    return results


def run(args):
    ckpt_path = os.path.join(args.wandb_folder, "weight", "last.ckpt")
    cfg_path = os.path.join(args.wandb_folder, "config.yaml")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vis_dir = os.path.join(args.wandb_folder, "vis")
    os.makedirs(vis_dir, exist_ok=True)

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

    for bi, (bs, bt) in tqdm.tqdm(enumerate(loader), total=len(loader)):
        outputs = inference(KMOPS, bs, bt, device)

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

        if bi % 100 == 0:
            batch_img_l, batch_img_r, _ = bs.decompose()
            imgs_l = normalize_batch(batch_img_l, MEAN, STD)
            imgs_r = normalize_batch(batch_img_r, MEAN, STD)
            # visualize each sample in the batch separately
            for idx, res in enumerate(outputs):
                visualize_with_gt(
                    imgs_l[idx], imgs_r[idx],
                    res['pred_box3ds'],
                    res['pred_scores'],
                    res['pred_class_ids'],
                    res['pred_ax3ds'],
                    res['gt_box3ds'],
                    res['gt_class_ids'],
                    res['gt_ax3ds'],
                    cfg.dataset.names,
                    bt['proj_matrix_l'][idx].cpu().to(torch.float32),
                    bt['proj_matrix_r'][idx].cpu().to(torch.float32),
                    os.path.join(vis_dir, f"batch_{bi}_{idx}.png")
                )

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
