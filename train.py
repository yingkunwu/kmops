import os
import math
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from kmops import build_model, build_criterion
from dataset import build_dataset
from evaluators import build_validator
from utils.util import kpts_to_boxes
from utils.vis import save_attention_loc, visualize_with_gt

# Faster, but less precise
torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)


class DETRModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.get("batch_size")
        self.lr = cfg.get("learning_rate")
        self.backbone_lr = cfg.get("backbone_learning_rate")
        self.weight_decay = cfg.get("weight_decay")
        if cfg.get("scheduler") is not None:
            s = cfg.get("scheduler")
            self.lr_step = s["milestones"]
            self.lr_factor = s["gamma"]

        self.model = build_model(cfg)
        self.criterion = build_criterion(cfg)
        # for 2D keypoint evaluation
        # use it to check if the 2D keypoint estimation on each view is good
        # To get accurate 6D pose estimation, the keypoint estimation has to
        # be accurate.
        self.validator_l = build_validator(cfg, "pose")
        self.validator_r = build_validator(cfg, "pose")
        # for 3D box and pose evaluation (used in the paper)
        self.validator_pose = build_validator(cfg, "pose6d")
        self.validator_l.init_metrics()
        self.validator_r.init_metrics()

    def configure_optimizers(self):
        def is_backbone(n): return 'backbone' in n

        params = list(self.model.named_parameters())
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in params if not is_backbone(n)]},
            {'params': [p for n, p in params if is_backbone(n)],
             'lr': self.backbone_lr},
        ], lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, batch, batch_idx):
        samples, targets = batch

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)

        loss_value = loss_dict["total_loss"].detach()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            raise Exception("Loss is not finite")

        return loss_dict, outputs

    def training_step(self, batch, batch_idx):
        loss_dict, outputs = self.forward(batch, batch_idx)

        log = {}
        for key, value in loss_dict.items():
            log[f"train/{key}"] = value.detach()

        self.log_dict(
            log,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size)

        if batch_idx == 0:
            _, targets = batch
            outputs = self.model.postprocess(outputs, targets["ori_shape"])
            figure = save_attention_loc(
                batch[0].image_l, batch[0].image_r, outputs,
                num_display=min(4, self.batch_size), num_objs=5
            )
            # grab the W&B Run
            self.logger.log_image(
                key="plot/train_attention_map",
                images=[figure],
                caption=[f"epoch_{self.current_epoch}_batch_{batch_idx}"]
            )

        return {"loss": loss_dict["total_loss"]}

    def validation_step(self, batch, batch_idx):
        loss_dict, outputs = self.forward(batch, batch_idx)

        log = {}
        for key, value in loss_dict.items():
            log[f"val/{key}"] = value.detach()

        self.log_dict(
            log,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size)

        _, targets = batch
        outputs = self.model.postprocess(
            outputs, targets["ori_shape"], targets)

        results = []
        preds_l, preds_r, pred_kpts_l, pred_kpts_r = [], [], [], []
        targs_l, targs_r, tart_kpts_l, tart_kpts_r = [], [], [], []
        for i in range(len(targets['kpts_3d'])):
            scores = outputs['score'][i]
            labels = outputs['label'][i]
            kpts_l = outputs['kpts_l'][i]
            kpts_r = outputs['kpts_r'][i]
            gt_kpts_l = outputs['gt_kpts_l'][i]
            gt_kpts_r = outputs['gt_kpts_r'][i]
            gt_labels = outputs['gt_label'][i]
            conf = outputs['conf'][i]

            PL = targets['proj_matrix_l'][i].to(torch.float32).cpu()
            PR = targets['proj_matrix_r'][i].to(torch.float32).cpu()
            baseline = targets['baseline'][i].to(torch.float32).cpu()

            # get predicted pose
            k3d = self.model.reprojection(PL, PR, baseline, kpts_l, kpts_r)
            preds = self.model.posefitting(k3d, conf)

            gt_k3d = targets['kpts_3d'][i].cpu()
            gts = self.model.posefitting(gt_k3d)

            # 6D pose evaluation
            results.append({
                "gt_class_ids": gt_labels.numpy(),
                "gt_scales": gts['scales'],
                "gt_RTs": gts['poses'],
                'gt_box3ds': gts['box3ds'],
                'gt_ax3ds': gts['ax3ds'],  # for visualization
                "pred_class_ids": labels.numpy(),
                "pred_scales": preds['scales'],
                "pred_RTs": preds['poses'],
                "pred_scores": scores.numpy(),
                'pred_box3ds': preds['box3ds'],
                'pred_ax3ds': preds['ax3ds'],  # for visualization
            })

            # 2D box and keypoint evaluation
            box_l = kpts_to_boxes(kpts_l)
            box_r = kpts_to_boxes(kpts_r)
            gt_box_l = kpts_to_boxes(gt_kpts_l)
            gt_box_r = kpts_to_boxes(gt_kpts_r)

            gt_kpts_vis = targets["kpts"][i][..., 2:].cpu()
            gt_kpts_l = torch.cat([gt_kpts_l, gt_kpts_vis], dim=-1)
            gt_kpts_r = torch.cat([gt_kpts_r, gt_kpts_vis], dim=-1)

            preds_l.append(torch.cat(
                [box_l, scores[:, None], labels[:, None].float()], dim=-1))
            preds_r.append(torch.cat(
                [box_r, scores[:, None], labels[:, None].float()], dim=-1))
            targs_l.append(torch.cat(
                [gt_box_l, gt_labels[:, None].float()], dim=-1))
            targs_r.append(torch.cat(
                [gt_box_r, gt_labels[:, None].float()], dim=-1))

            pred_kpts_l.append(kpts_l)
            pred_kpts_r.append(kpts_r)
            tart_kpts_l.append(gt_kpts_l)
            tart_kpts_r.append(gt_kpts_r)

        for res in results:
            self.validator_pose.add_result(res)
        self.validator_l.update_metrics(
            preds_l, targs_l, pred_kpts_l, tart_kpts_l)
        self.validator_r.update_metrics(
            preds_r, targs_r, pred_kpts_r, tart_kpts_r)

        if batch_idx == 0:
            figure = save_attention_loc(
                batch[0].image_l, batch[0].image_r, outputs,
                num_display=min(4, self.batch_size), num_objs=5
            )
            self.logger.log_image(
                key="plot/val_attention_map",
                images=[figure],
                caption=[f"epoch_{self.current_epoch}_batch_{batch_idx}"]
            )

            images, captions = [], []
            for idx, res in enumerate(results):
                img_l, img_r, _ = batch[0].decompose()
                image = visualize_with_gt(
                    img_l[idx], img_r[idx],
                    res['pred_box3ds'],
                    res['pred_scores'],
                    res['pred_class_ids'],
                    res['pred_ax3ds'],
                    res['gt_box3ds'],
                    res['gt_class_ids'],
                    res['gt_ax3ds'],
                    self.cfg.dataset.names,
                    batch[1]['proj_matrix_l'][idx].cpu().to(torch.float32),
                    batch[1]['proj_matrix_r'][idx].cpu().to(torch.float32),
                    normalize_image=True
                )
                images.append(image[..., ::-1])  # convert BGR to RGB
                captions.append(f"epoch_{self.current_epoch}_idx_{idx}")
                if idx >= 3:
                    break

            self.logger.log_image(
                key="plot/val_prediction",
                images=images,
                caption=captions
            )

    def on_validation_epoch_end(self):
        self.validator_pose.compute_metrics()
        res = self.validator_pose.get_result()
        self.log_dict(
            {f"metric_pose/{k}": v for k, v in res.items()},
            on_epoch=True
        )
        self.validator_pose.init_metrics()
        res_l = self.validator_l.get_result()
        self.log_dict(
            {f"metric_left/{k}": v for k, v in res_l.items()},
            on_epoch=True
        )
        self.validator_l.init_metrics()
        res_r = self.validator_r.get_result()
        self.log_dict(
            {f"metric_right/{k}": v for k, v in res_r.items()},
            on_epoch=True
        )
        self.validator_r.init_metrics()


@hydra.main(config_path="conf", config_name="", version_base="1.3")
def run(cfg: DictConfig):
    # instead of print(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))

    logger = WandbLogger(
        name=cfg.run_name,
        project=cfg.project_name,
        offline=cfg.wandb_online is False,
        save_dir="./"
    )
    logger.log_hyperparams(cfg)
    run_root = os.path.dirname(logger.experiment.dir)

    # save the entire config to the log directory
    OmegaConf.save(cfg, os.path.join(run_root, "config.yaml"))

    train_loader = build_dataset(cfg, "train")
    val_loader = build_dataset(cfg, "val")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(run_root, "weight"),
        filename="best",
        monitor='metric_pose/10°2cm',
        mode='max',
        save_top_k=1,
        save_last=True)
    callbacks = [lr_monitor, ckpt_cb]

    trainer = Trainer(accelerator='gpu',
                      devices=[cfg.gpu],
                      precision=32,
                      max_epochs=cfg.num_epochs,
                      gradient_clip_val=0.1,
                      deterministic=False,
                      num_sanity_val_steps=1,
                      logger=logger,
                      callbacks=callbacks)

    module = DETRModule(cfg)

    # Check that the checkpoint path exists before passing to Trainer
    if len(cfg.ckpt_path) > 0:
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint path '{cfg.ckpt_path}' does not exist")
        ckpt = cfg.ckpt_path
    else:
        ckpt = None

    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt)


if __name__ == "__main__":
    run()
