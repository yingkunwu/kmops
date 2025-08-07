from .kmops import KMOPS
from .resnet import Resnet
from .encoder import HybridEncoder
from .decoder import DeformableTransformer
from .matcher import HungarianMatcher
from .loss import SetCriterion


def build_backbone(cfg):
    backbone = cfg.model.backbone
    if backbone in ["resnet50"]:
        backbone = Resnet("resnet50", True, True, False)
    elif backbone in ["resnet101"]:
        backbone = Resnet("resnet101", True, True, False)
    else:
        raise ValueError(f"Backbone {backbone} not supported")

    return backbone


def build_encoder(cfg):
    return HybridEncoder(
        in_channels=cfg.model.in_channels,
        feat_strides=cfg.model.feat_strides,
        use_encoder_idx=cfg.model.use_encoder_idx,
        num_encoder_layers=cfg.model.enc_layers,
        hidden_dim=cfg.model.hidden_dim,
        nhead=cfg.model.nheads,
        dropout=cfg.model.dropout,
        dim_feedforward=cfg.model.dim_feedforward
    )


def build_decoder(cfg):
    return DeformableTransformer(
        num_classes=len(cfg.dataset.names),
        hidden_dim=cfg.model.hidden_dim,
        num_queries=cfg.model.num_queries,
        num_keypoints=cfg.num_k,
        feat_channels=[cfg.model.hidden_dim] * len(cfg.model.feat_strides),
        feat_strides=cfg.model.feat_strides,
        num_levels=len(cfg.model.feat_strides),
        num_decoder_points=cfg.model.num_decoder_points,
        nhead=cfg.model.nheads,
        num_decoder_layers=cfg.model.dec_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        aux_loss=cfg.aux_loss,
    )


def build_model(cfg):
    model = KMOPS(
        backbone=build_backbone(cfg),
        encoder=build_encoder(cfg),
        decoder=build_decoder(cfg)
    )
    return model


def build_criterion(cfg):
    weight_dict = {
        "loss_ce": cfg.ce_loss_coef,
        "loss_oks": cfg.oks_loss_coef,
        "loss_kpts": cfg.kpts_loss_coef,
        "loss_conf": cfg.conf_loss_coef
    }

    return SetCriterion(
        num_classes=len(cfg.dataset.names),
        num_keypoints=cfg.num_k,
        num_queries=cfg.model.num_queries,
        weight_dict=weight_dict,
        matcher=HungarianMatcher(
            coef_class=cfg.o2o_coef_class,
            coef_kpts=cfg.o2o_coef_kpts,
            coef_oks=cfg.o2o_coef_oks,
            coef_conf=cfg.o2o_coef_conf,
            num_keypoints=cfg.num_k
        )
    )
