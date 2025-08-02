import torch
from torch import nn
import torch.nn.functional as F

from utils.util import NestedTensor


class KMOPS(nn.Module):
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.add_module("backbone", backbone)
        self.add_module("encoder", encoder)
        self.add_module("decoder", decoder)

    def forward(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
        - samples.mask: a binary mask of shape [batch_size x H x W],
                        containing 1 on padded pixels

        It returns a dict with the following elements:
          "pred_logits" [batch_size x num_queries x (num_classes + 1)]:
            the classification logits (including no-object) for all queries.
         "pred_boxes" [batch_size x num_queries x 4]:
            The normalized boxes coordinates for all queries, represented as
            (center_x, center_y, height, width). These values are normalized
            in [0, 1], relative to the size of each individual image
            (disregarding possible padding). See PostProcess for information
            on how to retrieve the unnormalized bounding box.
        """
        src_l, src_r, mask = samples.decompose()
        features = self.backbone(torch.cat([src_l, src_r], dim=0))

        src_list, mask_list = [], []
        for layer, src in features.items():
            if layer == '0':
                continue
            m = mask.clone()
            m = F.interpolate(m[None].float(), size=src.shape[-2:])
            m = m[0].to(torch.bool)

            src_list.append(src)
            mask_list.append(m)

        src_list = self.encoder(src_list, mask_list)
        out = self.decoder(src_list, mask_list)

        return out
