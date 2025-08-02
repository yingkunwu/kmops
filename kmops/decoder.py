import math
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat

from utils.util import kpts_pose_disp_to_left_right
from .util import (
    deformable_attention_core_func, get_activation, inverse_sigmoid,
    bias_init_with_prob,
)


class MLP(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=3,
                 num_points=4,
                 num_queries=300):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_queries = num_queries
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 4)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) \
            * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2)\
            .tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(
            1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        grid_init = torch.cat([grid_init, grid_init], dim=-1)
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self,
                query,
                reference_points_l,
                reference_points_r,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor):
                [bs, query_length, n_levels, num_keypoints * 2 + 4],
                range in [0, 1], top-left (0,0), bottom-right (1, 1),
                including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2],
                [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length],
                True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        _, len_in, _ = value.shape
        value_spatial_shapes = torch.as_tensor(
            value_spatial_shapes, dtype=torch.long, device=value.device)
        assert (
            value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]
        ).sum() == len_in

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = rearrange(value, 'b l (h d) -> b l h d', h=self.num_heads)

        sampling_offsets = rearrange(
            self.sampling_offsets(query),
            'bs lq (nh nl np s) -> bs lq nh nl np s',
            nh=self.num_heads, nl=self.num_levels, np=self.num_points, s=4)
        attention_weights = rearrange(
            self.attention_weights(query),
            'bs lq (nh nl np) -> bs lq nh (nl np)',
            nh=self.num_heads, nl=self.num_levels, np=self.num_points)
        attention_weights = rearrange(
            F.softmax(attention_weights, dim=-1),
            'bs lq nh (nl np) -> bs lq nh nl np',
            nl=self.num_levels, np=self.num_points)

        if reference_points_l.shape[-1] == 2:
            total_nq = reference_points_l.shape[1]

            # nq x nk = lq (length of query)
            reference_points_l = rearrange(
                reference_points_l, 'b nq nl nk s -> b (nq nk) nl s', s=2)
            reference_points_r = rearrange(
                reference_points_r, 'b nq nl nk s -> b (nq nk) nl s', s=2)

            offset_normalizer = torch.stack(
                [value_spatial_shapes[:, 1], value_spatial_shapes[:, 0]], -1)

            sampling_locs_l = (
                reference_points_l[:, :, None, :, None, :]
                + sampling_offsets[..., :2]
                / offset_normalizer[None, None, None, :, None, :]
            )
            sampling_locs_r = (
                reference_points_r[:, :, None, :, None, :]
                + sampling_offsets[..., 2:]
                / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2, but get {} instead.".
                format(reference_points_l.shape[-1]))

        sampling_locs = torch.cat([sampling_locs_l, sampling_locs_r], dim=-2)
        attention_weights_ = torch.cat(
            [attention_weights, attention_weights], dim=-1)

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locs, attention_weights_)
        output = self.output_proj(output)

        sampling_locs_l = rearrange(
            sampling_locs_l,
            'bs (nq n) nh nl np s -> bs nq n (nh nl np) s', nq=total_nq)
        sampling_locs_r = rearrange(
            sampling_locs_r,
            'bs (nq n) nh nl np s -> bs nq n (nh nl np) s', nq=total_nq)
        attention_weights = rearrange(
            attention_weights,
            'bs (nq n) nh nl np -> bs nq n (nh nl np)', nq=total_nq)

        log_info = (
            sampling_locs_l[:, :, 0].detach(),  # only take the center
            attention_weights[:, :, 0].detach(),
            sampling_locs_r[:, :, 0].detach(),  # only take the center
            attention_weights[:, :, 0].detach(),
        )

        return output, log_info


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 num_queries=300):
        super(TransformerDecoderLayer, self).__init__()

        # within-instance self-attention
        self.within_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True)
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)

        # across-instance self-attention
        self.across_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True)
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, num_queries)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        if pos is not None:
            # pos = torch.cat([torch.zeros_like(pos[:, :, :1]), pos], dim=2)
            return tensor + pos
        return tensor

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                reference_points_l,
                reference_points_r,
                memory,
                memory_spatial_shapes,
                query_pos_embed=None,
                memory_mask=None,
                o2m_match=False,
                attn_mask=None):

        # nq- > number of queries
        # n -> number of keypoints + 1
        # d -> hidden dimension
        bs, nq, n, d = tgt.shape

        # within-instance self-attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        tgt2, _ = self.within_attn(
            rearrange(q, 'bs nq n d -> (bs nq) n d'),
            rearrange(k, 'bs nq n d -> (bs nq) n d'),
            value=rearrange(tgt, 'bs nq n d -> (bs nq) n d'))
        tgt2 = rearrange(tgt2, '(bs nq) n d -> bs nq n d', bs=bs)
        tgt = tgt + self.within_dropout(tgt2)
        tgt = self.within_norm(tgt)

        # across-instance self-attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2, _ = self.across_attn(
            rearrange(q, 'bs nq n d -> (bs n) nq d'),
            rearrange(k, 'bs nq n d -> (bs n) nq d'),
            value=rearrange(tgt, 'bs nq n d -> (bs n) nq d'),
            attn_mask=attn_mask)
        tgt2 = rearrange(tgt2, '(bs n) nq d -> bs nq n d', bs=bs)
        tgt = tgt + self.across_dropout(tgt2)
        tgt = self.across_norm(tgt)

        # deformable cross-attention
        query = self.with_pos_embed(tgt, query_pos_embed)
        query = rearrange(query, 'bs nq n d -> bs (nq n) d')
        tgt2, log_info = self.cross_attn(
            query,
            reference_points_l,
            reference_points_r,
            memory,
            memory_spatial_shapes,
            value_mask=memory_mask)
        tgt2 = rearrange(tgt2, 'bs (nq n) d -> bs nq n d', nq=nq)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt_o2o = self.forward_ffn(tgt)

        return tgt_o2o, log_info


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    @staticmethod
    def prepend_center(r):
        ctr = r.mean(dim=2, keepdim=True)
        return torch.cat([ctr, r], dim=2)

    def forward(self,
                tgt,
                ref_mid,
                ref_disp,
                memory,
                memory_spatial_shapes,
                score_head,
                pose_head,
                disp_head,
                conf_head,
                query_pose_head,
                memory_mask=None,
                o2m_match=False,
                attn_mask=None):
        output = tgt
        log_infos = []

        outs = {
            'o2o_logits': [], 'o2o_pose': [], 'o2o_disp': [], 'o2o_conf': []
        }

        # Initial “detached” refs
        ref_mid_det = ref_mid.detach()
        ref_disp_det = ref_disp.detach()

        for i, layer in enumerate(self.layers):
            # Build the joint query-pos embedding from midpoints + disparity
            joint = torch.cat([ref_mid_det, ref_disp_det], dim=-1)
            query_pos_embed = query_pose_head(self.prepend_center(joint))

            ref_l_det, ref_r_det \
                = kpts_pose_disp_to_left_right(ref_mid_det, ref_disp_det)

            output, log_info = layer(
                output,
                self.prepend_center(ref_l_det).unsqueeze(2),
                self.prepend_center(ref_r_det).unsqueeze(2),
                memory,
                memory_spatial_shapes,
                query_pos_embed,
                memory_mask,
                o2m_match=o2m_match,
                attn_mask=attn_mask
            )
            log_infos.append(log_info)

            cls_out = score_head[i](output[:, :, 0])
            pose_out = pose_head[i](output[:, :, 1:])
            disp_out = disp_head[i](output[:, :, 1:])
            conf_out = conf_head[i](output[:, :, 1:])

            outs['o2o_logits'].append(cls_out)
            outs['o2o_conf'].append(conf_out)
            inter_ref_mid = F.sigmoid(
                pose_out + inverse_sigmoid(ref_mid_det))
            inter_ref_disp = F.sigmoid(
                disp_out + inverse_sigmoid(ref_disp_det))

            if i == 0:
                ref_mid_o2o = inter_ref_mid
                ref_disp_o2o = inter_ref_disp
            else:
                ref_mid_o2o = F.sigmoid(pose_out + inverse_sigmoid(ref_mid))
                ref_disp_o2o = F.sigmoid(disp_out + inverse_sigmoid(ref_disp))

            outs['o2o_pose'].append(ref_mid_o2o)
            outs['o2o_disp'].append(ref_disp_o2o)

            ref_mid = inter_ref_mid
            ref_mid_det = ref_mid.detach()
            ref_disp = inter_ref_disp
            ref_disp_det = ref_disp.detach()

        return {
            'o2o_out_logits': outs['o2o_logits'],
            'o2o_out_pose': outs['o2o_pose'],
            'o2o_out_disp': outs['o2o_disp'],
            'o2o_out_conf': outs['o2o_conf'],
            'log_infos': log_infos
        }


class DeformableTransformer(nn.Module):
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 num_keypoints=17,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eps=1e-2,
                 aux_loss=True):

        super(DeformableTransformer, self).__init__()
        assert len(feat_channels) == num_levels  # only support this case
        assert len(feat_strides) == len(feat_channels)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_keypoints = num_keypoints
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # denoising training
        self.num_denoising = 50
        self.denoising_class_embed = nn.Embedding(
            num_classes + 1, hidden_dim, padding_idx=num_classes)

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            n_head=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_levels,
            n_points=num_decoder_points,
            num_queries=num_queries)
        self.decoder = TransformerDecoder(
            hidden_dim=hidden_dim,
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.keypoint_embedding = nn.Embedding(num_keypoints, hidden_dim)
        self.instance_embedding = nn.Embedding(1, hidden_dim)

        self.query_pose_head = MLP(3, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # TODO: change variable names from pose to kpts
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_pose_head = MLP(
            hidden_dim, hidden_dim, num_keypoints * 2, num_layers=3)
        self.enc_disp_head = MLP(
            hidden_dim, hidden_dim, num_keypoints * 1, num_layers=3)
        self.enc_conf_head = MLP(
            hidden_dim, hidden_dim, num_keypoints * 1, num_layers=2)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_pose_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 2, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self.dec_disp_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 1, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self.dec_conf_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 1, num_layers=2)
            for _ in range(num_decoder_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s),
                 int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides]
            self.anchors, self.valid_mask = \
                self._generate_anchors(spatial_shapes)

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_conf_head.layers[-1].bias, bias)
        init.constant_(self.enc_pose_head.layers[-1].weight, 0)
        init.constant_(self.enc_pose_head.layers[-1].bias, 0)
        init.constant_(self.enc_disp_head.layers[-1].weight, 0)
        init.constant_(self.enc_disp_head.layers[-1].bias, 0)

        for cls_, conf_, reg1_, reg2_ in zip(
                self.dec_score_head, self.dec_conf_head,
                self.dec_pose_head, self.dec_disp_head):
            init.constant_(cls_.bias, bias)
            init.constant_(conf_.layers[-1].bias, bias)
            init.constant_(reg1_.layers[-1].weight, 0)
            init.constant_(reg1_.layers[-1].bias, 0)
            init.constant_(reg2_.layers[-1].weight, 0)
            init.constant_(reg2_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pose_head.layers[0].weight)
        init.xavier_uniform_(self.query_pose_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(
                        in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

    def _get_encoder_input(self, feats, masks):
        # get encoder inputs
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, (feat, mask) in enumerate(zip(feats, masks)):
            # TODO: check if this is necessary
            feat = self.input_proj[i](feat)  # get projection features

            _, _, h, w = feat.shape
            feat_flatten.append(
                rearrange(feat, '(n b) c h w -> b (h w) (n c)', n=2))
            mask_flatten.append(rearrange(mask, 'b h w -> b (h w)'))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        level_start_index.pop()
        return feat_flatten, mask_flatten, spatial_shapes, level_start_index

    def _generate_anchors(self,
                          spatial_shapes,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            anchors.append(grid_xy.reshape(-1, h * w, 2))

        anchors = torch.cat(anchors, 1).to(device)
        valid_mask = (
            (anchors > self.eps) * (anchors < 1 - self.eps)
        ).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))  # inverse sigmoid
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes):
        bs = memory.shape[0]

        # Generate or load anchors + valid mask
        if self.training or self.eval_spatial_size is None:
            anchors, valid = self._generate_anchors(
                spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors.to(memory.device)
            valid = self.valid_mask.to(memory.device)

        # Mask memory invalid positions
        memory = memory * valid.to(memory.dtype)

        # Encoder outputs
        enc_mem = self.enc_output(memory)
        cls_out = self.enc_score_head(enc_mem)
        pose_out = self.enc_pose_head(enc_mem)
        disp_out = self.enc_disp_head(enc_mem)
        conf_out = self.enc_conf_head(enc_mem)

        # pose_out: [B, L, nk * 2] before sigmoid
        anchors = repeat(anchors, '1 l p -> b l (q p)',
                         b=bs, q=self.num_keypoints)
        pose_out = F.sigmoid(pose_out + anchors)
        disp_out = F.sigmoid(disp_out)

        # Select top-K queries by classification score
        scores, _ = cls_out.max(-1)
        _, topk_idx = scores.topk(self.num_queries, dim=1)

        def gather(x):
            return x.gather(1, repeat(topk_idx, 'b q -> b q c', c=x.shape[-1]))

        # Gather and reshape
        topk_logits = gather(cls_out)
        topk_pose = gather(pose_out)  # average_pose
        topk_disp = gather(disp_out)
        topk_conf = gather(conf_out)

        topk_pose = rearrange(topk_pose, 'b q (nk p) -> b q nk p', p=2)
        topk_disp = rearrange(topk_disp, 'b q nk -> b q nk 1')
        topk_conf = rearrange(topk_conf, 'b q nk -> b q nk 1')

        # Initial decoder input (learned or from enc_mem)
        if self.learnt_init_query:
            target = repeat(self.tgt_embed.weight, 'q d -> b q d', b=bs)
        else:
            target = gather(enc_mem).detach()

        # Build query embeddings: global + pose
        tgt_global = repeat(self.instance_embedding.weight, 'i d -> b q i d',
                            b=bs, q=target.shape[1])
        tgt_pose = repeat(self.keypoint_embedding.weight, 'k d -> b q k d',
                          b=bs, q=target.shape[1])
        target = torch.cat(
            [tgt_global, tgt_pose + target.unsqueeze(-2)], dim=2)

        return target, topk_logits, topk_pose, topk_disp, topk_conf

    def forward(self, feats, masks):
        # input projection and embedding
        memory, padding_mask, spatial_shapes, _ \
            = self._get_encoder_input(feats, masks)

        target, topk_logits, topk_pose, topk_disp, topk_conf \
            = self._get_decoder_input(memory, spatial_shapes)

        # decoder
        result = self.decoder(
            target,
            topk_pose,
            topk_disp,
            memory,
            spatial_shapes,
            self.dec_score_head,
            self.dec_pose_head,
            self.dec_disp_head,
            self.dec_conf_head,
            self.query_pose_head,
            memory_mask=padding_mask)

        result["o2o_out_logits"] = torch.stack(result["o2o_out_logits"], dim=0)
        result["o2o_out_pose"] = torch.stack(result["o2o_out_pose"], dim=0)
        result["o2o_out_disp"] = torch.stack(result["o2o_out_disp"], dim=0)
        result["o2o_out_conf"] = torch.stack(result["o2o_out_conf"], dim=0)

        out = self._make_pred(
            result["o2o_out_logits"][-1],
            result["o2o_out_pose"][-1],
            result["o2o_out_disp"][-1],
            result["o2o_out_conf"][-1],
        )

        if self.aux_loss:
            out['enc_outputs'] = self._make_pred(
                topk_logits, topk_pose, topk_disp, topk_conf
            )

            out['aux_outputs'] = self._set_aux_output(
                result["o2o_out_logits"][:-1],
                result["o2o_out_pose"][:-1],
                result["o2o_out_disp"][:-1],
                result["o2o_out_conf"][:-1],
            )

        # Log deformable sampling points information for visualization
        out.update({"log_infos": result['log_infos'][-1]})
        return out

    def _make_pred(self, logits, pose, disp, conf):
        return {
            'pred_logits': logits,
            'pred_pose': pose,
            'pred_disp': disp,
            'pred_conf': conf,
        }

    def _set_aux_output(self, *output_list):
        return [
            self._make_pred(*out) for out in zip(*output_list)
        ]
