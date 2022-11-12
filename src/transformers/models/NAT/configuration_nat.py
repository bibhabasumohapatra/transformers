""" Neigbourhood Attention Transformer model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...utils import logging

import torch.nn as nn

logger = logging.get_logger(__name__)

SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nat_mini": (
        "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth"
    ),
}


class NATConfig(PretrainedConfig):
    
    model_type = "NAT"

    # base model nat_mini

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }


    def __init__(
        self,
        image_size=224,
        num_channels=3,
        embed_dim=64,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        kernel_size=7,
        mlp_ratio=3.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.2,
        dilations=None,
        hidden_act="gelu",
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.qk_scale = qk_scale,
        self.drop_rate = drop_rate,
        self.attn_drop_rate = attn_drop_rate,
        self.norm_layer = norm_layer ,
        self.layer_scale = layer_scale,
        
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))