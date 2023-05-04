import torch.nn as nn
import torch
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixerBatch(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    m = nn.Sequential()
    # initial convolution
    m.add_module('c1', nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size))
    m.add_module('act',nn.GELU())
    m.add_module('norm',nn.BatchNorm2d(dim))
    # add residual blocks
    blocks=[(nn.Sequential(), nn.Sequential()) for i in range(depth)]
    for i, (m_1, m_2) in enumerate(blocks):
        m_1.add_module('conv_res',nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"))
        m_1.add_module('act',nn.GELU())
        m_1.add_module('norm',nn.BatchNorm2d(dim))
        m_2.add_module('conv',nn.Conv2d(dim, dim, kernel_size=1))
        m_2.add_module('act',nn.GELU())
        m_2.add_module('norm',nn.BatchNorm2d(dim))
        m.add_module('res_block_{}'.format(i), nn.Sequential(Residual(m_1),m_2))
    # pooling, flattening and linear output layer
    m.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
    m.add_module('flatten',nn.Flatten())
    m.add_module('linear',nn.Linear(dim, n_classes))
    return m

def ConvMixerGroup(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    m = nn.Sequential()
    # initial convolution
    m.add_module('c1', nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size))
    m.add_module('act',nn.GELU())
    m.add_module('norm',nn.GroupNorm(num_groups=32,num_channels=dim))
    # add residual blocks
    blocks=[(nn.Sequential(), nn.Sequential()) for i in range(depth)]
    for i, (m_1, m_2) in enumerate(blocks):
        m_1.add_module('conv_res',nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"))
        m_1.add_module('act',nn.GELU())
        m_1.add_module('norm',nn.GroupNorm(num_groups=32,num_channels=dim))
        m_2.add_module('conv',nn.Conv2d(dim, dim, kernel_size=1))
        m_2.add_module('act',nn.GELU())
        m_2.add_module('norm',nn.GroupNorm(num_groups=32,num_channels=dim))
        m.add_module('res_block_{}'.format(i), nn.Sequential(Residual(m_1),m_2))
    # pooling, flattening and linear output layer
    m.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
    m.add_module('flatten',nn.Flatten())
    m.add_module('linear',nn.Linear(dim, n_classes))
    return m

def ConvMixerLayer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    m = nn.Sequential()
    # initial convolution
    m.add_module('c1', nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size))
    m.add_module('act',nn.GELU())
    m.add_module('norm',LayerNorm(dim, data_format='channels_first'))
    # add residual blocks
    blocks=[(nn.Sequential(), nn.Sequential()) for i in range(depth)]
    for i, (m_1, m_2) in enumerate(blocks):
        m_1.add_module('conv_res',nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"))
        m_1.add_module('act',nn.GELU())
        m_1.add_module('norm',LayerNorm(dim, data_format='channels_first'))
        m_2.add_module('conv',nn.Conv2d(dim, dim, kernel_size=1))
        m_2.add_module('act',nn.GELU())
        m_2.add_module('norm',LayerNorm(dim, data_format='channels_first'))
        m.add_module('res_block_{}'.format(i), nn.Sequential(Residual(m_1),m_2))
    # pooling, flattening and linear output layer
    m.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
    m.add_module('flatten',nn.Flatten())
    m.add_module('linear',nn.Linear(dim, n_classes))
    return m


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x