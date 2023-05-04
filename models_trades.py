# from Maksyms understanding SAM code
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []

    def forward(self, preact):
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())
        act = F.relu(preact)
        return act


class ModuleWithStats(nn.Module):
    def __init__(self):
        super(ModuleWithStats, self).__init__()

    def forward(self, x):
        for layer in self._model:
            if type(layer) == CustomReLU:
                layer.avg_preacts = []

        out = self._model(x)

        avg_preacts_all = [layer.avg_preacts for layer in self._model if type(layer) == CustomReLU]
        self.avg_preact = np.mean(avg_preacts_all)
        return out


class PreActBlock(nn.Module):
    """ Pre-activation version of the BasicBlock. """
    expansion = 1

    def __init__(self, in_planes, planes, learnable_bn, norm=nn.BatchNorm2d, stride=1, activation='relu', droprate=0.0,
                 gn_groups=32, in_shape=(3, 32, 32), out_shape=(3, 32, 32)):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.droprate = droprate
        self.avg_preacts = []
        self.bn1 = norm(in_planes) if 'LayerNormSmall' in str(norm) else (
            norm(normalized_shape=in_shape) if 'Layer' in str(norm) else (
                norm(in_planes, affine=learnable_bn) if 'Batch' in str(norm) else norm(gn_groups, in_planes)))
        #         self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else nn.GroupNorm(gn_groups, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        #         self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else nn.GroupNorm(gn_groups, planes)
        self.bn2 = norm(planes) if 'LayerNormSmall' in str(norm) else (
            norm(normalized_shape=out_shape) if 'Layer' in str(norm) else (
                norm(planes, affine=learnable_bn) if 'Batch' in str(norm) else norm(gn_groups, planes)))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
            # print((act == 0).float().mean().item(), (act.norm() / act.shape[0]).item(), (act.norm() / np.prod(act.shape)).item())
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        #         print('PreactBlock in, ', x.shape)
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        #         print('PreactBlock out, ', out.shape)
        out = self.act_function(self.bn2(out))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out += shortcut

        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, model_width=64, cuda=True, half_prec=False, activation='relu',
                 droprate=0.0, norm=nn.BatchNorm2d):
        super(PreActResNet, self).__init__()
        self.half_prec = half_prec
        self.bn_flag = 'Batch' in str(norm)
        self.gn_groups = model_width // 2  # in particular, 32 for model_width=64 as in the original GroupNorm paper
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = model_width
        self.avg_preact = None
        self.activation = activation
        self.n_cls = n_cls
        self.norm = norm
        # self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        # self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
        self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)

        if cuda:
            self.mu, self.std = self.mu.cuda(), self.std.cuda()

        # if half_prec:
        #     self.mu, self.std = self.mu.half(), self.std.half()
        # compute output shape of convolutional layer
        def get_output_shape(planes, h_in, kernel_size, stride, dilation, padding):
            h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            return (planes, int(h_out), int(h_out))

        output_shape = (3, 32, 32)
        self.normalize = Normalize(self.mu, self.std)
        input_shape = output_shape
        output_shape = get_output_shape(model_width, output_shape[1], kernel_size=3, stride=1, dilation=1, padding=1)
        #         print(output_shape)
        self.conv1 = nn.Conv2d(3, model_width, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        input_shape = output_shape
        output_shape = get_output_shape(model_width, output_shape[1], kernel_size=3, stride=1, dilation=1, padding=1)
        #         print(output_shape)
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], 1, droprate, in_shape=input_shape,
                                       out_shape=output_shape)
        input_shape = output_shape
        output_shape = get_output_shape(2 * model_width, output_shape[1], kernel_size=3, stride=2, dilation=1,
                                        padding=1)
        #         print(output_shape)
        self.layer2 = self._make_layer(block, 2 * model_width, num_blocks[1], 2, droprate, in_shape=input_shape,
                                       out_shape=output_shape)
        input_shape = output_shape
        output_shape = get_output_shape(4 * model_width, output_shape[1], kernel_size=3, stride=2, dilation=1,
                                        padding=1)
        #         print(output_shape)
        self.layer3 = self._make_layer(block, 4 * model_width, num_blocks[2], 2, droprate, in_shape=input_shape,
                                       out_shape=output_shape)
        input_shape = output_shape
        output_shape = get_output_shape(8 * model_width, output_shape[1], kernel_size=3, stride=2, dilation=1,
                                        padding=1)
        #         print(output_shape)
        self.layer4 = self._make_layer(block, 8 * model_width, num_blocks[3], 2, droprate, in_shape=input_shape,
                                       out_shape=output_shape)
        self.bn = self.norm(8 * model_width * block.expansion) if 'LayerNormSmall' in str(norm) else (
            self.norm(normalized_shape=output_shape) if 'Layer' in str(self.norm) else (
                self.norm(8 * model_width * block.expansion) if 'Batch' in str(self.norm) else self.norm(self.gn_groups,
                                                                                                         8 * model_width * block.expansion)))
        self.linear = nn.Linear(8 * model_width * block.expansion, 1 if n_cls == 2 else n_cls)

    # compute output shape of convolutional layer
    def get_output_shape(planes, h_in, kernel_size, stride, dilation, padding):
        h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        return (planes, int(h_out), int(h_out))

    def _make_layer(self, block, planes, num_blocks, stride, droprate, in_shape=(3, 32, 32), out_shape=(3, 32, 32)):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, self.learnable_bn, self.norm, stride, self.activation,
                                droprate, self.gn_groups, in_shape=in_shape if i == 0 else out_shape,
                                out_shape=out_shape))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        # x = x / ((x**2).sum([1, 2, 3], keepdims=True)**0.5 + 1e-6)  # numerical stability is needed for RLAT
        out = self.normalize(x)
        out = self.conv1(out)
        #         print(out.shape)
        out = self.layer1(out)
        #         print(out.shape)
        out = self.layer2(out)
        #         print(out.shape)
        out = self.layer3(out)
        #         print(out.shape)
        out = self.layer4(out)
        #         print(out.shape)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if out.shape[1] == 1:
            out = torch.cat([torch.zeros_like(out), out], dim=1)

        return out


def PreActResNet34BatchNorm(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', droprate=0.0):
    #     bn_flag = True
    return PreActResNet(PreActBlock, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, droprate=droprate, norm=nn.BatchNorm2d)


def PreActResNet34GroupNorm(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', droprate=0.0):
    #     bn_flag = False  # bn_flag==False means that we use GroupNorm with 32 groups
    return PreActResNet(PreActBlock, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, droprate=droprate, norm=nn.GroupNorm)


def PreActResNet34LayerNorm(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', droprate=0.0):
    #     bn_flag = False  # bn_flag==False means that we use GroupNorm with 32 groups
    return PreActResNet(PreActBlock, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, droprate=droprate, norm=nn.LayerNorm)


def PreActResNet34LayerNormSmall(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', droprate=0.0):
    #     bn_flag = False  # bn_flag==False means that we use GroupNorm with 32 groups
    return PreActResNet(PreActBlock, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, droprate=droprate, norm=LayerNormSmall)


def init_weights(model, scale_init=0.0):
    def init_weights_linear(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # m.weight.data.zero_()
            m.weight.data.normal_()
            m.weight.data *= scale_init / (m.weight.data ** 2).sum() ** 0.5
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights_he(m):
        # if isinstance(m, nn.Conv2d):
        #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        # elif isinstance(m, nn.Linear):
        #     n = m.in_features
        #     m.weight.data.normal_(0, math.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

        # From Rice et al.
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    if model == 'linear':
        return init_weights_linear
    else:
        return init_weights_he


class LayerNormSmall(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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