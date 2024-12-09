import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Downsample(nn.Module):
    """
    自定义下采样层，用于实现滤波和下采样操作。
    """
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2)),
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2))
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.channels = channels

        # 根据滤波器大小生成权重
        if self.filt_size == 1:
            a = np.array([1.])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    """
    根据填充类型返回相应的填充层。
    """
    if pad_type in ['reflect', 'refl']:
        return nn.ReflectionPad2d
    elif pad_type in ['replicate', 'repl']:
        return nn.ReplicationPad2d
    elif pad_type == 'zero':
        return nn.ZeroPad2d
    else:
        raise ValueError(f"未知的填充类型: {pad_type}")

class BottleNeck(nn.Module):
    """
    ResNet的BottleNeck模块。
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class BBN_ResNet(nn.Module):
    """
    BBN网络的ResNet实现。
    """
    def __init__(self, num_classes, block, num_blocks, last_layer_stride=2):
        super(BBN_ResNet, self).__init__()
        self.inplanes = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            Downsample(filt_size=5, stride=2, channels=64)
        )

        # 构建残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=last_layer_stride)

        # 分类模块
        self.cb_block = block(self.inplanes, self.inplanes // 4, stride=1)
        self.rb_block = block(self.inplanes, self.inplanes // 4, stride=1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        cb_features = self.cb_block(x)
        rb_features = self.rb_block(x)
        features = torch.cat([cb_features, rb_features], dim=1)
        logits = self.fc(features.view(features.size(0), -1))
        return logits

def bbn_res50(num_classes, pretrain_path=None):
    """
    返回BBN的ResNet50实现。
    """
    model = BBN_ResNet(num_classes, BottleNeck, [3, 4, 6, 3])
    if pretrain_path:
        print(f"加载预训练模型: {pretrain_path}")
        model.load_state_dict(torch.load(pretrain_path))
    return model

class Network(nn.Module):
    """
    通用网络模块，支持多种骨干网络和分类器。
    """
    def __init__(self, num_classes):
        super(Network, self).__init__()

        self.backbone = bbn_res50(
            num_classes=num_classes,
            pretrain_path="F:\\iNaturalist\\resnet50-19c8e357.pth"
        )
        self.module = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048 * 2, self.num_classes, bias=True)

    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.classifier(x)

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def extract_feature(self, x, **kwargs):
        x = self.backbone(x, **kwargs)
        x = self.module(x)
        x = x.view(x.shape[0], -1)
        return x
