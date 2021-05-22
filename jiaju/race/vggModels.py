
import torch.nn as nn
import torch

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features

        # 构建分类网络结构
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 第一层全连接层
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 50%的比例随机失活
            nn.Linear(4096, 4096),  # 第二层全连接层
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)  # 第三层全连接层
        )
        if init_weights:  # 是否进行权重初始化
            self._initialize_weights()

    # 正向传播过程
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)  # 输入到特征提取网络
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)  # 展平处理，从第1维度展平(第0维度为batch)
        # N x 512*7*7
        x = self.classifier(x)  # 输入到分类网络中，得到输出
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 构建提取特征网络结构
def make_features(cfg: list):  # 传入对应配置的列表
    layers = []  # 定义空列表，存放每一层的结构
    in_channels = 3  # 输入为RGB图片，输入通道为3
    for v in cfg:  # 遍历配置列表
        if v == "M":  # 如果为M，则为池化层，创建一个最大池化下采样层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # 不等于M，则为数字，创建卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]  # 每个卷积层都采用RELU激活函数，将定义好的卷积层和RELU拼接
            in_channels = v
    return nn.Sequential(*layers)  # 非关键字参数，*layers可以传递任意数量的实参，以元组的形式导入


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 实例化配置模型
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)  # 可以传递任意数量的实参，以字典的形式导入
    return model
