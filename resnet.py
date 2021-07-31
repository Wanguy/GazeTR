import torch.nn as nn
import torch.utils.model_zoo as model_zoo



model_urls = {
     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth' 
}

'''
in_planes：输入通道数
out_planes：输出的通道数
卷积步长 stride=1
扩张大小 dilation=1（也就是 padding）
groups 是分组卷积参数，这里 groups=1 相当于没有分组
'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


'''
super(BasicBlock, self).__init__() 这句是固定的标准写法。
一般神经网络的类都继承自 torch.nn.Module，__init()__ 和 forward() 是自定义类的两个主要函数，
在自定义类的 __init()__ 中需要添加一句 super(Net, self).__init()__，其中 Net 是自定义的类名，用于继承父类的初始化函数。
注意在 __init()__ 中只是对神经网络的模块进行了声明，真正的搭建是在 forward() 中实现。
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
与基础版本的 BasicBlock 不同的是这里有 3 个卷积，分别为1 * 1，3 * 3，1 * 1 大小的卷积核，分别用于压缩维度、卷积处理、恢复维度。

inplanes：输入通道数
planes：输出通道数 / expansion
expansion：对输出通道数的倍乘（注意在基础版本 BasicBlock 中 expansion 是 1，此时相当于没有倍乘，输出的通道数就等于 planes。）

在使用 Bottleneck 时，它先对通道数进行压缩，再放大，
所以传入的参数 planes 不是实际输出的通道数，而是 block 内部压缩后的通道数，真正的输出通道数为 plane * expansion
这样做的主要目的是，使用 Bottleneck 结构可以减少网络参数数量
'''
# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


'''
ResNet 共有五个阶段，其中第一阶段为一个 7*7 的卷积，stride = 2，padding = 3，
然后经过 BN、ReLU 和 maxpooling，此时特征图的尺寸已成为输入的 1/4

接下来是四个阶段，也就是代码中 layer1，layer2，layer3，layer4。
这里用 _make_layer 函数产生四个 Layer，
需要用户输入每个 layer 的 block 数目（即layers列表)以及采用的 block 类型（基础版 BasicBlock 还是 Bottleneck 版）
'''
class ResNet(nn.Module):

    def __init__(self, block, layers, maps=32):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.conv = nn.Sequential(
            nn.Conv2d(512, maps, 1),
            nn.BatchNorm2d(maps),
            nn.ReLU(inplace=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    '''
    block：选择要使用的模块（BasicBlock / Bottleneck）
    planes：该模块的输出通道数
    blocks：每个 blocks 中包含多少个 residual 子结构
    '''
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
          
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 
          
        x = self.conv(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model
