# 导入相关包
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
# import matplotlib.pyplot as plt
import numpy as np
import os

# 设置超参量
modelPath='./RESNETmodel.pkl'
Learning_Rata = 0.001
Batch_Size = 500
Epoch = 100
# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]
)
transform_test= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]
)
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 获取训练集与测试集
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = Data.DataLoader(train_data, batch_size=Batch_Size, shuffle=True)

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = Data.DataLoader(test_data, batch_size=Batch_Size, shuffle=False)

# 分类
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义网络模型
'''
创建VGG块
参数分别为输入通道数，输出通道数，卷积层个数，是否做最大池化
'''
#深度残差网络ResNet18
class ResidualBlock(nn.Module):

    # 实现子module: Residual Block

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

def make_vgg_block(in_channel, out_channel, convs, pool=True):
    net = []

    # 不改变图片尺寸卷积
    net.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
    net.append(nn.BatchNorm2d(out_channel))
    net.append(nn.ReLU(inplace=True))

    for i in range(convs - 1):
        # 不改变图片尺寸卷积
        net.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        net.append(nn.BatchNorm2d(out_channel))
        net.append(nn.ReLU(inplace=True))

    if pool:
        # 2*2最大池化，图片变为w/2 * h/2
        net.append(nn.MaxPool2d(2))

    return nn.Sequential(*net)


# 定义网络模型

class ResNet(nn.Module):

    # 实现主module：ResNet34
    # ResNet34 包含多个layer，每个layer又包含多个residual block
    # 用子module来实现residual block，用_make_layer函数来实现layer

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(16, 16, 3)
        self.layer2 = self._make_layer(16, 32, 4, stride=1)
        self.layer3 = self._make_layer(32, 64, 6, stride=1)
        self.layer4 = self._make_layer(64, 64, 3, stride=1)
        self.fc = nn.Linear(256, num_classes)  # 分类用的全连接

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建layer,包含多个residual block
        shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 1, stride, bias=False), nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#net = VGG16Net().to(device)


if os.path.exists(modelPath):
    print('model exists')
    resnet = torch.load(modelPath)

    print('model loaded')
    net = resnet.cuda()

else:
    print('model not exists')
    net = ResNet().to(device)


print('Training Started')


# 训练模型
def train():
    # 参数优化
    optimizer = optim.SGD(net.parameters(), lr=Learning_Rata, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    # 训练网络
    for epoch in range(Epoch):
        time_start = time.time()
        runing_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # 将输入和目标在每一步都送入GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # forward+backward+optim
            outputs = net(inputs)
            loss = loss_func(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 统计数据
            runing_loss += loss.item()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, runing_loss / 500))
                runing_loss = 0.0
                #_, predicted = torch.max(outputs.data, 1)
                #total += labels.size(0)
                #correct += (predicted == labels).sum().item()
                print('Accuracy of the vgg16work on the %d train images: %.3f %%' % (total, 100.0 * correct / total))
        print('epoch %d cost %3f sec' % (epoch, time.time() - time_start))
        total = 0
        correct = 0

        print("save model......")
        torch.save(net, './VGG16model.pkl')


    print('Finished Training')


# # 从测试集中显示一个图像
# dataiter = iter(test_loader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#

# 测试模型
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集中不需要反向传播
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the vgg16work on the 10000 test images: %.3f %%' % (100.0 * correct / total))
    # return 100.0 * correct / total




if __name__ == '__main__':
    train()
    test()