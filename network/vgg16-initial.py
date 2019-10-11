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
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置超参量
modelPath='./VGG16model.pkl'
Learning_Rata = 0.001
Batch_Size = 500
Epoch = 100
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(227),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 获取训练集与测试集
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = Data.DataLoader(train_data, batch_size=Batch_Size, shuffle=True)

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = Data.DataLoader(test_data, batch_size=Batch_Size, shuffle=False)

# 分类
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义网络模型
'''
创建VGG块
参数分别为输入通道数，输出通道数，卷积层个数，是否做最大池化
'''
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
class VGG16Net(nn.Module):
    def __init__(self):
        super(VGG16Net, self).__init__()

        net = []

        # 输入32*32，(32-3+2*1)/2+1=16,输出16*16
        net.append(make_vgg_block(3, 64, 2))

        # (16-3+2*1)/2+1=8,输出8*8
        net.append(make_vgg_block(64, 128, 2))

        # (8-3+2*1)/2+1=4,输出4*4
        net.append(make_vgg_block(128, 256, 3))

        # (4-3+2*1)/2+1=2,输出2*2
        net.append(make_vgg_block(256, 512, 3))

        # (2-3+2*1)/2+1=1,输出1*1
        net.append(make_vgg_block(512, 512, 3, True))

        self.cnn = nn.Sequential(*net)

        self.fc = nn.Sequential(
            # 512个feature，每个feature 2*2
            nn.Linear(512*1*1, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

#net = VGG16Net().to(device)


if os.path.exists(modelPath):
    print('model exists')
    vgg16 = torch.load(modelPath)

    print('model loaded')
    net = vgg16.cuda()

else:
    print('model not exists')
    net = VGG16Net().to(device)


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