import torch as t
import os
import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import datetime

#定义全局变量
modelPath='./myVGGmodel.pkl'
batch_size=64
test_batch_size=1
epochs=20
log_interval=100
#定义Summary_Writer
writer=SummaryWriter('./Result')  #数据存放在这个文件夹
#cuda
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 样本读取线程数
WORKERS = 4

# 目标函数
loss_func = nn.CrossEntropyLoss()

# 最优结果
global best_acc
best_acc = 0




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


'''
训练并测试网络
net：网络模型
train_data_load：训练数据集
optimizer：优化器
epoch：第几次训练迭代
log_interval：训练过程中损失函数值和准确率的打印频率
'''
def net_train(net, train_data_load, optimizer, epoch, log_interval):
    net.train()

    begin = datetime.datetime.now()

    # 样本总数
    total = len(train_data_load.dataset)

    # 样本批次训练的损失函数值的和
    train_loss = 0

    # 识别正确的样本数
    ok = 0

    for i, data in enumerate(train_data_load, 0):
        img, label = data
        img, label = img.cuda(), label.cuda()

        optimizer.zero_grad()

        outs = net(img)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()

        # 累加损失值和训练样本数
        train_loss += loss.item()
        # total += label.size(0)

        _, predicted = t.max(outs.data, 1)
        # 累加识别正确的样本数
        ok += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            # 训练结果输出

            # 损失函数均值
            loss_mean = train_loss / (i + 1)

            # 已训练的样本数
            traind_total = (i + 1) * len(label)

            # 准确度
            acc = 100. * ok / traind_total

            # 一个迭代的进度百分比
            progress = 100. * traind_total / total

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, traind_total, total, progress, loss_mean, acc))

    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)


'''
用测试集检查准确率
'''
def net_test(net, test_data_load, epoch):
    net.eval()

    ok = 0

    for i, data in enumerate(test_data_load):
        img, label = data
        img, label = img.cuda(), label.cuda()

        outs = net(img)
        _, pre = t.max(outs.data, 1)
        ok += (pre == label).sum()

    acc = ok.item() * 100. / (len(test_data_load.dataset))
    print('EPOCH:{}, ACC:{}\n'.format(epoch, acc))

    global best_acc
    if acc > best_acc:
        best_acc = acc


'''
显示数据集中一个图片
'''
def img_show(dataset, index):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    show = ToPILImage()

    data, label = dataset[index]
    print('img is a ', classes[label])
    show((data + 1) / 2).resize((100, 100)).show()


def main():

    # 图像数值转换，ToTensor源码注释
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
        Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        """
# 归一化把[0.0, 1.0]变换为[-1,1], ([0, 1] - 0.5) / 0.5 = [-1, 1]

    transform = tv.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 加载数据集（训练集和测试集）
    train_data = tv.datasets.CIFAR10(root='./Cifar-10', train=True, download=True, transform=transform)
    test_data = tv.datasets.CIFAR10(root='./Cifar-10', train=False, download=False, transform=transform)

    train_load = t.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_load = t.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)


    #如果模型存在，加载模型
    if os.path.exists(modelPath):
        print('model exists')
        model = torch.load(modelPath)
        print('model loaded')
        net = model.to(device)
        #net_test(net, test_load, 0)
    else:
        print('model not exists')
        net = VGG16Net().to(device)

    #net = VGG16Net().cuda()
    
    #print(net)

    # 如果不训练，直接加载保存的网络参数进行测试集验证


    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start_time = datetime.datetime.now()

    for epoch in range(1, epochs + 1):

        net_train(net, train_load, optimizer, epoch, log_interval)

        # 每个epoch结束后用测试集检查识别准确度
        net_test(net, test_load, epoch)

        end_time = datetime.datetime.now()
        #保存模型
        torch.save(net,'./myVGGmodel.pkl')
    

    #global best_acc
    print('CIFAR10 pytorch VGGNet Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(epochs, batch_size, 0.001, best_acc))
    print('train spend time: ', end_time - start_time)
    print('Training Finished')


if __name__=='__main__':
   
    main()
