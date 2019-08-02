#导入相关包
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


#设置超参量
Learning_Rata = 0.001
Batch_Size = 500
Epoch = 100
#数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(227),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
)
#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#获取训练集与测试集
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = Data.DataLoader(train_data, batch_size=Batch_Size, shuffle=True)

test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = Data.DataLoader(test_data, batch_size=Batch_Size, shuffle=False)

#分类
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#打印训练集

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()



# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))




#模型搭建
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # 1 conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        # 2 conv layer

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        # 3 conv layer

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        # 4 conv layer

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

        # 5 conv layer

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # 6 fc layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Dropout(0.5),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 512*1*1)
        out = self.linear(out)

        return out


vgg16 = VGG16().to(device)
print(vgg16)


# 训练模型
def train():

# 参数优化
    optimizer = optim.SGD(vgg16.parameters(), lr=Learning_Rata, momentum=0.9)
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
            #梯度清零
            optimizer.zero_grad()
            #forward+backward+optim
            outputs = vgg16(inputs)
            loss = loss_func(outputs, labels).to(device)
            loss.backward()
            optimizer.step()
            #统计数据
            runing_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, runing_loss / 500))
                runing_loss = 0.0
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))
        print('epoch %d cost %3f sec' % (epoch, time.time()-time_start))
    print('Finished Training')
    
#从测试集中显示一个图像
dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))    
    
    
    #测试模型
def test():
    correct = 0
    total = 0
    with torch.no_grad():               # 测试集中不需要反向传播
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            _, predicted = torch.max(outputs.data, 1)            # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.3f %%' % (100.0 * correct / total))
    # return 100.0 * correct / total


torch.save(vgg16, './vgg16model.pkl')

if __name__ == '__main__':
    train()
    test()

