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

#设置超参量
Learning_Rata = 0.001
Batch_Size = 300
Epoch = 20
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

text_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
text_loader = Data.DataLoader(text_data, batch_size=Batch_Size, shuffle=True)

#模型搭建
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 1 conv layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, padding=2),                  # 64*16*16
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 2 conv layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),                                                  #128*8*8
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 3 conv layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),                                                     # 256*4*4
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 4 conv layer
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 5 conv layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 6 fc layer
        self.layer6 = nn.Sequential(
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
    def forward(self, x):
        x = self.layer1(x),
        x = self.layer2(x),
        x = self.layer3(x),
        x = self.layer4(x),
        x = self.layer5(x),
        x = x.view(-1, 512*1*1),
        x = self.layer6(x)

        return x

vgg16 = VGG16().to(device)
print(vgg16)


#训练函数
def train():

#参数优化
    optimizer = optim.Adam(vgg16.parameters(), lr=Learning_Rata)
    loss_func = nn.CrossEntropyLoss()
    iter = 0
    num = 1
#训练网络
    for epoch in range(Epoch):
        runing_loss = 0
        time_start = time.time()
        iter = iter+1
        for i, data in enumerate(train_loader, 0);
            # get the inputs
            inputs, labels = data
            # 将输入和目标在每一步都送入GPU
            inputs, labels = inputs.to(device), labels.to(device)
            #梯度清零
            optimizer.zero_grad()
            #forward+backward+optim
            outputs = self(inputs)
            loss = loss_func(outputs, labels).to(device)
            loss.backward()
            optimizer,step()









