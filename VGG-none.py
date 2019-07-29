#导入相关包
import torch
import torch.nn as nn
import torchvision.datasets as Data
import torchvision.transforms as transforms
from torch.autograd import Variable
#超参量
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCH = 5
DOWNLOAD_CIFAR10 = True

# transform = transforms.Compose([
#     transforms.RandomSizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                          std=[0.5, 0.5, 0.5]),
# ])

#获取训练与测试集
train_data = Data.CIFAR10(
    root='./CIFAR10/',
    train=True,  # this is training data
    transform=transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to tensor
    download=DOWNLOAD_CIFAR10,
)
text_data = Data.CIFAR10(root='./CIFAR10', train=False)
#数据转载
train_Loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_Loader = torch.utils.data.DataLoader(dataset=text_data, batch_size=BATCH_SIZE, shuffle=True)
#模型搭建
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(                        # input shape (3, 32, 32)

            # 1-1 conv layer
            nn.Conv2d(3, 32, kernel_size=3, padding=1),      # input shape (32, 28, 28)
            nn.BatchNorm2d(32),                              # .........
            nn.ReLU(),                                      # .........

            # 1-2 conv layer                                 # .........
            nn.Conv2d(32,32, kernel_size=3, padding=1),      # input shape (8, 28, 28)
            nn.BatchNorm2d(32),                              # .........
            nn.ReLU(),                                      # .........

            # 1 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2)         #input shape (32, 14, 14)
        )

        self.layer2 = nn.Sequential(

            # 2-1 conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  #input shape (16, 14, 14)
            nn.BatchNorm2d(64),                            # .........
            nn.ReLU(),                                     # .........

            # 2-2 conv layer
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),                             # .........
            nn.ReLU(),                                      # .........

            # 2 Pooling lyaer
            nn.MaxPool2d(kernel_size=2, stride=2)                #input shape (64, 7, 7)
        )

        self.layer3 = nn.Sequential(

            # 3-1 conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  #input shape (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 3-2 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   #input shape (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(

            # 4-1 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  #input shape (64, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 4-2 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),    # .........
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 4 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2)            #input shape (64, 3, 3)
        )

        self.layer5 = nn.Sequential(

            # 5-1 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),     #input shape (128, 3, 3)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 5-2 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 5 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2)           #input shape (128,1, 1)
        )

        self.layer6 = nn.Sequential(

            # 6 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(512,64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(

            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.layer8 = nn.Sequential(

            # 8 output layer
            nn.Linear(32, 10),
            nn.BatchNorm1d(10),
            nn.Softmax()
        )

    def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            vgg16_features = out.view(out.size(0), -1)
            out = self.layer6(vgg16_features)
            out = self.layer7(out)
            out = self.layer8(out)

            return vgg16_features, out


vgg16 = VGG16()
vgg16.cuda()
print(vgg16)

# Loss and Optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(EPOCH):
    for step, (images, labels) in enumerate(train_Loader):
        #for images, labels in train_Loader:

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        out = vgg16(images)[0]
        loss = cost(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#    if (i+1) % 100 == 0 :
#      print ('Epoch [%d/%d], Iter[%d/%d] Loss. %.4f' %
#          (epoch+1, EPOCH, i+1, len(trainData)//BATCH, loss.data[0]))

# Test the model
vgg16.eval()
correct = 0
total = 0

for images, labels in test_Loader:
    images = Variable(images).cuda()
    outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')