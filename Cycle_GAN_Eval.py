import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_data():
    colors = [transforms.Lambda(lambda x: torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), 0)),
              transforms.Lambda(lambda x: torch.cat((torch.zeros_like(x), x, torch.zeros_like(x)), 0)),
              transforms.Lambda(lambda x: torch.cat((torch.zeros_like(x), torch.zeros_like(x), x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0 * x, x, x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, 0 * x, x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.8 * x, 0.2 * x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.85 * x, 0.2 * x, 0.35 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.64 * x, 0.4 * x, 0.66 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.17 * x, 0.5 * x, 0.55 * x), 0))]
    ds_lst = []
    for i in range(10):
        mnist_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.Compose(
            [transforms.ToTensor(), colors[i], transforms.Normalize((0.5,), (0.5,))]), download=True)
        index = [idx for idx, target in enumerate(mnist_dataset.targets) if target in [i]]
        ds_lst.append(torch.utils.data.Subset(mnist_dataset, index))

    train_data = torch.utils.data.ConcatDataset(ds_lst)

    # transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: color_mnist1(x,mnist_dataset.targets[x[2]] ))])
    data_loader1 = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True)

    colors = [transforms.Lambda(lambda x: torch.cat((0.17 * x, 0.5 * x, 0.55 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), 0)),
              transforms.Lambda(lambda x: torch.cat((torch.zeros_like(x), x, torch.zeros_like(x)), 0)),
              transforms.Lambda(lambda x: torch.cat((torch.zeros_like(x), torch.zeros_like(x), x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0 * x, x, x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, 0 * x, x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.8 * x, 0.2 * x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.85 * x, 0.2 * x, 0.35 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.64 * x, 0.4 * x, 0.66 * x), 0))
              ]
    ds_lst2 = []
    for i in range(10):
        mnist_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.Compose(
            [transforms.ToTensor(), colors[i], transforms.Normalize((0.5,), (0.5,))]), download=True)
        index = [idx for idx, target in enumerate(mnist_dataset.targets) if target in [i]]
        ds_lst2.append(torch.utils.data.Subset(mnist_dataset, index))

    train_data2 = torch.utils.data.ConcatDataset(ds_lst2)
    data_loader2 = torch.utils.data.DataLoader(
        train_data2,
        batch_size=64,
        shuffle=True)

    return data_loader1, data_loader2
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.Block3_64 = nn.Sequential(nn.Conv2d(3,64,3,1,1),nn.LeakyReLU(0.2),nn.AvgPool2d(2))
    self.Block64_128 = nn.Sequential(nn.Conv2d(64,128,3,1,2),nn.InstanceNorm2d(128),nn.LeakyReLU(0.2),nn.AvgPool2d(2))
    self.Block128_256 = nn.Sequential(nn.Conv2d(128,256,3,1,1),nn.InstanceNorm2d(256),nn.LeakyReLU(0.2),nn.AvgPool2d(2))
    self.Block256_512 = nn.Sequential(nn.Conv2d(256,512,3,1,1),nn.InstanceNorm2d(512),nn.LeakyReLU(0.2),nn.AvgPool2d(2))
    self.Block512_1024 = nn.Sequential(nn.Conv2d(512,1024,3,1,1),nn.InstanceNorm2d(1024),nn.LeakyReLU(0.2),nn.AvgPool2d(2))
    self.linear = nn.Linear(1024,1)

  def forward(self,x):
    x = self.Block3_64(x)
    # print(x.shape)
    x = self.Block64_128(x)
    # print(x.shape)
    x = self.Block128_256(x)
    # print(x.shape)
    x = self.Block256_512(x)
    # print(x.shape)
    x = self.Block512_1024(x)
    # print(x.shape)
    x = x.reshape(-1,1024)
    # print(x.shape)
    return F.sigmoid(self.linear(x))


from torch.nn.modules.conv import ConvTranspose2d


class Block(nn.Module):
    def __init__(self, in_, out, down=True, act=True, **Kwargs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_, out, **Kwargs) if down
                                  else nn.ConvTranspose2d(in_, out, **Kwargs),
                                  nn.InstanceNorm2d(out),
                                  nn.ReLU(inplace=True) if act else nn.Identity())

    def forward(self, x):
        return self.conv(x)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            Block(channels, channels, kernel_size=3, padding=1),
            Block(channels, channels, act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_residuals=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.down_blocks = nn.ModuleList([
            Block(64, 128, kernel_size=3, stride=2, padding=1),
            Block(128, 256, kernel_size=3, stride=2)
        ])
        self.residuals = nn.Sequential(*[Residual(256) for _ in range(num_residuals)])

        self.up_blocks = nn.ModuleList([
            Block(256, 128, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
            Block(128, 64, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
            Block(64, 32, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
            Block(32, 16, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
        ])

        self.last = nn.Conv2d(16, img_channels, kernel_size=6, stride=1, padding=0)

    def forward(self, x):
        x = self.initial(x)

        # print(x.shape)
        for layer in self.down_blocks:
            x = layer(x)
            # print(x.shape)
        x = self.residuals(x)
        # print(x.shape)
        for layer in self.up_blocks:
            x = layer(x)
            # print(x.shape)
        x = torch.tanh(self.last(x))
        # print(x.shape)
        return x
G1 = pickle.load(open("Generator1.pkl", 'rb'))
G2= pickle.load(open("Generator2.pkl", 'rb'))
D1= pickle.load(open("Discriminator1.pkl", 'rb'))
D2= pickle.load(open("Discriminator2.pkl", 'rb'))