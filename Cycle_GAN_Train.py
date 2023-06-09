# -*- coding: utf-8 -*-
"""DL_hw2_q2 (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZN_C6ob7V5hOX9e2cpyMQfhysLNHe5uT
"""

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
start_time = time.time()
def get_data():
  colors=[transforms.Lambda(lambda x : torch.cat((x,torch.zeros_like(x),torch.zeros_like(x)),0)),
    transforms.Lambda(lambda x : torch.cat((torch.zeros_like(x),x,torch.zeros_like(x)),0)),
    transforms.Lambda(lambda x : torch.cat((torch.zeros_like(x),torch.zeros_like(x),x),0)),
    transforms.Lambda(lambda x : torch.cat((x,x,0*x),0)),
    transforms.Lambda(lambda x : torch.cat((0*x,x,x),0)),
    transforms.Lambda(lambda x : torch.cat((x,0*x,x),0)),
    transforms.Lambda(lambda x : torch.cat((0.8*x,0.2*x,0*x),0)),
    transforms.Lambda(lambda x : torch.cat((0.85*x,0.2*x,0.35*x),0)),
    transforms.Lambda(lambda x : torch.cat((0.64*x,0.4*x,0.66*x),0)),
    transforms.Lambda(lambda x : torch.cat((0.17*x,0.5*x,0.55*x),0))]
  ds_lst = []
  for i in range(10):
    mnist_dataset = dsets.MNIST(root='./data', train=True,transform=transforms.Compose([transforms.ToTensor(),colors[i],transforms.Normalize((0.5,),(0.5,))]), download=True)
    index = [idx for idx, target in enumerate(mnist_dataset.targets) if target in [i]]
    ds_lst.append(torch.utils.data.Subset(mnist_dataset,index))
  
  train_data = torch.utils.data.ConcatDataset(ds_lst)



  
  # transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: color_mnist1(x,mnist_dataset.targets[x[2]] ))])
  data_loader1 = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True)
  

  colors=[transforms.Lambda(lambda x : torch.cat((0.17*x,0.5*x,0.55*x),0)),transforms.Lambda(lambda x : torch.cat((x,torch.zeros_like(x),torch.zeros_like(x)),0)),
    transforms.Lambda(lambda x : torch.cat((torch.zeros_like(x),x,torch.zeros_like(x)),0)),
    transforms.Lambda(lambda x : torch.cat((torch.zeros_like(x),torch.zeros_like(x),x),0)),
    transforms.Lambda(lambda x : torch.cat((x,x,0*x),0)),
    transforms.Lambda(lambda x : torch.cat((0*x,x,x),0)),
    transforms.Lambda(lambda x : torch.cat((x,0*x,x),0)),
    transforms.Lambda(lambda x : torch.cat((0.8*x,0.2*x,0*x),0)),
    transforms.Lambda(lambda x : torch.cat((0.85*x,0.2*x,0.35*x),0)),
    transforms.Lambda(lambda x : torch.cat((0.64*x,0.4*x,0.66*x),0))
    ]
  ds_lst2 = []
  for i in range(10):
    mnist_dataset = dsets.MNIST(root='./data', train=True,transform=transforms.Compose([transforms.ToTensor(),colors[i],transforms.Normalize((0.5,),(0.5,))]), download=True)
    index = [idx for idx, target in enumerate(mnist_dataset.targets) if target in [i]]
    ds_lst2.append(torch.utils.data.Subset(mnist_dataset,index))
  
  
  train_data2 = torch.utils.data.ConcatDataset(ds_lst2)
  data_loader2 = torch.utils.data.DataLoader(
        train_data2,
        batch_size=64,
        shuffle=True)

  return data_loader1,data_loader2



loader1,loader2 = get_data()

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
  def __init__(self, in_,out,down=True,act=True,**Kwargs):
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_,out,**Kwargs) if down
                              else nn.ConvTranspose2d(in_,out,**Kwargs),
                              nn.InstanceNorm2d(out),
                              nn.ReLU(inplace=True) if act else nn.Identity())
  def forward(self,x):
    return self.conv(x)


class Residual(nn.Module):
  def __init__(self,channels):
    super().__init__()
    self.block = nn.Sequential(
      Block(channels,channels,kernel_size=3,padding=1),
      Block(channels,channels,act=False,kernel_size=3,padding=1)
  )

  def forward(self,x):
    return x + self.block(x)


class Generator(nn.Module):
  def __init__(self,img_channels,num_residuals = 6):
    super().__init__()
    self.initial = nn.Sequential(
        nn.Conv2d(img_channels,64,3,stride=2,padding = 1),nn.ReLU(inplace=True))
    self.down_blocks = nn.ModuleList([
         Block(64,128,kernel_size = 3, stride = 2,padding = 1),
          Block(128,256,kernel_size = 3, stride = 2)
    ])
    self.residuals = nn.Sequential(*[Residual(256) for _ in range(num_residuals)])

    self.up_blocks = nn.ModuleList([
        Block(256,128,down = False,kernel_size=3,stride=2,padding=1,output_padding=0 ),
        Block(128,64,down = False,kernel_size=3,stride=2,padding=1,output_padding=0 ),
        Block(64,32,down = False,kernel_size=3,stride=2,padding=1,output_padding=0 ),
        Block(32,16,down = False,kernel_size=3,stride=2,padding=1,output_padding=0 ),
    ])

    self.last= nn.Conv2d(16,img_channels,kernel_size=6,stride =1 , padding = 0)

  def forward(self,x):
    x= self.initial(x)
    
    # print(x.shape)
    for layer in self.down_blocks:
      x= layer(x)
      # print(x.shape)
    x = self.residuals(x)
    # print(x.shape)
    for layer in self.up_blocks:
      x= layer(x)
      # print(x.shape)
    x = torch.tanh(self.last(x))
    # print(x.shape)
    return x

model = Generator(3,6)

x = torch.randn((3,28,28))

print(model(x).shape)

from torchvision.utils import save_image
def train_model_q2():
  disc1 = Discriminator().to(device)
  disc2 = Discriminator().to(device)
  gen_1 = Generator(3,6).to(device)
  gen_2 = Generator(3,6).to(device)

  opt_disc= torch.optim.Adam(list(disc1.parameters())+list(disc2.parameters()), lr = 0.0003)
  opt_gen= torch.optim.Adam(list(gen_1.parameters())+list(gen_2.parameters()), lr = 0.0003)

  L1=nn.L1Loss()
  mse = nn.MSELoss()

  loader1,loader2 = get_data()
  num_epochs = 10

  for epoch in range(num_epochs):
    # loop = tqdm.tqdm(zip(loader1,loader2),leave = True)
    for j in range(1):
      batch_loss = []
      acc=[]
      loop = tqdm.tqdm(zip(loader1,loader2),leave = True)
      for i, data in enumerate(loop):
        x1,_,x2,_ = data[0][0],data[0][1],data[1][0],data[1][1]
        x1 = x1.to(device)
        x2 = x2.to(device)

        fake_x2 = gen_2(x1)
        D_x2_real = disc2(x2)
        D_x2_fake = disc2(fake_x2.detach())
        D_x2_real_loss = mse(D_x2_real,torch.ones_like(D_x2_real))
        D_x2_fake_loss = mse(D_x2_fake,torch.zeros_like(D_x2_fake))
        D_x2_loss = D_x2_real_loss + D_x2_fake_loss

        fake_x1 = gen_1(x2)
        D_x1_real = disc1(x1)
        D_x1_fake = disc1(fake_x1.detach())
        D_x1_real_loss = mse(D_x1_real,torch.ones_like(D_x1_real))
        D_x1_fake_loss = mse(D_x1_fake,torch.zeros_like(D_x1_fake))
        D_x1_loss = D_x1_real_loss + D_x1_fake_loss
        preds_real_correctx1 = torch.where(D_x1_real > 0.5 , 1.0,0)
        preds_fake_correctx1 = torch.where(D_x1_fake > 0.5 , 0,1.0)
        preds_real_correctx2 = torch.where(D_x2_real > 0.5 , 1.0,0)
        preds_fake_correctx2 = torch.where(D_x2_fake > 0.5 , 0,1.0)
        preds_total = (torch.sum(preds_real_correctx1).item() + torch.sum(preds_fake_correctx1).item()\
          +torch.sum(preds_real_correctx2).item() + torch.sum(preds_fake_correctx2).item())/256
        acc.append(preds_total)



        D_loss = (D_x1_loss+D_x2_loss)/2
        batch_loss.append(D_loss.item())
        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
      print(sum(batch_loss)/938)
      print(sum(acc)/938)
    for j in range(2):
      acc = []
      loop = tqdm.tqdm(zip(loader1,loader2),leave = True)
      for i, data in enumerate(loop):
        x1,_,x2,_ = data[0][0],data[0][1],data[1][0],data[1][1]
        x1 = x1.to(device)
        x2 = x2.to(device)
        fake_x2 = gen_2(x1)
        fake_x1 = gen_1(x2)
        D_x1_fake = disc1(fake_x1)
        D_x2_fake = disc2(fake_x2)
        loss_G_x1 = mse(D_x1_fake,torch.ones_like(D_x1_fake))
        loss_G_x2 = mse(D_x2_fake,torch.ones_like(D_x2_fake))
        fooled_x1 = torch.where(D_x1_fake>0.5, 1.0,0)
        fooled_x2 = torch.where(D_x2_fake>0.5, 1.0,0)
        acc.append((torch.mean(fooled_x1).item()+torch.mean(fooled_x2).item())/2)
        cycle_x1 = gen_1(fake_x2)
        cycle_x2 = gen_2(fake_x1)
        cycle_x1_loss = L1(x1,cycle_x1)
        cycle_x2_loss = L1(x2,cycle_x2)

        identity_x1 = gen_1(x1)
        identity_x2 = gen_2(x2)
        identity_x1_loss = L1(x1,identity_x1)
        identity_x2_loss = L1(x2,identity_x2)

        g_loss = (loss_G_x1*1 + loss_G_x2*1 + cycle_x1_loss*3  +cycle_x2_loss*3 + identity_x1_loss*1 + identity_x2_loss*1)

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        if i% 500 ==0 :
          save_image(fake_x1*0.5+0.5,"FakeX1.png")
          save_image(x1*0.5+0.5,"RealX1.png")
          save_image(fake_x2*0.5+0.5,"FakeX2.png")
          save_image(x2*0.5+0.5,"RealX2.png")
      print(sum(acc)/938)
  with open("Generator1.pkl", "wb") as f:
         pickle.dump(gen_1, f)
  with open("Discriminator1.pkl", "wb") as f:
         pickle.dump(disc1, f)
  with open("Generator2.pkl", "wb") as f:
         pickle.dump(gen_2, f)
  with open("Discriminator2.pkl", "wb") as f:
         pickle.dump(disc2, f)

train_model_q2()