import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import os
import math
import scipy.io as scio

def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]


class Data_myself(Dataset):

    def __init__(self, listroot=None, labelroot=None, shuffle=True):
        self.listroot = listroot
        self.labelroot = labelroot
        self.transform = transforms.ToTensor()
        listfile_root = self.listroot#os.path.join(self.listroot, 'train_img_label.txt')

        with open(listfile_root, 'r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)
        # self.nSamples = len(self.lines[:30]) if debug else len(self.lines)
        self.nSamples = len(self.lines)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath_labelpath = self.lines[index].rstrip()
        img_rgb, img_hsv, img_ycbcr, img_rgb_hsv_ycbcr, label = self.load_data_label(imgpath_labelpath)
        return (img_rgb, img_hsv, img_ycbcr, img_rgb_hsv_ycbcr, label)

    def load_data_label(self, imgpath):
        img_path = imgpath
        data = scio.loadmat(img_path)
        texture_flow_data_rgb = data.get('texture_flow_data_rgb')  # 取出字典里的label
        texture_flow_data_hsv = data.get('texture_flow_data_hsv')  # 取出字典里的label
        texture_flow_data_ycbcr = data.get('texture_flow_data_ycbcr')  # 取出字典里的label
        texture_flow_data_rgb_hsv_ycbcr = data.get('texture_flow_data_rgb_hsv_ycbcr')  # 取出字典里的label
        label = data.get('label')  # 取出字典里的data
        texture_flow_data_rgb = self.transform(texture_flow_data_rgb).float()
        texture_flow_data_hsv = self.transform(texture_flow_data_hsv).float()
        texture_flow_data_ycbcr = self.transform(texture_flow_data_ycbcr).float()
        texture_flow_data_rgb_hsv_ycbcr = self.transform(texture_flow_data_rgb_hsv_ycbcr).float()
        #texture_flow_data = texture_flow_data.view(-1, texture_flow_data.shape[0], texture_flow_data.shape[1], texture_flow_data.shape[2])
        return texture_flow_data_rgb, texture_flow_data_hsv, texture_flow_data_ycbcr, texture_flow_data_rgb_hsv_ycbcr, label


class AttentionNet(nn.Module):
    def __init__(self, img_channel):
        super(AttentionNet, self).__init__()
        # map attention module
        self.conv_1_attention_map = nn.Conv2d(img_channel, img_channel, 3, 1, 1)
        self.pool_1_attention_map = nn.MaxPool2d(2, stride=2)  # 96
        self.conv_2_attention_map = nn.Conv2d(img_channel, img_channel, 3, 1, 1)
        self.pool_2_attention_map = nn.MaxPool2d(2, stride=2)  # 48

        # channel attention module
        self.conv_1_attention_channel = nn.Conv2d(img_channel, img_channel, 3, 1, 1)
        self.pool_1_attention_channel = nn.MaxPool2d(4, stride=4)  # 48
        self.conv_2_attention_channel = nn.Conv2d(img_channel, img_channel, 3, 1, 1)
        self.pool_2_attention_channel = nn.MaxPool2d(4, stride=4)  # 12
        self.conv_3_attention_channel = nn.Conv2d(img_channel, img_channel, 1)
        self.pool_3_attention_channel = nn.MaxPool2d(4, stride=4)  # 3
        self.fc_attention_channel = nn.Linear(3 * 3 * img_channel, img_channel)

        # feature extraction module
        self.conv_1_1 = nn.Conv2d(img_channel, 128, 3, 1, 1)
        self.pool_1_1 = nn.MaxPool2d(2, stride=2)  # 96
        self.conv_2_1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool_2_1 = nn.MaxPool2d(2, stride=2)  # 48
        self.conv_3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool_3_1 = nn.MaxPool2d(2, stride=2)  # 24
        self.conv_4_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pool_4_1 = nn.MaxPool2d(2, stride=2)  # 12
        self.conv_5_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.pool_5_1 = nn.MaxPool2d(2, stride=2)  # 6
        self.conv_6_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pool_6_1 = nn.MaxPool2d(2, stride=2)  # 3
        self.fc = nn.Linear(3 * 3 * 512, 2)

    def forward(self, img):
        # map attention module
        img_conv_1_attention_map = nn.functional.relu(self.conv_1_attention_map(img))
        img_pool_1_attention_map = self.pool_1_attention_map(img_conv_1_attention_map)
        img_conv_2_attention_map = nn.functional.relu(self.conv_2_attention_map(img_pool_1_attention_map))
        img_pool_2_attention_map = self.pool_2_attention_map(img_conv_2_attention_map)
        img_attention_weight_map = nn.functional.interpolate(img_pool_2_attention_map, scale_factor=4, mode='bilinear', align_corners=True)
        img_attention_map = img.mul(img_attention_weight_map)

        # channel attention module
        img_conv_1_attention_channel = nn.functional.relu(self.conv_1_attention_channel(img_attention_map))
        img_pool_1_attention_channel = self.pool_1_attention_channel(img_conv_1_attention_channel)
        img_conv_2_attention_channel = nn.functional.relu(self.conv_2_attention_channel(img_pool_1_attention_channel))
        img_pool_2_attention_channel = self.pool_2_attention_channel(img_conv_2_attention_channel)
        img_conv_3_attention_channel = nn.functional.relu(self.conv_3_attention_channel(img_pool_2_attention_channel))
        img_pool_3_attention_channel = self.pool_3_attention_channel(img_conv_3_attention_channel)
        img_pool_3_attention_channel = img_pool_3_attention_channel.view(img_pool_3_attention_channel.size(0), -1)
        img_fc_attention_channel = self.fc_attention_channel(img_pool_3_attention_channel)
        img_fc_attention_channel = nn.functional.softmax(img_fc_attention_channel, dim=1)
        img_fc_attention_channel = img_fc_attention_channel.view(img_fc_attention_channel.size(0), img_fc_attention_channel.size(1), 1, 1)
        img_attention_channel = img_attention_map * img_fc_attention_channel * img_attention_map.shape[1]

        # Conv layer - 1
        img_conv_1_1 = nn.functional.relu(self.conv_1_1(img_attention_channel))
        img_pool_1_1 = self.pool_1_1(img_conv_1_1)
        # Conv layer - 2
        img_conv_2_1 = nn.functional.relu(self.conv_2_1(img_pool_1_1))
        img_pool_2_1 = self.pool_2_1(img_conv_2_1)
        # Conv layer - 3
        img_conv_3_1 = nn.functional.relu(self.conv_3_1(img_pool_2_1))
        img_pool_3_1 = self.pool_3_1(img_conv_3_1)
        # Conv layer - 4
        img_conv_4_1 = nn.functional.relu(self.conv_4_1(img_pool_3_1))
        img_pool_4_1 = self.pool_4_1(img_conv_4_1)
        # Conv layer - 5
        img_conv_5_1 = nn.functional.relu(self.conv_5_1(img_pool_4_1))
        img_pool_5_1 = self.pool_5_1(img_conv_5_1)
        # Conv layer - 6
        img_conv_6_1 = nn.functional.relu(self.conv_6_1(img_pool_5_1))
        img_pool_6_1 = self.pool_6_1(img_conv_6_1)
        # fold feature maps
        img_pool_6_1 = img_pool_6_1.view(img_pool_6_1.size(0), -1)
        # Linear layer
        predict = self.fc(img_pool_6_1)
        return predict  # 返回

img_transforms = transforms.ToTensor()
# load image and label
batch_size = 1
epoch_num = 50

'''
# printed attack
#############################rgb#########################################
# set network
net = AttentionNet(img_channel=45)
# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
criterion_CrossEntropyLoss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model_save_path = "./results/model_shallow_cnn_map_channel_attention_overall_rgb"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

list_path = "./data/MatList/train/train_deep_flow_overall.txt"

train_loss = open(model_save_path + "/train_loss.txt", "w")
for epoch in range(epoch_num):

    train_data = Data_myself(listroot=list_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    b_counter = 0
    for batch_image_rgb, batch_image_hsv, batch_image_ycbcr, batch_image_rgb_hsv_ycbcr, batch_label in train_loader:
        b_counter = b_counter + 1
        # zero the parameter gradients
        optimizer.zero_grad()
        ################################################################################
        # two classification
        # forward + backward + optimize
        batch_image_rgb = batch_image_rgb.to(device)
        batch_label = batch_label.type(torch.LongTensor)
        batch_label = batch_label.view(1)
        batch_label = batch_label.to(device)
        outputs = net(batch_image_rgb)
        # cross entropy loss
        CrossEntropyLoss = criterion_CrossEntropyLoss(outputs, batch_label)

        # update parameters
        loss = CrossEntropyLoss
        loss.backward()
        optimizer.step()

        # print(loss)
        running_crossentropy = CrossEntropyLoss.item()

        print('Printed Attack [%d, %d] rgb crossentropy-all: %.7f' %
              (epoch + 1, b_counter, running_crossentropy))

        train_loss.write("epoch_" + str(epoch) + "_batch_index_" + str(b_counter) + " " + str(running_crossentropy) + "\n")

    torch.save(net.state_dict(), (model_save_path + "/" + "net_epoch_" + str(epoch) + ".pkl"))

train_loss.close()
print('Finished Training')
'''
'''
#############################hsv#########################################
# set network
net = AttentionNet(img_channel=45)
# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
criterion_CrossEntropyLoss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model_save_path = "./results/model_shallow_cnn_map_channel_attention_overall_hsv"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

list_path = "./data/MatList/train/train_deep_flow_overall.txt"

train_loss = open(model_save_path + "/train_loss.txt", "w")
for epoch in range(epoch_num):

    train_data = Data_myself(listroot=list_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    b_counter = 0
    for batch_image_rgb, batch_image_hsv, batch_image_ycbcr, batch_image_rgb_hsv_ycbcr, batch_label in train_loader:
        b_counter = b_counter + 1
        # zero the parameter gradients
        optimizer.zero_grad()
        ################################################################################
        # two classification
        # forward + backward + optimize
        batch_image_hsv = batch_image_hsv.to(device)
        batch_label = batch_label.type(torch.LongTensor)
        batch_label = batch_label.view(1)
        batch_label = batch_label.to(device)
        outputs = net(batch_image_hsv)
        # cross entropy loss
        CrossEntropyLoss = criterion_CrossEntropyLoss(outputs, batch_label)

        # update parameters
        loss = CrossEntropyLoss
        loss.backward()
        optimizer.step()

        # print(loss)
        running_crossentropy = CrossEntropyLoss.item()

        print('Printed Attack [%d, %d] hsv crossentropy-all: %.7f' %
              (epoch + 1, b_counter, running_crossentropy))

        train_loss.write("epoch_" + str(epoch) + "_batch_index_" + str(b_counter) + " " + str(running_crossentropy) + "\n")

    torch.save(net.state_dict(), (model_save_path + "/" + "net_epoch_" + str(epoch) + ".pkl"))

train_loss.close()
print('Finished Training')
'''

#############################ycbcr#########################################
# set network
net = AttentionNet(img_channel=45)
# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
criterion_CrossEntropyLoss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model_save_path = "./results/model_shallow_cnn_map_channel_attention_overall_ycbcr"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

list_path = "./data/MatList/train/train_deep_flow_overall.txt"

train_loss = open(model_save_path + "/train_loss.txt", "w")
for epoch in range(epoch_num):

    train_data = Data_myself(listroot=list_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    b_counter = 0
    for batch_image_rgb, batch_image_hsv, batch_image_ycbcr, batch_image_rgb_hsv_ycbcr, batch_label in train_loader:
        b_counter = b_counter + 1
        # zero the parameter gradients
        optimizer.zero_grad()
        ################################################################################
        # two classification
        # forward + backward + optimize
        batch_image_ycbcr = batch_image_ycbcr.to(device)
        batch_label = batch_label.type(torch.LongTensor)
        batch_label = batch_label.view(1)
        batch_label = batch_label.to(device)
        outputs = net(batch_image_ycbcr)
        # cross entropy loss
        CrossEntropyLoss = criterion_CrossEntropyLoss(outputs, batch_label)

        # update parameters
        loss = CrossEntropyLoss
        loss.backward()
        optimizer.step()

        # print(loss)
        running_crossentropy = CrossEntropyLoss.item()

        print('Printed Attack [%d, %d] ycbcr crossentropy-all: %.7f' %
              (epoch + 1, b_counter, running_crossentropy))

        train_loss.write("epoch_" + str(epoch) + "_batch_index_" + str(b_counter) + " " + str(running_crossentropy) + "\n")

    torch.save(net.state_dict(), (model_save_path + "/" + "net_epoch_" + str(epoch) + ".pkl"))

train_loss.close()
print('Finished Training')
