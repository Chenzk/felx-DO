import os
import sys
import argparse
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
# from dataset import *
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='h01.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

logging.debug('\n')


class Discriminator(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.layer = nn.Linear(num_classes, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = x.sort(1)[0]
        out = self.layer(x)
        out = nn.Sigmoid()(out)
        return out

    def update(self, x, y_true):
        self.optimizer.zero_grad()
        y_out = self.forward(x)
        loss_d = nn.CrossEntropyLoss()(y_out, y_true)
        loss_d.backward()
        self.optimizer.step()


class ModelWrapper(nn.Module):
    ''''
    for Resnet, add the mask
    '''

    def __init__(self, model, mask=None):
        super().__init__()
        self.model = model

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.conv2_x(output)
        output = self.model.conv3_x(output)
        output = self.model.conv4_x(output)
        output = self.model.conv5_x(output)
        output = self.model.avg_pool(output)
        output = output.view(output.size(0), -1)
        # out = mask * out , mask 1 or 0
        output = (torch.sign(self.mask) * 0.5 + 0.5).repeat(output.size(0), 1) * output
        output = self.model.fc(output)
        return output


def train(epoch):
    THRESHOLD = 0.5
    model_B.train()
    count_A = 1e-6  # 抢答数
    count_B = 1e-6
    correct_A = 1e-6  # 抢答之后正确数
    correct_B = 1e-6

    mask_full = (torch.ones(1, 512)).cuda()
    mask = (torch.rand(1, 512) - 0.8).cuda()

    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        model_A.set_mask(mask)
        model_B.set_mask(mask)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        optimizer_b.zero_grad()
        out_A = model_A(images)
        out_A = F.softmax(out_A, dim=1)
        label_A = out_A.max(1)[1]
        a_ans = (label_A == labels).sum() / images.size(0) > THRESHOLD
        # top_A, _ = out_A.topk(2, 1)

        out_B = model_B(images)
        out_B = F.softmax(out_B, dim=1)

        label_B = out_B.max(1)[1]
        b_ans = (label_B == labels).sum() / images.size(0) > THRESHOLD
        # print(b_ans.size())
        # top_B, _ = out_B.topk(2, 1)
        # print(label_B.size()) [batch]
        # print(labels.size())

        model_B.set_mask(mask_full)
        out_B_full = model_B(images)
        out_B_full = F.softmax(out_B, dim=1)
        label_B_full = out_B_full.max(1)[1]

        da_logit = d_a(out_A)
        db_logit = d_b(out_B)

        da_out = da_logit.sum() / images.size(0)
        db_out = db_logit.sum() / images.size(0)

        # print(da_out,db_out)

        index = db_logit > THRESHOLD  # 仅判别要抢答的图像才会成为loss
        # print(index.size()) [batch,1]
        index = index.squeeze()  # 可能全是FALSE

        if da_out > THRESHOLD and db_out > THRESHOLD:  # 俩者均抢
            print('flag1')
            mask = mask - 0.2 * torch.rand(1, 512).cuda()
            if da_out > db_out:
                count_A += 1
                if a_ans:  # model a答对
                    correct_A += 1
                    loss = nn.CrossEntropyLoss()(out_B[index, :], labels[index])
                    loss.backward()
                    optimizer_b.step()
                    if b_ans:  # b 答对
                        d_b.update(Variable(out_B.data), torch.ones(db_logit.size(0),dtype=torch.long).cuda())
                else:  # a 答错
                    mask = mask + 0.2 * torch.rand(1, 512).cuda()
            else:
                count_B += 1
                loss = nn.CrossEntropyLoss()(out_B[index, :], labels[index])
                loss.backward()
                optimizer_b.step()
                if not b_ans:
                    if label_B_full is not label_B:
                        d_b.update(Variable(out_B.data), torch.zeros(db_logit.size(0),dtype=torch.long).cuda())
                else:  # b true
                    correct_B += 1

        if da_out > THRESHOLD and db_out < THRESHOLD:  # a 抢; b不抢
            print('flag2')
            count_A += 1
            if a_ans:  # model a答对
                correct_A += 1
                loss = nn.CrossEntropyLoss()(out_B[index, :], labels[index])
                loss.backward()
                optimizer_b.step()
                if b_ans:  # b 答对
                    d_b.update(Variable(out_B.data), torch.ones(db_logit.size(0,dtype=torch.long)).cuda())
            else:  # a 答错
                mask = mask + 0.2 * torch.rand(1, 512).cuda()

        if da_out < THRESHOLD and db_out > THRESHOLD:  # a不抢; b抢
            print('flag3')
            count_B += 1
            loss = nn.CrossEntropyLoss()(out_B[index, :], labels[index])
            loss.backward()
            optimizer_b.step()
            if not b_ans:
                if label_B_full is not label_B:
                    d_b.update(Variable(out_B.data), torch.zeros(db_logit.size(0,dtype=torch.long)).cuda())
            else:  # b true
                correct_B += 1

        if da_out < THRESHOLD and db_out < THRESHOLD:  # 俩者均不抢
            # print('flag4')
            mask = mask + 0.2 * torch.rand(1, 512).cuda()
            # print(db_logit.size())
            d_b.update(Variable(out_B.data), torch.zeros(db_logit.size(0),dtype=torch.long).cuda())

    s = f'A抢答:{count_A},num:{correct_A},acc:{float(correct_A)/(128*count_A)},B抢答:{count_B},num:{correct_B},acc:{float(correct_B)/(128*count_B)}'
    print(s)
    logging.debug(s)


def val(epoch):
    model_A.eval()
    model_B.eval()

    correct_A_1 = 0.0
    correct_B_1 = 0.0
    correct_A_5 = 0.0
    correct_B_5 = 0.0

    mask = (torch.ones(1, 512)).cuda()
    model_B.set_mask(mask)
    model_A.set_mask(mask)
    for (images, labels) in cifar100_test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        out_A = model_A(images)
        # label_A = out_A.max(1, keepdim=True)[1]
        _, pred_A = out_A.topk(5, 1, largest=True, sorted=True)
        out_B = model_B(images)
        # label_B = out_B.max(1, keepdim=True)[1]
        _, pred_B = out_B.topk(5, 1, largest=True, sorted=True)

        labels = labels.view(labels.size(0), -1).expand_as(pred_A)

        correct_A = pred_A.eq(labels).float()
        correct_B = pred_B.eq(labels).float()

        correct_A_5 += correct_A[:, :5].sum()
        correct_B_5 += correct_B[:, :5].sum()

        correct_A_1 += correct_A[:, :1].sum()
        correct_B_1 += correct_B[:, :1].sum()

    s = 'epoch:{},mask:{}Test set: MA Acc :Top1:{:.4f},Top5:{:.4f},MB Acc: Top1:{:.4f},Top5:{:.4f}'.format(
        epoch,
        model_B.mask.sum().item(),  # must 512
        correct_A_1.float() / len(cifar100_test_loader.dataset),
        correct_A_5.float() / len(cifar100_test_loader.dataset),
        correct_B_1.float() / len(cifar100_test_loader.dataset),
        correct_B_5.float() / len(cifar100_test_loader.dataset)
    )
    logging.debug(s)
    print(s)
    # torch.save(model_B.model.state_dict(), os.path.join(save_path, f'resnet18-{epoch}.pth'))


if __name__ == '__main__':
    model_path = './checkpoint/resnet18/resnet18-160-best.pth'
    # save_path = './checkpoint/resnet18_copy_t'

    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=64,
        shuffle=False
    )
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=30,
        shuffle=False
    )
    from models.resnet import resnet18

    model_A = resnet18().cuda()
    model_A.load_state_dict(torch.load(model_path))
    model_A = ModelWrapper(model_A)

    model_B = resnet18().cuda()
    model_B.load_state_dict(torch.load(model_path))
    model_B = ModelWrapper(model_B)

    d_a = Discriminator().cuda()
    d_b = Discriminator().cuda()

    optimizer_b = optim.SGD(model_B.model.parameters(), lr=1e-4)

    for epoch in range(0, 30):
        train(epoch)
        val(epoch)
        # next epoch
        torch.save(model_B.model.state_dict(), './model.pth')
        torch.save(d_b.state_dict(), './d.pth')
        model_A = resnet18().cuda()
        model_A.load_state_dict(torch.load('./model.pth'))
        model_A = ModelWrapper(model_A)
        d_a.load_state_dict(torch.load('./d.pth'))
