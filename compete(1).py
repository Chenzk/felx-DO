import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import sys

sys.path.append('./models')
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import argparse
from densenet import DenseNet121
from resnet import ResNet18
from vgg import VGG
from acc import validation
import warnings
warnings.filterwarnings("ignore")
from train_models import get_cifar10


class D_net(nn.Module):
    def __init__(self, input_size=10, out_size=2):
        super(D_net, self).__init__()
        self.layer1 = nn.Linear(input_size, 5)
        self.layer2 = nn.Linear(5, out_size)

    def forward(self, logits):
        out = F.relu(self.layer1(logits))
        out = self.layer2(out)
        return F.sigmoid(out)


def pre_train(model, limitation=10000):
    optimizer = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9, weight_decay=5e-4)

    model.train()
    for idx, (X, y) in enumerate(train_loader):
        if idx > limitation:
            break
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


def generate_mask(perc, img_x=32, img_y=32):
    import numpy as np
    x, y = int(perc * img_x), int(perc * img_y)
    pad_x_left, pad_y_left = (img_x - x) // 2, (img_y - y) // 2
    pad_x_right, pad_y_right = img_x - x - pad_x_left, img_y - y - pad_y_left
    mask = np.ones((x, y))
    mask = np.pad(mask, ((pad_x_left, pad_x_right), (pad_y_left, pad_y_right)), 'constant')
    return mask.astype(np.float32)


def generate_masks():
    m1 = generate_mask(0.3)
    m2 = generate_mask(0.6)
    m3 = generate_mask(1.0)
    mask1 = torch.from_numpy(m1).reshape(1, 1, 32, 32).repeat(1, 3, 1, 1)
    mask2 = torch.from_numpy(m2).reshape(1, 1, 32, 32).repeat(1, 3, 1, 1)
    mask3 = torch.from_numpy(m3).reshape(1, 1, 32, 32).repeat(1, 3, 1, 1)
    return [mask1, mask2, mask3]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    train_loader, test_loader = get_cifar10(1, 1)
    criterion = nn.CrossEntropyLoss()

    A = ResNet18().to(device)
    B = ResNet18().to(device)
    D = D_net().to(device)

    load_A = True
    if load_A:
        A.load_state_dict(torch.load('../model_ckpt/ResNet18-C10.pth', map_location='cpu'))
    else:
        pre_train(A, limitation=40000)
        torch.save(A.state_dict(), 'A.pth')

    print('initial A is done')
    # validation(A, device, test_loader)#acc:0.9139

    B.load_state_dict(A.state_dict())

    opt_A = optim.SGD(A.parameters(), lr=2e-2, momentum=0.9, weight_decay=5e-4)
    opt_B = optim.SGD(B.parameters(), lr=2e-2, momentum=0.9, weight_decay=5e-4)
    opt_D = optim.SGD(D.parameters(), lr=2e-2, momentum=0.9, weight_decay=5e-4)

    masks = generate_masks()
    print('begin compete')

    counts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # record the statement times
    for idx, (X, y) in enumerate(train_loader):
        mask_idx = 0
        # print(idx)
        while (mask_idx <= 2):
            mask = masks[mask_idx]
            X, y = mask * X.to(device), y.to(device)
            out_A = F.softmax(A(X))
            out_B = F.softmax(B(X))
            # print(f'out_A.size():{out_A.size()}')#torch.Size([10])
            l = [torch.sort(out_A)[0], torch.sort(out_B)[0]]
            # print(f'l:{l}')
            # l: [tensor([0.0604, 0.0689, 0.0764, 0.0877, 0.0896, 0.0920, 0.0924, 0.1257, 0.1487,
            #             0.1580], grad_fn= < SortBackward >), tensor(
            #     [0.0604, 0.0689, 0.0764, 0.0877, 0.0896, 0.0920, 0.0924, 0.1257, 0.1487,
            #      0.1580], grad_fn= < SortBackward >)]
            in_D = torch.stack(l).squeeze()
            opt_D.zero_grad()
            # print(in_D.size()) #(2,10)
            out_D = D(in_D)
            pred_D = out_D.max(1)[1]
            # print(out_D,pred_D)
            # theshold = 0.5
            pred_A = out_A.max(1)[1]
            pred_B = out_B.max(1)[1]

            if pred_D[0] >= pred_D[1]:  # A guess [1,0] or [1,1]

                if pred_A == y:  # A correct
                    if pred_B == y:  # B also correct
                        counts[0] += 1
                        # D_target = torch.tensor([1, 1]).to(device)
                        # loss_D = criterion(out_D, D_target)
                        # loss_D.backward()
                        # opt_D.step()
                        # update B
                        opt_B.zero_grad()
                        out_B = B(X)
                        loss_B = criterion(out_B, y)
                        loss_B.backward()
                        opt_B.step()
                    else:  # B fault
                        counts[1] += 1
                        # update B
                        opt_B.zero_grad()
                        out_B = B(X).squeeze()
                        target_B = torch.tensor(out_B.data).to(device)
                        target_B[pred_B] = 0
                        # print(f'compare:{target_B,out_B}')  # here the slot is different
                        loss_B = nn.MSELoss()(out_B, target_B)
                        loss_B.backward()
                        opt_B.step()
                else:
                    if pred_B == y:
                        counts[2] += 1
                        opt_B.zero_grad()
                        out_B = B(X)
                        loss_B = criterion(out_B, y)
                        loss_B.backward()
                        opt_B.step()

                        # update D
                        D_target = torch.tensor([0, 1]).to(device)
                        loss_D = criterion(out_D, D_target)
                        loss_D.backward()
                        opt_D.step()

                    else:
                        counts[3] += 1
                        opt_B.zero_grad()
                        out_B = B(X).squeeze()
                        target_B = torch.tensor(out_B.data).to(device)
                        target_B[pred_B] = 0
                        loss_B = nn.MSELoss()(out_B, target_B)
                        loss_B.backward()
                        opt_B.step()

                        D_target = torch.tensor([0, 0]).to(device)
                        loss_D = criterion(out_D, D_target)
                        loss_D.backward()
                        opt_D.step()
                break
            else:  # B guess
                if pred_B == y:  # B correct
                    # update B
                    counts[4] += 1
                    opt_B.zero_grad()
                    out_B = B(X)
                    loss_B = criterion(out_B, y)
                    loss_B.backward()
                    opt_B.step()
                    break
                else:
                    # update D
                    if mask_idx < 2:
                        if pred_A == y:
                            # update B
                            counts[5] += 1
                            opt_B.zero_grad()
                            out_B = B(X).squeeze()
                            target_B = torch.tensor(out_B.data).to(device)
                            target_B[pred_B] = 0
                            # print(f'compare:{target_B,out_B}')  # here the slot is different
                            loss_B = nn.MSELoss()(out_B, target_B)
                            loss_B.backward()
                            opt_B.step()

                            D_target = torch.tensor([1, 0]).to(device)
                            print(out_D,D_target)
                            loss_D = criterion(out_D, D_target)
                            loss_D.backward()
                            opt_D.step()
                            break
                        else:
                            counts[6] += 1
                            mask_idx += 1
                    else:
                        counts[7] += 1
                        opt_B.zero_grad()
                        out_B = B(X).squeeze()
                        target_B = torch.tensor(out_B.data).to(device)
                        target_B[pred_B] = 0
                        # print(f'compare:{target_B,out_B}')  # here the slot is different
                        loss_B = nn.MSELoss()(out_B, target_B)
                        loss_B.backward()
                        opt_B.step()

                        D_target = torch.tensor([0, 0]).to(device)
                        loss_D = criterion(out_D, D_target)
                        loss_D.backward()
                        opt_D.step()
                        mask_idx += 1

    print(f'val B')
    validation(B, device, test_loader)
    print(f'counts:{counts}')
