import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copy
from collections import namedtuple
import os
import random
import shutil
import time
from tqdm import tqdm
from PIL import Image


class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            self.expansion * out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(
                in_channels,
                self.expansion * out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):

        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(model.state_dict(), "init_params.pt")

    def range_test(
        self, iterator, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5
    ):

        lrs = []
        losses = []
        best_loss = float("inf")

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)

        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            # update lr
            lr_scheduler.step()

            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        # reset model to initial parameters
        model.load_state_dict(torch.load("init_params.pt"))

        return lrs, losses

    def _train_batch(self, iterator):

        self.model.train()

        self.optimizer.zero_grad()

        x, y = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred, _ = self.model(x)

        loss = self.criterion(y_pred, y)

        loss.backward()

        self.optimizer.step()

        return loss.item()


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)


class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Define a transform to normalize the data
transform = transforms.Compose(
    [
        transforms.Resize(256),  # resizing the image to 256x256 pixels
        transforms.CenterCrop(
            224
        ),  # cropping the image to 224x224 pixels about the center
        transforms.ToTensor(),  # convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # normalize the image
    ]
)

# ==================================================LOAD DATA HERE====================================
# df = pd.read_csv('./data/classifier_trainset_update.csv')
CURATED_FOLDER = "/lustre1/g/geog_pyloo/05_timemachine/_curated/c_classifier"
train_df = pd.read_parquet(os.path.join(CURATED_FOLDER, "c_train_small.parquet"))[
    ["path", "label", "data_group", "city"]
]
test_df = pd.read_parquet(os.path.join(CURATED_FOLDER, "c_test_small.parquet"))[
    ["path", "label", "data_group", "city"]
]

train_df["label"] = train_df["label"] - 1  # make sure the label starts from 0
test_df["label"] = test_df["label"] - 1

# val_df = df[df['data_group']=='val']
# Filter out the training, test, and validation sets using the set column
train_dataset = CustomImageDataset(df=train_df, img_dir="", transform=transform)
test_dataset = CustomImageDataset(df=test_df, img_dir="", transform=transform)
# val_dataset = CustomImageDataset(df = val_df, img_dir = '', transform = transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

classes = train_df[["label", "city"]].drop_duplicates()
classes = dict(zip(classes["label"], classes["city"]))
print("total number of classses: ", len(list(classes.keys())))
print("*" * 100)

# =================================== LOAD MODELS HERE =======================================
ResNetConfig = namedtuple("ResNetConfig", ["block", "n_blocks", "channels"])
resnet18_config = ResNetConfig(
    block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512]
)

resnet50_config = ResNetConfig(
    block=Bottleneck, n_blocks=[3, 4, 6, 3], channels=[64, 128, 256, 512]
)

pretrained_model = models.resnet18(pretrained=True)

# set the last layer
IN_FEATURES = pretrained_model.fc.in_features
OUTPUT_DIM = train_df["label"].nunique()
print("Number of classes:", OUTPUT_DIM)
fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pretrained_model.fc = fc
model = ResNet(resnet18_config, OUTPUT_DIM)
model.load_state_dict(pretrained_model.state_dict())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("*" * 100)

criterion = nn.CrossEntropyLoss()
# SENT TO DEVICE
# use multiple gpu
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)
criterion = criterion.to(device)

# ============== HYPER PARAMETERS ==========================
FOUND_LR = 1e-3
params = [
    {"params": model.conv1.parameters(), "lr": FOUND_LR / 10},
    {"params": model.bn1.parameters(), "lr": FOUND_LR / 10},
    {"params": model.layer1.parameters(), "lr": FOUND_LR / 8},
    {"params": model.layer2.parameters(), "lr": FOUND_LR / 6},
    {"params": model.layer3.parameters(), "lr": FOUND_LR / 4},
    {"params": model.layer4.parameters(), "lr": FOUND_LR / 2},
    {"params": model.fc.parameters()},
]

optimizer = optim.Adam(params, lr=FOUND_LR)
EPOCHS = 10
STEPS_PER_EPOCH = len(train_dataloader)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
MAX_LRS = [p["lr"] for p in optimizer.param_groups]

scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LRS, total_steps=TOTAL_STEPS)


def calculate_topk_accuracy(y_pred, y, k=5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k


def train(model, iterator, optimizer, criterion, scheduler, device):

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.train()

    for x, y in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_5


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.eval()

    with torch.no_grad():

        for x, y in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_5


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float("inf")

for epoch in tqdm(range(EPOCHS)):

    start_time = time.monotonic()

    train_loss, train_acc_1, train_acc_5 = train(
        model, train_dataloader, optimizer, criterion, scheduler, device
    )
    valid_loss, valid_acc_1, valid_acc_5 = evaluate(
        model, test_dataloader, criterion, device
    )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f"./checkpoint/model-small-{epoch}.pt")
        print("Saving model")

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(
        f"\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | "
        f"Train Acc @5: {train_acc_5*100:6.2f}%"
    )
    print(
        f"\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | "
        f"Valid Acc @5: {valid_acc_5*100:6.2f}%"
    )
