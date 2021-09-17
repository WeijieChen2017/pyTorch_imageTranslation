import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Net
from data import DatasetFromFolder

def train_a_epoch(data_loader, epoch, device, loss_batch_cnt):

    epoch_loss = np.zeros((len(data_loader)))
    loss_batch = np.zeros((loss_batch_cnt))
    for iteration, batch in enumerate(data_loader, 1): # start from 0
        batch_x, batch_y = batch[0].to(device), batch[1].to(device)
        # batch_x = torch.from_numpy(batch_x).double()
        # batch_y = torch.from_numpy(batch_y).double()
        # print(batch_x.size())
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        loss_batch[iteration % loss_batch_cnt] = loss.item()
        epoch_loss[iteration - 1] = loss.item()

        if iteration % loss_batch_cnt == 0:
            loss_mean = np.mean(loss_batch)
            loss_std = np.std(loss_batch)
            print("===> Epoch[{}]({}/{}): ".format(epoch, iteration, len(data_loader)), end='')
            print("Loss mean: {:.6f} Loss std: {:.6f}".format(loss_mean, loss_std))
            loss_batch = []

    return epoch_loss


# training setting
parser = argparse.ArgumentParser(description='Use 3d Unet to translate NAC PET to CT')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--loss_batch_cnt', type=int, default=16, help='loss display batch')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--data_worker', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=813, help='random seed to use.')
parser.add_argument('--cpu', action='store_true', help='use cuda?')
parser.add_argument('--block_size', type=int, default=64, help='the block size of each input')
parser.add_argument('--stride', type=int, default=32, help='the stride in dataset')
parser.add_argument('--depth', type=int, default=2, help='the depth of unet')
parser.add_argument('--num_filters', type=int, default=16, help='the number of starting filters')
parser.add_argument('--model_tag', type=str, default="b64s32d2f16", help='tag of the current model')
opt = parser.parse_args()
print(opt)

# basic setting
torch.manual_seed(opt.seed)
device = torch.device("cuda" if not opt.cpu else "cpu")
print("Device: ", device)

# set the dataset
trainFolderX = "./data_train/X"+str(opt.block_size)+"/train/"
trainFolderY = "./data_train/Y"+str(opt.block_size)+"/train/"
testFolderX = "./data_train/X"+str(opt.block_size)+"/test/"
testFolderY = "./data_train/Y"+str(opt.block_size)+"/test/"
valFolderX = "./data_train/X"+str(opt.block_size)+"/val/"
valFolderY = "./data_train/Y"+str(opt.block_size)+"/val/"

dataset_train = DatasetFromFolder(data_dir_X = trainFolderX,
                                  data_dir_Y = trainFolderY,
                                  batch_size = 1)
dataset_val = DatasetFromFolder(data_dir_X = valFolderX,
                                data_dir_Y = valFolderY,
                                batch_size = 1)

dataloader_train = DataLoader(dataset=dataset_train,
                              num_workers=opt.data_worker,
                              batch_size=opt.batch_size,
                              shuffle=True)

dataloader_val = DataLoader(dataset=dataset_val,
                            num_workers=opt.data_worker,
                            batch_size=opt.batch_size,
                            shuffle=True)
print("===> Datasets and Dataloders are set")

# build the network
model = Net(block_size = opt.block_size,
            num_filters = opt.num_filters,
            num_level = opt.depth,
            verbose = False).to(device)
model.double()
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
input = torch.randn(4, 1, opt.block_size, opt.block_size, opt.block_size).double().to(device)
model.summary(input)
print("===> The network, loss, optimizer are set")

# start the training

val_loss_best = 1e6
for epoch in range(opt.epochs):

    epoch_loss = train_a_epoch(dataloader_train, epoch, device, opt.loss_batch_cnt)
    epoch_mean = np.mean(epoch_loss)
    epoch_std = np.std(epoch_loss)
    np.save("epoch_Loss_{}.npy".format(epoch), epoch_std)
    print("===> Epoch {} Complete Loss, Avg: {:.6f}, Std: {:.6f}".format(epoch, epoch_mean, epoch_std))

    val_loss = np.asarray(train_a_epoch(dataloader_val, epoch, device, opt.loss_batch_cnt))
    val_mean = np.mean(val_loss)
    val_std = np.std(val_loss)
    np.save("val{}.npy".format(epoch), val_loss)
    print("===> Val {} Complete Loss, Avg: {:.6f}, Std: {:.6f}".format(epoch, val_mean, val_std))

    if val_loss < val_loss_best:
        model_save_path = "model_best_{}.pth".format(opt.model_tag)
        torch.save(model, model_save_path)
        print("Checkpoint saved to {}".format(model_save_path))

# # (N,C,D,H,W)
# input = torch.randn(4, 1, 32, 32, 32).double()
# model.summary(input)


