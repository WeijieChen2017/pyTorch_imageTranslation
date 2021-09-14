import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Net
from data import DatasetFromFolder

# training setting
parser = argparse.ArgumentParser(description='Use 3d Unet to translate NAC PET to CT')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--data_worker', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=813, help='random seed to use.')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--block_size', type=int, default=32, help='the block size of each input')
parser.add_argument('--stride', type=int, default=32, help='the stride in dataset')
parser.add_argument('--depth', type=int, default=2, help='the depth of unet')
parser.add_argument('--num_filters', type=int, default=16, help='the number of starting filters')
opt = parser.parse_args()
print(opt)

# basic setting
torch.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")

# set the dataset
testFolderX = "./data_train/X/test/"
testFolderY = "./data_train/Y/test/"

dataset_test = DatasetFromFolder(data_dir_X = testFolderX,
                                 data_dir_Y = testFolderY,
                                 batch_size = 1)

dataloader_train = DataLoader(dataset=dataset_test,
                              num_workers=opt.data_worker,
                              batch_size=opt.batch_size,
                              shuffle=True)

print("===> Datasets and Dataloders are set")

# build the network
# model = Net(block_size = opt.block_size,
#             num_filters = opt.num_filters,
#             num_level = opt.depth).to(device)
# model.double()
# criterion = nn.HuberLoss()
# optimizer = optim.Adam(model.parameters(), lr=opt.lr)
model.load_state_dict(torch.load("model_epoch_9.pth"))
model.eval()
print("===> The network, loss, optimizer are set")

