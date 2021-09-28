import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Net
from data import DatasetFromFolder

# training setting

parser = argparse.ArgumentParser(description='Use 3d Unet to translate NAC PET to CT')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--data_worker', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=813, help='random seed to use.')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--block_size', type=int, default=128, help='the block size of each input')
parser.add_argument('--model_save_path', type=str, default='model_best_XYZ111_Huber_modelv2.pth')

opt = parser.parse_args()
print(opt)

# basic setting
torch.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")

# set the dataset
testX = "./data_train/RSPET/RS_011.nii.gz"
testY = "./data_train/RSCT/RS_011.nii.gz"
testSave = "./data_pred/"

fileX = nib.load(testX)
fileY = nib.load(testY)
print("Shape X: ", fileX.shape, " Shape Y: ", fileY)

print("===> Datasets and Dataloders are set")

print("Model name: ", opt.model_save_path)
criterion = nn.HuberLoss()
model = torch.load(opt.model_save_path).to(device)
model.eval()
print("===> The model {} are loaded.".format(opt.model_save_path))

# monai.inferers.sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.25, mode=BlendMode.CONSTANT, sigma_scale=0.125, padding_mode=PytorchPadMode.CONSTANT, cval=0.0, sw_device=None, device=None, *args, **kwargs)[source]¶



# epoch_loss = 0
# epoch_loss_list = []
# for iteration, batch in enumerate(dataloader_test, 1):
#     batch_x, batch_y, filename = batch[0].to(device), batch[1].to(device), batch[2]
#     sample_name = os.path.basename(filename[0][0])
#     sample_path = os.path.join(testSaveFolder, sample_name.replace("X", "pred"))
#     pred = model(batch_x)
#     np.save(sample_path, np.squeeze(pred.detach().numpy()))
#     loss = criterion(pred, batch_y)
#     epoch_loss += loss.item()
#     print("===> ({}/{}): Loss: {:.6f}".format(iteration, len(dataloader_test), loss.item()))
#     epoch_loss_list.append(loss.item())

# print("The loss is ", epoch_loss / len(dataloader_test))
# np.save("val_loss.npy", np.asarray(epoch_loss_list))

