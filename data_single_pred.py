from PIL import Image
import nibabel as nib
import numpy as np
import glob
import os

def normX(data):
    data[data<0] = 0
    data[data>6000] = 6000
    data = data / 6000
    return data

def normY(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

def denormX(data):
    return data * 6000

def denormY(data):
    return data * 4000 - 1000

folderX = "./data_train/NPR_SRC/"
folderY = "./data_train/CT_SRC/"
valRatio = 0.2
testRatio = 0.1
channelX = 1
channelY = 1
block_size = 128
stride = 64

testList = ['./data_train/X128/test/*.npy']
folder_pred_cube = "./data_pred/cube/"

print('-'*50)
print("Testing list: ", testList)
print('-'*50)

for pathX in testList:
    pathY = pathX.replace("X", "Y")
    filenameX = os.path.basename(pathX)[2:5]
    filenameY = os.path.basename(pathY)[2:5]
    dataX = np.load(pathX)
    dataY = np.load(pathY)
    dataNormX = normX(dataX)
    dataNormY = normY(dataY)

    
    nib.save(pred_file, pred_name)
    nib.save(diff_file, diff_name)

    print("&"*10)
    print(pred_name)
