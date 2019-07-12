import numpy as np
import numpy as np
import matplotlib.pyplot as pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
import gc

def getData():
    trainData = []
    alteredTrainData = []
    testData = []
    alteredTestData = []

    imgIndex = 1
    imageMax = 1000
    for filename in os.listdir("COIL-20/train"):
        img = Image.open("COIL-20/train/%s" % (filename,))
        imgTensor = torch.from_numpy(np.asarray(img))
        imgTensor = imgTensor.reshape(-1)
        imgTensor = imgTensor.float()
        imgTensor = torch.div(imgTensor, 127.5) - 1

        avgPixelValue = torch.mean(imgTensor).item()
        maxPixelValue = torch.max(imgTensor).item()
        minPixelValue = torch.min(imgTensor).item()
        divisor = max(avgPixelValue - minPixelValue, maxPixelValue - avgPixelValue)
        imgTensor = torch.div((imgTensor - avgPixelValue), divisor)
        # imgTensor = imgTensor.to("cuda")

        obj = int(filename[3])
        trainData.append(imgTensor)
        imgIndex += 1
        if imgIndex > imageMax:
            break

    for image in trainData:
        alteredImgTensor = image.clone()

        for element in range(128 * 48, 128 * 80):
            alteredImgTensor[element] = 0

        alteredTrainData.append(alteredImgTensor)

    imgIndex = 1
    for filename in os.listdir("COIL-20/test"):
        img = Image.open("COIL-20/test/%s" % (filename,))
        imgTensor = torch.from_numpy(np.asarray(img))
        imgTensor = imgTensor.reshape(-1)
        imgTensor = imgTensor.float()
        imgTensor = torch.div(imgTensor, 127.5) - 1

        avgPixelValue = torch.mean(imgTensor).item()
        maxPixelValue = torch.max(imgTensor).item()
        minPixelValue = torch.min(imgTensor).item()
        divisor = max(avgPixelValue - minPixelValue, maxPixelValue - avgPixelValue)
        imgTensor = torch.div((imgTensor - avgPixelValue), divisor)
        # imgTensor = imgTensor.to("cuda")

        obj = int(filename[3])
        testData.append(imgTensor)
        imgIndex += 1
        if imgIndex > imageMax:
            break

    for image in testData:
        alteredImgTensor = image.clone()

        for element in range(128 * 48, 128 * 80):
            alteredImgTensor[element] = 0

        alteredTestData.append(alteredImgTensor)

    return [trainData, alteredTrainData, testData, alteredTestData]