# GPU VM

"""
I used CUDA with a google cloud vm to try to accelerate the training process. If u don't
have a GPU just delete all the .to("cuda") 's
"""

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

# USE CAUTIOUSLY! Converting tensor to array destroys grad data
def convertTo2DArray(tensor, shape):
    array = tensor.cpu().detach().numpy()
    array = np.reshape(array, shape)
    array = ((array + 1) * 127.5).astype(int)
    return array

# plots array as a greyscale image
def plotArray(array, xLabel=""):
    pyplot.imshow(array, "gray", vmin=0, vmax=255)
    pyplot.xlabel(xLabel)
    pyplot.show()

#SETUP:
data = imageProcessing.getData()
trainData = data[0]
alteredTrainData = data[1]
testData = data[2]
alteredTestData = data[3]

# Getting the nets
discriminator = networks.Discriminator(); discriminator.cuda()
generator = networks.Generator(); generator.cuda()

criterion = nn.BCELoss()
auxCriterion = nn.MSELoss()

d_lr = 1e-4
d_optimizer = optim.Adam(discriminator.parameters(), d_lr)#, momentum=0.6)

g_lr = 2e-3
g_optimizer = optim.SGD(generator.parameters(), g_lr)#, momentum=0.5)

pyplot.imshow(convertTo2DArray(alteredTrainData[0], [128, 128]), "gray", vmin = 0, vmax = 255)
pyplot.show()

def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))

def test(imgNum):
    batches = formBatches(5)
    imgBatches = batches[0]
    batchNum = int(imgNum / 5)
    for batchIndex in range(batchNum):
        g_output = generator(imgBatches[batchIndex].to("cuda"), 5, False)
        for imgIndex in range(5):
            img = torch.cat((alteredImgBatch[imgIndex][:128 * 48],
                             g_output[imgIndex], alteredImgBatch[imgIndex][128 * 80:]))
            img = convertTo2DArray(img, [128, 128])
            plotArray(img)

def formBatches(b_size):
    imgBatches = []
    alteredImgBatches = []

    for imgIndex in range(len(trainData)):
        if ((imgIndex + 1) % b_size == 0 and imgIndex != 0):
            batch = torch.stack(trainData[imgIndex - (b_size - 1):imgIndex + 1])
            alteredBatch = torch.stack(alteredTrainData[imgIndex - (b_size - 1):imgIndex + 1])

            imgBatches.append(batch)
            alteredImgBatches.append(alteredBatch)
        elif imgIndex == len(trainData) - 1:
            batch = torch.stack(trainData[imgIndex - ((imgIndex) % b_size):imgIndex + 1])
            alteredBatch = torch.stack(alteredTrainData[imgIndex - ((imgIndex) % b_size):imgIndex + 1])

            imgBatches.append(batch)
            alteredImgBatches.append(alteredBatch)

    return [imgBatches, alteredImgBatches]

# TRAINING/TESTING:

batchSize = 10
batches = formBatches(batchSize)
imgBatches = batches[0]
alteredImgBatches = batches[1]

d_fakeErrors = []
d_realErrors = []
g_errors = []

for epoch in range(60):
    print("\t\tEPOCH:", epoch)
    # print("memory allocated: %E (the lower the better)" % torch.cuda.memory_allocated(device=None))
    for batchNum in range(len(imgBatches)):
        print("\tBatchNum:", batchNum)
        imgBatch = imgBatches[batchNum].to("cuda")  # a tensor of tensors
        alteredImgBatch = alteredImgBatches[batchNum].to("cuda")  # a tensor of tensors
        d_steps = 4
        for d_iter in range(d_steps):
            d_optimizer.zero_grad()
            discriminator.zero_grad()

            g_output = generator(alteredImgBatch, batchSize, False)

            d_fakeInput = []
            for imgNum in range(batchSize):
                d_fakeInput.append(torch.cat((alteredImgBatch[imgNum][:128 * 48],
                                              g_output[imgNum], alteredImgBatch[imgNum][128 * 80:])))
            d_fakeInput = torch.stack(d_fakeInput).to("cuda")

            d_fakeOutput = discriminator(d_fakeInput, batchSize, False)

            d_fakeError = criterion(d_fakeOutput, Variable(torch.mul(torch.ones([batchSize, 1]), 0.9)).to("cuda"))
            d_fakeError.backward(retain_graph=False)

            d_realOutput = discriminator(imgBatch, batchSize, False)
            d_realError = criterion(d_realOutput, Variable(torch.mul(torch.ones([batchSize, 1]), 0.05)).to("cuda"))
            d_realError.backward(retain_graph=False)

            d_fakeErrors.append(d_fakeError.item())
            d_realErrors.append(d_realError.item())

            # error is a 0-dim tensor
            # D output is a [5, 1] tensor
            print("D} fakeError:", round(d_fakeError.item(), 3), "realError:", round(d_realError.item(), 3),
                  "lastFakeOutput:", round(d_fakeOutput[batchSize - 1][0].item(), 3),
                  "lastRealOutput:", round(d_realOutput[batchSize - 1][0].item(), 3))
            if d_fakeError < 0.35:
                print("D broken.")
                break
            d_optimizer.step()

        g_steps = 10
        for g_iter in range(g_steps):
            g_optimizer.zero_grad()
            generator.zero_grad()

            printImgs = False
            if batchNum == len(imgBatches) - 1 and g_iter == g_steps - 1 and imgNum == len(imgBatch) - 1:
                notify("Conv print!", "")
                printImgs = True

            # print(alteredImgBatch.size())
            g_output = generator(alteredImgBatch, batchSize, printImgs)

            correct_g = []
            for imgTensor in imgBatch:
                correct_g.append(imgTensor[128 * 48:128 * 80])
            correct_g = torch.stack(correct_g).to("cuda")
            g_auxError = auxCriterion(g_output, correct_g)
            # print("g_auxError:", g_auxError)
            # g_auxError.backward(retain_graph=True)

            d_fakeInput = []
            for imgNum in range(batchSize):
                d_fakeInput.append(torch.cat((alteredImgBatch[imgNum][:128 * 48],
                                              g_output[imgNum], alteredImgBatch[imgNum][128 * 80:])))
            d_fakeInput = torch.stack(d_fakeInput).to("cuda")

            if printImgs:
                print("train set:")
                for imgIndex in range(5):
                    imgTensor = convertTo2DArray(d_fakeInput[imgIndex], [128, 128])
                    plotArray(imgTensor)
                print("test set:")
                test(5)

            d_fakeOutput = discriminator(d_fakeInput, batchSize, printImgs)
            # print("d_fakeOutput size:", d_fakeOutput.size())

            g_mainError = criterion(d_fakeOutput, Variable(torch.mul(torch.ones([batchSize, 1]), 0.001)).to("cuda"))

            g_error = g_mainError + torch.mul(g_auxError, 1.2)

            g_error.backward(retain_graph=False)

            g_optimizer.step()

            g_errors.append(g_error.item())

            print("G} g_mainError:", round(g_mainError.item(), 3), "g_auxError:", round(g_auxError.item(), 3))

np.set_printoptions(threshold=sys.maxsize)

notify("Run finished!", "")

d_y = np.linspace(1, len(d_realErrors), len(d_realErrors))
g_y = np.linspace(1, len(g_errors), len(g_errors))

pyplot.scatter(d_y, np.asarray(d_realErrors));
pyplot.ylabel("d_realLoss");
pyplot.show()
pyplot.scatter(d_y, np.asarray(d_fakeErrors));
pyplot.ylabel("d_fakeLoss");
pyplot.show()
pyplot.scatter(g_y, np.asarray(g_errors));
pyplot.ylabel("g_loss");
pyplot.show()

print("TEST TIME")

test(25)