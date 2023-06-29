import torch
import numpy as np
from torch import nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.optim import Adam
import os


class CNN_Model(Module):

    def __init__(self, numChannels=50):  ## n channels has to be equal to batchsize ? Or I did something wrong there ...
        # call the parent constructor (but we have no parent yet ...)
        super().__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=numChannels,
        kernel_size=(3, 3),padding='same')
        self.relu1 = ReLU()
        self.conv2 = Conv2d(in_channels=numChannels, out_channels=numChannels,
        kernel_size=(3, 3),padding='same')
        self.relu2 = ReLU()
        # initialize our softmax classifier
        self.conv3 = Conv2d(in_channels=numChannels, out_channels=numChannels,
        kernel_size=(1, 1),padding='same')

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x



class Data_raw():

    def __init__(self,Nworld,Nstep,Idim,Jdim):
        self.path = os.getcwd() + "/inputs/"
        outFiles = os.listdir(self.path)
        self.data = np.zeros((Nworld*Nstep,Idim,Jdim),dtype='int')

        for files in outFiles:
            world,worldName = readWorld(self.path,files)
            nwrld = worldName[6:9]
            nstep = worldName[14:17]
            self.data[(int(nwrld)-1)*Nstep+int(nstep)-1,:,:] = world


class Data_in(Dataset):

    def __init__(self,data):
        self.data = data

    def __len__(self):
        shp = self.data.shape
        return shp[0]


    def __getitem__(self, idx):
        # How can we ensure that the "label" is actually the image following the previous one ?
        image = self.data[idx,:,:]
        label = self.data[idx+1,:,:]
        return image, label

def readWorld(outDir,fileIn):

    worldName = fileIn[0:-4]

    with open(outDir + fileIn) as f:

        iLine = 0

        lines = f.readlines()
        cells = [int(x) for x in lines[0].split()]

        nRows = len(lines)
        nCol = len(cells)

        world = np.zeros((nRows,nCol),dtype=int)

        for line in lines:
            cells = [int(x) for x in line.split()]
            nCol = len(cells)

            world[iLine,:] = cells
            iLine += 1

    return world, worldName



def main():
    print('Reading data')
    data_in = Data_raw(399),50,80,80)

    print('Building model')
    model = CNN_Model()
    opt = Adam(model.parameters(), lr=10e-3)
    loss_fn = nn.MSELoss()

    print('Prepartin data data')
    training_data = Data_in(data_in.data[0:300,:,:])
    test_data = Data_in(data_in.data[300::,:,:])

    train_dataloader = DataLoader(training_data, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=50, shuffle=True)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to('cpu'), y.to('cpu')

            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad() # reset gradients to zero (because they normally adds up?? why ??)
            loss.backward() # backpropagate (to compute the gradients ?)
            optimizer.step() # adjust model parameters from the gradients computed in the backward propagation


    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to('cpu'), y.to('cpu')
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
    
        test_loss /= num_batches
        print(f"Avg loss: {test_loss:>8f} \n")

        return test_loss
    


    epochs = 12
    test_lss = []; train_lss = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train(train_dataloader, model, loss_fn, opt)

        test_loss = test(test_dataloader, model, loss_fn)
        train_loss = test(train_dataloader, model, loss_fn)
        test_lss.append(test_loss)
        train_lss.append(train_loss)






if __name__ == '__main__':
    main()