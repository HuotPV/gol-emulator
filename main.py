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
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os

#print('IT IS WORKINGGGGG')

class CNN_Model(Module):

    def __init__(self, numChannels=1,nlayer=2, final_act=0):  ## n channels has to be equal to batchsize ? Or I did something wrong there ...
        # call the parent constructor (but we have no parent yet ...)
        super().__init__()
        self.nlayer=nlayer
        self.final_act=final_act
        # initialize first set of CONV => RELU 
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=5,
        kernel_size=(3, 3),padding='same')
        self.relu1 = ReLU()
        self.conv2 = Conv2d(in_channels=5, out_channels=5,
        kernel_size=(3, 3),padding='same')
        self.relu2 = ReLU()
        self.conv3 = Conv2d(in_channels=5, out_channels=5,
        kernel_size=(3, 3),padding='same')
        self.relu3 = ReLU()
        self.conv4 = Conv2d(in_channels=5, out_channels=5,
        kernel_size=(3, 3),padding='same')
        self.relu4 = ReLU()
        self.conv5 = Conv2d(in_channels=5, out_channels=5,
        kernel_size=(3, 3),padding='same')
        self.relu5 = ReLU()
        self.conv6 = Conv2d(in_channels=5, out_channels=5,
        kernel_size=(3, 3),padding='same')
        self.relu6 = ReLU()

        self.conv_final = Conv2d(in_channels=5, out_channels=1,
        kernel_size=(1, 1),padding='same')

        if self.final_act == 1:
            self.final_act = ReLU()



    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv_final(x)

        if self.final_act==1:
            x = self.final_act(x)
        
        return x



class Data_raw():

    def __init__(self,Nworld,Nstep,Idim,Jdim,lag=1):
        self.path = os.getcwd() + "/inputs/"
        outFiles = os.listdir(self.path)
        self.data = np.zeros((Nworld*(Nstep-lag),Idim,Jdim),dtype='float32') # Can we give int to a conv2D ??? It only works with float ...
        self.label = np.zeros((Nworld*(Nstep-lag),Idim,Jdim),dtype='float32') # Can we give int to a conv2D ??? It only works with float ...

        print(self.data.shape)

        #for files in outFiles:
        #    world,worldName = readWorld(self.path,files)
        #    nwrld = worldName[6:9]
        #    nstep = worldName[14:17]
        #    self.data[(int(nwrld)-1)*Nstep+int(nstep)-1,:,:] = world

        for nw in range(1,Nworld+1):
            for ns in range(1,Nstep+1-lag):
                world,worldName = readWorld(self.path,'world-{:03}_step{:03}.txt'.format(nw,ns))
                self.data[(nw-1)*(Nstep-lag)+ns-1,:,:] = world

        for nw in range(1,Nworld+1):
            for ns in range(1+lag,Nstep+1):
                world,worldName = readWorld(self.path,'world-{:03}_step{:03}.txt'.format(nw,ns))
                self.label[(nw-1)*(Nstep-lag)+ns-1-lag,:,:] = world



class Data_in(Dataset):

    def __init__(self,data,label):
        #data_t = torch.from_numpy(data)
        #self.data = data_t.type(torch.LongTensor)
        self.data = data
        self.label = label
        self.transform = ToTensor()
        print(data.dtype)
        print(self.data.dtype)

    def __len__(self):
        shp = self.data.shape
        return shp[0]


    def __getitem__(self, idx):
        # How can we ensure that the "label" is actually the image following the previous one ?
        image = self.transform(self.data[idx,:,:])
        label = self.transform(self.label[idx,:,:])
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
    
    torch.manual_seed(0)
    train_model = True
    lag = 1
    nworld = 10
    epochs = 20
    nlayer = 2 # hard coded, 1 or 2 convolotiunal layers
    final_act = 0 # add a final activation layer (ReLU)


    def plotWorld(worldin,label):
        [s1,s2] = worldin.shape
        c = plt.pcolor(range(0,s2),range(0,s1),worldin,cmap='Blues')
        ax = plt.colorbar(c)
        ax.set_label('Cell status', rotation=270)
        ax.set_ticks([0,1])
        ax.set_ticklabels(['Dead','Alive'])
        plt.xlabel('X dimension')
        plt.ylabel('Y dimension')
        plt.title(label)
        plt.clim(0,1) 
        plt.savefig('world_'+label+'.png',dpi=300)
        plt.close()


    print('Reading data')
    data_in = Data_raw(nworld,50,80,80,lag=lag)


    plotWorld(data_in.data[80,:,:],'test_init')
    plotWorld(data_in.label[80,:,:],'test_model')


    print('Building model')
    model = CNN_Model(nlayer=nlayer,final_act=final_act)
    opt = Adam(model.parameters(), lr=10e-3)
    loss_fn = nn.MSELoss()

    print('Preparing data data')
    train_dt,test_dt = train_test_split(data_in.data,train_size=0.8,shuffle=False)
    train_lb,test_lb = train_test_split(data_in.label,train_size=0.8,shuffle=False)

    training_data = Data_in(train_dt,train_lb)
    test_data = Data_in(test_dt,test_lb)

    plotWorld(training_data.data[40,:,:],'test_ds_init')
    plotWorld(training_data.label[40,:,:],'test_ds_model')

    print(training_data.data.shape)

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True) ## Shuffle or not 
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)


    for idx, batch in enumerate(train_dataloader):
        print('Batch index: ', idx)
        print('Batch size: ', batch[0].size())
        print('Batch label: ', batch[1])
        data_sample = batch[0]
        label_sample = batch[1]
        break

    plotWorld(np.squeeze(data_sample.numpy()),'test_dl_init')
    plotWorld(np.squeeze(label_sample.numpy()),'test_dl_model')

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        model.train()
        train_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to('cpu'), y.to('cpu')

            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad() # reset gradients to zero (because they normally adds up?? why ??)
            loss.backward() # backpropagate (to compute the gradients ?)
            optimizer.step() # adjust model parameters from the gradients computed in the backward propagation

            train_loss+=loss.item()

        train_loss /= num_batches

        return train_loss


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
    


    test_lss = []; train_lss = []

    if train_model:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            train_loss = train(train_dataloader, model, loss_fn, opt)
            print(train_loss)

            print('Loss with test:')
            test_loss = test(test_dataloader, model, loss_fn)
            print('Loss with train:')
            train_loss = test(train_dataloader, model, loss_fn)
            test_lss.append(test_loss)
            train_lss.append(train_loss)

        torch.save(model.state_dict(), "model.pth")

        plt.plot(test_lss,'r',label='test')
        plt.plot(train_lss,'k',label='train')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('loss_nlayer{:01}_fact{:01}_nwrld{:03}_nepoch{:03}.png'.format(nlayer,final_act,nworld,epochs))
        plt.close()

    else:
        model = CNN_Model()
        model.load_state_dict(torch.load("model.pth"))


    def plotWorld(worldin,label):
        [s1,s2] = worldin.shape
        c = plt.pcolor(range(0,s2),range(0,s1),worldin,cmap='Blues')
        ax = plt.colorbar(c)
        ax.set_label('Cell status', rotation=270)
        ax.set_ticks([0,1])
        ax.set_ticklabels(['Dead','Alive'])
        plt.xlabel('X dimension')
        plt.ylabel('Y dimension')
        plt.title(label)
        plt.clim(0,1)  # identical to caxis([-4,4]) in MATLAB
        plt.savefig('model_nlayer{:01}_fact{:01}_nwrld{:03}_nepoch{:03}_world_'.format(nlayer,final_act,nworld,epochs)+label+'.png',dpi=300)
        plt.close()


    for idx, batch in enumerate(test_dataloader):
        print('Batch index: ', idx)
        print('Batch size: ', batch[0].size())
        print('Batch label: ', batch[1])
        sample_data = batch[0]
        sample_label = batch[1]
        break


    # make prediction
    with torch.no_grad():
        pred = model(sample_data)


    plotWorld(np.squeeze(sample_data.numpy()),'initial')
    plotWorld(np.squeeze(sample_label.numpy()),'modeled')
    plotWorld(np.squeeze(pred.numpy()),'emulated')

    print(pred.numpy())    




if __name__ == '__main__':
    main()