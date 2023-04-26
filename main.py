import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import os

class Data_in:

    def __init__(self,Nworld,Nstep,Idim,Jdim):
        self.path = os.getcwd() + "/inputs/"
        outFiles = os.listdir(self.path)
        self.data = np.zeros((Nworld,Nstep,Idim,Jdim),dtype='int')

        for files in outFiles:
            world,worldName = readWorld(self.path,files)
            nwrld = worldName[6:9]
            nstep = worldName[14:17]
            self.data[int(nwrld)-1,int(nstep)-1,:,:] = world

        print(self.data.shape)


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
    data_in = Data_in(1000,50,80,80)



if __name__ == '__main__':
    main()