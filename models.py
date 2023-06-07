## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 5) 
        self.pool2 = nn.MaxPool2d(2, 2)
        self.norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3) 
        self.pool3 = nn.MaxPool2d(2, 2)
        self.norm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.norm4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(11*11*256, 3000)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(3000, 1500)
        self.do2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(1500, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = F.relu(self.norm1(self.pool1(self.conv1(x))))
        x = F.relu(self.norm2(self.pool2(self.conv2(x))))
        x = F.relu(self.norm3(self.pool3(self.conv3(x))))
        x = F.relu(self.norm4(self.pool4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)      # Flatten the inputs
        
        x = self.do1(F.relu(self.fc1(x)))
        x = self.do2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
