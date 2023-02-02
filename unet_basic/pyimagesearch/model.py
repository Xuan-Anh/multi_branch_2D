from . import config
from torch.nn import ConvTranspose2d    
from torch.nn import Conv2d
from torch.nn import MaxPool2d, Module, ModuleList, ReLU
from torch.nn import functional as F
from torchvision.transforms import CenterCrop
import torch 

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, kernel_size=3)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, kernel_size=3)
        
    def forward(self, x):
        # apply CONV => RELU => CONV  block to the input and return it 
        return self.conv2(self.relu1(self.conv1(x)))
    
class Encoder(Module):
    def __init__(self, channels = (3, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = MaxPool2d(kernel_size=2) 
        
    def forward(self, x): 
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        
        print("create Encoder..............")
        for block in self.encBlocks:
            # apply the block to the input and append the output to the list
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        print("final of Encoder")         
        print('len list blockOutputs  = ', len(blockOutputs)) 
        # return the list of intermediate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels = (64, 32, 16 )):
        super().__init__()
        # initialize the decoder blocks and upsampling layer 
        self.channels = channels
        self.upconvs = ModuleList(
                        [ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
                         for i in range(len(channels)-1)])
        
        self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
        for  i in range(len(self.channels)-1):
            # apply the upsampling layer to the input 
            x  = self.upconvs[i](x)
            
            # crop the current features from the encoder blocks 
            # concanate them with the current upsampled features 
            # and pass the concanated output through the current 
            # decoder block 
            encFeat  = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x
    
    def crop(self, encFeatures, x):
        # grab the dimentions of the input and crop the encoder 
        # features to match the input dimentions
        (_, _, h, w) = x.shape
        encFeatures = CenterCrop((h, w))(encFeatures)
        
        # return the cropped encoder features
        return encFeatures
    
    
            
class UNet(Module):
    def __init__(self, encChannels = (3, 16, 32, 64),
                 decChannels = (64, 32, 16),
                 nbClasses  = 1, # number classes
                 retainDim  = True,
                 outSize = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        #initialize the encoder and decoder 
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        
        # initialize the regression head and store the class variavbles
        self.head = Conv2d(decChannels[-1], nbClasses, kernel_size=1)
        self.retainDim = retainDim
        self.outSize = outSize
        
    
    def forward(self, x):
        # grab the features from encoder 
        encFeatures = self.encoder(x)
        
        # pass  the encoder features through the decoder making sure that their dimentions are suited for concanate 
        # truyền feature đầu tiên vào  khối còn lại 
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        
        # [::-1] đảo ngược list
        # pass decoder through the regression head to obtain the segmentation map
        map = self.head(decFeatures)
        print("[SHAPE] shape map = ", map.shape)
        
        print('[SHAPE] shape outSize  = ', self.outSize)
        # check to see if we retain the original output dimentions and if so, then resize the output to match them 
        if self.retainDim:
            map = F.interpolate(map, self.outSize) 
        
        # return the segmentation map
        return map 
    
       
        
        
        
        