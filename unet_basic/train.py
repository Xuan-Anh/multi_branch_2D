# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

# partition the data into training and testing splits using 85% 
# of the data for training anf the remaining 15% for testing

split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)


# unpack the data split 
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:] 

# write the testing image paths to disk so that we can use them
# when evaluating/testing our model
print('[INFO] saving testing image paths...')
f = open(config.TEST_PATHS, 'w')
f.write('\n'.join(testImages))
f.close()


# define transformations
transforms = transforms.Compose([ transforms.ToPILImage(),
                                 transforms.Resize((config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)),
                                 transforms.ToTensor()])

# create train and test datasets 
trainDS = SegmentationDataset(trainImages, trainMasks, transforms=transforms)
testDS = SegmentationDataset(testImages, testMasks, transforms=transforms)
print('[INFO] trainDS has {} images'.format(len(trainDS)))
print('[INFO] testDS has {} images'.format(len(testDS)))

# create the train and test dataloaders
trainLoader  = DataLoader(trainDS, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), 
                          pin_memory = config.PIN_MEMORY) 
testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(),
                        pin_memory = config.PIN_MEMORY)

# init Unet 
unet = UNet().to(config.DEVICE)

# initializee loss finction and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr = config.INIT_LR)

# calculate steps per epoch for training and testing
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps  = len(testDS) // config.BATCH_SIZE

# initialize a dictionary to store the training and testing loss
H = {'train_loss': [], 'test_loss': []}

# loop over opechs
print('[INFO] training model...')
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    unet.train()
    
    # initialize the total trainning and validation loss 
    totalTrainLoss = 0 
    totalTestLoss  = 0
    
    # loop over training set 
    for (i, (x, y)) in enumerate(trainLoader):
        # send the input device 
        (x, y ) = (x.to(config.DEVICE), y.to(config.DEVICE))
        
        # perform a forward pass and calcualte the training loss 
        pred = unet(x)
        loss = lossFunc(pred, y)
        
        # first, zero out any previously accumulated gradients, then 
        # preform backpropafation, and then update model paramerters 
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # add the loss to the total training loss so far
        totalTrainLoss += loss
        
    # switch off autograd 
    with torch.no_grad():
        #  set the model in evaluation mode 
        unet.eval()
        
        # loop over the validiation set
        for (x,y) in testLoader:
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            
            # make predictions and calculate the validation loss
            pred = unet(x)
            totalTestLoss += lossFunc(pred, y)
            
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps
    
    # update training history
    H['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
    H['test_loss'].append(avgTestLoss.cpu().detach().numpy())
    
    # print model training and validiation information 
    print('[INFO] epoch: {}/{} - train_loss: {:.4f} - test_loss: {:.4f}'.format(e+1, config.NUM_EPOCHS, avgTrainLoss, avgTestLoss))
    
# display total time needed to perform the training 
endTime  = time.time()
print('[INFO] training completed in {:.4f} seconds'.format(endTime - startTime))


# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
# serialize the model to disk
torch.save(unet, config.MODEL_PATH)




