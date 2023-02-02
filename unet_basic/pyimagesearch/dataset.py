from torch.utils.data import Dataset
import cv2 

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, masksPaths, transforms=None):
        self.imagePaths = imagePaths
        self.masksPaths = masksPaths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        
        # load the image and convert it to RGB
        # and read the associated mask in grayscale mode 
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masksPaths[idx], cv2.IMREAD_GRAYSCALE)
        
        # check to see if the transforms object is not None
        if self.transforms is not None:
            # apply the transform to the image and mask
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        
        # return a tuple of the image and mask
        return (image, mask)
    

