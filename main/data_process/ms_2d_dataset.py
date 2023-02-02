import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import random
import numpy as np
import config
class MS2DDataset(Dataset):
    def __init__(self, root_dir, transform=None, input_size = 256, type = 'train'):
        self.root_dir = root_dir
        self.transform = transform
        self.type = type
        self.input_size = input_size 
        ####
        self.file_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        data_t1 = data['t1']
        data_t2 = data['t2']
        data_flair = data['flair']
        data_pd = data['pd']
        data_mask = data['mask']
        
        data_return = {'t1': data_t1, 't2': data_t2, 'flair': data_flair, 'pd': data_pd, 'mask': data_mask}

        data_return = self.convert_size(data_return)
        
        # check image size to 256x256
        print('>>>>>>>>>>>>>>>>>>>')
        print('data_return: ', data_return['t1'].shape)
        if data_return['t1'].shape !=  (self.input_size, self.input_size):
            raise ValueError('Image size must has high and width equal 256')
        
        data_return = self.concatenate_data(data_return)
        
        # expand 1 dim for mask
        data_return['mask'] = data_return['mask'].unsqueeze(0)
        return  data_return

    def convert_size(self, data_return):
        h, w = data_return['t1'].shape 
        if self.type == 'train':
            if h > self.input_size  and w > self.input_size :
                pass
            if h < self.input_size  and w < self.input_size :
                random_pad_left = random.randint(0, 256 - self.input_size)
                random_pad_top = random.randint(0, 256 - self.input_size)

                for key in data_return.keys():   
                    data_return[key]  = np.pad(data_return[key]  , ((0,0), (random_pad_left, self.input_size - w - random_pad_left)), 'constant', constant_values = 0)
                    # pad in top image
                    data_return[key]  = np.pad(data_return[key]  , ((random_pad_top, self.input_size - h - random_pad_top), (0,0)), 'constant', constant_values = 0)
            
            data_return = {key: torch.from_numpy(data_return[key]) for key in data_return.keys()}

        return data_return
        
    def concatenate_data(self, data_return):
        data = torch.stack([data_return['t1'], data_return['t2'], data_return['flair']])
        mask = data_return['mask']
        data_return = {'data': data, 'mask': mask}
        return  data_return
    
if __name__ == '__main__':
    dataset = MS2DDataset(root_dir = config.TRAIN_PATH, type = 'train')
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 1)
    print(' len dataset: ', len(dataset))
    for i, data in enumerate(dataloader):
        print('data: ', data['data'].shape)
        print('mask: ', data['mask'].shape)
        break
    