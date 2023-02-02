import nibabel as nib
import shutil
import os
import json
import pickle
import sys
path_root = '/home/anhnx5/work/multi_branch_2d/' 
# add path to sys.path
sys.path.append(path_root)
from main import config

target_train_2d = os.path.join(config.PATH_DATASET, 'train_2d')
ids_json_path = os.path.join(config.PATH_DATASET, 'ids.json')
ids_json = json.load(open(ids_json_path, 'r'))
# người - lần chụp - loại chụp và mask 
if os.path.exists(target_train_2d):
    #remove folder 
    shutil.rmtree(target_train_2d)
    #create new folder
    os.mkdir(target_train_2d)
count_slice = 0
for patient_index in ids_json:
    patient = ids_json[patient_index]
    for scan_index in patient:
        patient_scan = patient[scan_index]
        modalities =  patient_scan['modalities']
        masks = patient_scan['mask']
        modalities_t1_path = modalities['t1']
        modalities_t2_path = modalities['t2']
        modalities_flair_path = modalities['flair']
        modalities_pd_path = modalities['pd']
        mask_1_path = masks['mask1']
        mask_2_path = masks['mask2']
        
        # load nii data
        modalities_t1 = nib.load(modalities_t1_path).get_fdata()
        modalities_t2 = nib.load(modalities_t2_path).get_fdata()
        modalities_flair = nib.load(modalities_flair_path).get_fdata()
        modalities_pd = nib.load(modalities_pd_path).get_fdata()
        
        mask_1 = nib.load(mask_1_path).get_fdata()
        mask_2 = nib.load(mask_2_path).get_fdata()
        
        for slice in range(modalities_t1.shape[2]):
            pkl_save_1 = {'t1': modalities_t1[:,:,slice], 't2': modalities_t2[:,:,slice], \
                'flair': modalities_flair[:,:,slice], 'pd': modalities_pd[:,:,slice], 'mask': mask_1[:,:,slice]} 
            
            pkl_name_1 = 'slice_1_' + str(count_slice) + '.pkl'
            
            # save pkl file
            target_path = os.path.join(target_train_2d, pkl_name_1)
            with open(target_path, 'wb') as f:
                pickle.dump(pkl_save_1, f)
            
            pkl_save_2 = {'t1': modalities_t1[:,:,slice], 't2': modalities_t2[:,:,slice], \
                'flair': modalities_flair[:,:,slice], 'pd': modalities_pd[:,:,slice], 'mask': mask_2[:,:,slice]}
            pkl_name_2 = 'slice_2_' + str(count_slice) + '.pkl'

            # save pkl file
            target_path = os.path.join(target_train_2d, pkl_name_2)
            with open(target_path, 'wb') as f:
                pickle.dump(pkl_save_2, f)
            count_slice += 1
