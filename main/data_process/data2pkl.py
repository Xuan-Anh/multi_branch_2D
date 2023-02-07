import nibabel as nib
import shutil
import os
import json
import pickle
import sys
import numpy as np
import statsmodels.api as sm
from scipy.signal import argrelextrema
import config

path_root = config.PATH_DATASET
# add path to sys.path
sys.path.append(path_root)
from main import config

def normalize_peak_image(vol, contrast):
    # copied from FLEXCONN
    # slightly changed to fit our implementation
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    # print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 0.00
    # print("%d peaks found." % (len(peaks)))

    # norm_vol = vol
    if contrast.lower() in ["t1", "mprage"]:
        peak = peaks[-1]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol/peak
        # norm_vol[norm_vol > 1.25] = 1.25
        # norm_vol = norm_vol/1.25
    elif contrast.lower() in ['t2', 'pd', 'flair', 'fl']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol / peak
        # norm_vol[norm_vol > 3.5] = 3.5
        # norm_vol = norm_vol / 3.5
    else:
        print("Contrast must be either t1,t2,pd, or flair. You entered %s. Returning 0." % contrast)

    # return peak, norm_vol
    norm_vol = vol / peak
    return norm_vol


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
        
        # load nii data
        modalities_t1 = nib.load(modalities_t1_path).get_fdata()
        modalities_t2 = nib.load(modalities_t2_path).get_fdata()
        modalities_flair = nib.load(modalities_flair_path).get_fdata()
        modalities_pd = nib.load(modalities_pd_path).get_fdata()

        # normalize data
        for modality in ['t1', 't2', 'flair', 'pd']:
            if modality == 't1':
                modalities_t1 = normalize_peak_image(modalities_t1, modality)
            elif modality == 't2':
                modalities_t2 = normalize_peak_image(modalities_t2, modality)
            elif modality == 'flair':
                modalities_flair = normalize_peak_image(modalities_flair, modality)
            elif modality == 'pd':
                modalities_pd = normalize_peak_image(modalities_pd, modality)

        mask_1_path = masks['mask1']
        mask_2_path = masks['mask2']
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
