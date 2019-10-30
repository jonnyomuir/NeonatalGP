#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:07:04 2018

@author: Jonathan
"""


import numpy as np
import GPy
import glob
import os
import dill
import time
import nibabel as nib



#Load Intensity Images and mask
img=nib.load('/mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz')
mask=img.get_data()
mask_size=np.shape(mask)
mask_coords=np.argwhere(mask)
mask_index=np.ndarray.flatten(mask[mask>0])
vector_size=np.size(mask)
mask=np.ndarray.flatten(mask>0)





#Load models
data=np.load('/mnt/lustre/users/kkth0282/dHCPModel_update/NormImage_alt.npz')
age_demean=np.float32(data['age_demean'])
age_time1_demean=np.float32(data['age_time1_demean'])
age_time2_demean=np.float32(data['age_time2_demean'])
age_injury_demean=np.float32(data['age_injury_demean'])
age_growth_demean=np.float32(data['age_growth_demean'])
age_premature_demean=np.float32(data['age_premature_demean'])

#Prep for cross validation (per voxel), 10-fold
xdim=np.shape(age_demean)[0]
n_all=xdim
n_time1=np.shape(age_time1_demean)[0]
n_time2=np.shape(age_time2_demean)[0]
n_injury=np.shape(age_injury_demean)[0]
n_growth=np.shape(age_growth_demean)[0]
n_prem=np.shape(age_premature_demean)[0]
print(n_injury)

#Get mean/std volumes
mean_vols=data['image_data_mean']
std_vols=data['image_data_std']

#First off save mean and std vols
image_size=np.hstack((mask_size, 5))
out_mean=np.zeros(image_size,dtype='float32'); out_std=np.zeros(image_size,dtype='float32')

for i in np.arange(0,5): 
    for j in np.arange(0,np.size(mask_index)):
        x=mask_coords[j,0]
        y=mask_coords[j,1]
        z=mask_coords[j,2]
        out_mean[x,y,z,i]=mean_vols[j,i]
        out_std[x,y,z,i]=std_vols[j,i]
    
#Test output    
out=['T1w', 'T2w', 'warpx', 'warpy', 'warpz']
img_ts=nib.load('/mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz')
header=img_ts.header
affine=img_ts.get_affine()

for i in np.arange(0,5):
    out_pred = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/orig_' + out[i] + '_mean.nii.gz'
    out_var = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/orig_' + out[i] + '_std.nii.gz'
    output_image = nib.Nifti1Image(out_mean[:,:,:,i:(i+1)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_pred)
    output_image = nib.Nifti1Image(out_std[:,:,:,i:(i+1)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_var)
    
    
    


#Find the CVs and loop the mapping, reshape output
cv_list = sorted(glob.glob('/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/C*npz'))

#Prep for cross validation loop (for overall image)
image_size=np.hstack((mask_size, (n_all)*5))
pred_out_all=np.zeros(image_size,dtype='float32'); var_out_all=np.zeros(image_size,dtype='float32')

for i in range(0, np.shape(cv_list)[0]):
    loaded=np.load(cv_list[i])
    pred=loaded['pred']
    var=loaded['var']
    start=loaded['start']
    finish=loaded['finish']
    print(cv_list[i])
    init=start
    for j in np.arange(start,finish):
        x=mask_coords[j,0]
        y=mask_coords[j,1]
        z=mask_coords[j,2]
        pred_out_all[x,y,z]=pred[:,(j-init)]
        var_out_all[x,y,z]=var[:,(j-init)]




#Test output    
for i in np.arange(0,5):
    out_pred = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_loocv_mean_' + out[i] + '.nii.gz'
    out_var = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_loocv_var_' + out[i] + '.nii.gz'
    output_image = nib.Nifti1Image(pred_out_all[:,:,:,(i*xdim):((i*xdim)+xdim)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_pred)
    output_image = nib.Nifti1Image(var_out_all[:,:,:,(i*xdim):((i*xdim)+xdim)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_var)
    


#Prep for cross validation loop (for repeat)
pred_out_all=None; var_out_all=None;
image_size=np.hstack((mask_size, (n_time2)*5))
pred_out_all=np.zeros(image_size,dtype='float32'); var_out_all=np.zeros(image_size,dtype='float32')

for i in range(0, np.shape(cv_list)[0]):
    loaded=np.load(cv_list[i])
    pred=loaded['pred_time2']
    var=loaded['var_time2']
    start=loaded['start']
    finish=loaded['finish']
    print(cv_list[i])
    init=start
    for j in np.arange(start,finish):
        x=mask_coords[j,0]
        y=mask_coords[j,1]
        z=mask_coords[j,2]
        pred_out_all[x,y,z]=pred[:,(j-init)]
        var_out_all[x,y,z]=var[:,(j-init)]


#Test output    
for i in np.arange(0,5):
    out_pred = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_time2_mean_' + out[i] + '.nii.gz'
    out_var = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_time2_var_' + out[i] + '.nii.gz'
    output_image = nib.Nifti1Image(pred_out_all[:,:,:,(i*n_time2):((i*n_time2)+n_time2)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_pred)
    output_image = nib.Nifti1Image(var_out_all[:,:,:,(i*n_time2):((i*n_time2)+n_time2)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_var)


#Prep for cross validation loop (for injury)
pred_out_all=None; var_out_all=None;
image_size=np.hstack((mask_size, (n_injury)*5))
pred_out_all=np.zeros(image_size,dtype='float32'); var_out_all=np.zeros(image_size,dtype='float32')

for i in range(0, np.shape(cv_list)[0]):
    loaded=np.load(cv_list[i])
    pred=loaded['pred_injury']
    var=loaded['var_injury']
    start=loaded['start']
    finish=loaded['finish']
    print(cv_list[i])
    init=start
    for j in np.arange(start,finish):
        x=mask_coords[j,0]
        y=mask_coords[j,1]
        z=mask_coords[j,2]
        pred_out_all[x,y,z]=pred[:,(j-init)]
        var_out_all[x,y,z]=var[:,(j-init)]




#Test output    
for i in np.arange(0,5):
    out_pred = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_injury_mean_' + out[i] + '.nii.gz'
    out_var = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_injury_var_' + out[i] + '.nii.gz'
    output_image = nib.Nifti1Image(pred_out_all[:,:,:,(i*n_injury):((i*n_injury)+n_injury)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_pred)
    output_image = nib.Nifti1Image(var_out_all[:,:,:,(i*n_injury):((i*n_injury)+n_injury)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_var)
    print(i)


#Prep for cross validation loop (for premature)
pred_out_all=None; var_out_all=None;
image_size=np.hstack((mask_size, (n_prem)*5))
pred_out_all=np.zeros(image_size,dtype='float32'); var_out_all=np.zeros(image_size,dtype='float32')

for i in range(0, np.shape(cv_list)[0]):
    loaded=np.load(cv_list[i])
    pred=loaded['pred_prem']
    var=loaded['var_prem']
    start=loaded['start']
    finish=loaded['finish']
    print(cv_list[i])
    init=start
    for j in np.arange(start,finish):
        x=mask_coords[j,0]
        y=mask_coords[j,1]
        z=mask_coords[j,2]
        pred_out_all[x,y,z]=pred[:,(j-init)]
        var_out_all[x,y,z]=var[:,(j-init)]


#Test output    
for i in np.arange(0,5):
    out_pred = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_prem_mean_' + out[i] + '.nii.gz'
    out_var = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_prem_var_' + out[i] + '.nii.gz'
    output_image = nib.Nifti1Image(pred_out_all[:,:,:,(i*n_prem):((i*n_prem)+n_prem)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_pred)
    output_image = nib.Nifti1Image(var_out_all[:,:,:,(i*n_prem):((i*n_prem)+n_prem)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_var)
    print(i)




##Prep for cross validation loop (for premature)
pred_out_all=None; var_out_all=None;
image_size=np.hstack((mask_size, (n_growth)*5))
pred_out_all=np.zeros(image_size,dtype='float32'); var_out_all=np.zeros(image_size,dtype='float32')

for i in range(0, np.shape(cv_list)[0]):
    loaded=np.load(cv_list[i])
    pred=loaded['pred_growth']
    var=loaded['var_growth']
    start=loaded['start']
    finish=loaded['finish']
    print(cv_list[i])
    init=start
    for j in np.arange(start,finish):
        x=mask_coords[j,0]
        y=mask_coords[j,1]
        z=mask_coords[j,2]
        pred_out_all[x,y,z]=pred[:,(j-init)]
        var_out_all[x,y,z]=var[:,(j-init)]


#Test output    
for i in np.arange(0,5):
    out_pred = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_growth_mean_' + out[i] + '.nii.gz'
    out_var = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_1_Alt/model_growth_var_' + out[i] + '.nii.gz'
    output_image = nib.Nifti1Image(pred_out_all[:,:,:,(i*n_growth):((i*n_growth)+n_growth)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_pred)
    output_image = nib.Nifti1Image(var_out_all[:,:,:,(i*n_growth):((i*n_growth)+n_growth)],affine,header,file_map=None)
    nib.loadsave.save(output_image, out_var)
    print(i)


