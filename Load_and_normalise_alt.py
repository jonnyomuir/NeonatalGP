#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:54:16 2019

@author: kkth0282
"""


import numpy as np
import nibabel as nib
import pandas as pd
from sklearn import preprocessing



#Prep Design Matrices
age=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos_alt/Subjects_unique.csv')
age_time1=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos_alt/Subjects_Time1.csv')
age_time2=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos_alt/Subjects_Time2.csv')
age_injury=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos_alt/Subjects_Injury.csv')
age_growth=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos_alt/Subjects_Growth.csv')
age_premature=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos_alt/Subjects_Premature.csv')

age_design=np.float32(pd.DataFrame.as_matrix(age.iloc[:,2:5]));
age_mean=np.mean(age_design,axis=0)
age_std=np.std(age_design,axis=0)

age_demean=np.float32((pd.DataFrame.as_matrix(age.iloc[:,2:5])-age_mean)/age_std)
age_time1_demean=np.float32((pd.DataFrame.as_matrix(age_time1.iloc[:,2:5])-age_mean)/age_std)
age_time2_demean=np.float32((pd.DataFrame.as_matrix(age_time2.iloc[:,2:5])-age_mean)/age_std)
age_injury_demean=np.float32((pd.DataFrame.as_matrix(age_injury.iloc[:,2:5])-age_mean)/age_std)
age_growth_demean=np.float32((pd.DataFrame.as_matrix(age_growth.iloc[:,0:3]-age_mean))/age_std)
age_premature_demean=np.float32((pd.DataFrame.as_matrix(age_premature.iloc[:,0:3]-age_mean))/age_std)

#Load Intensity Images and mask
img=nib.load('/mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz')
mask=img.get_data()
mask_size=np.shape(mask)
mask_coords=np.argwhere(mask)
mask_index=np.ndarray.flatten(mask[mask>0])
vector_size=np.size(mask)
mask=np.ndarray.flatten(mask>0)
mask_size=np.shape(mask_index)[0]


#Loop load training images
data_dir='/mnt/lustre/users/kkth0282/dHCP_RAW_reg_d/'
image_data=np.zeros((np.shape(age)[0],mask_size,5),dtype=np.float32)
n_subs=np.shape(age)[0]
for i in np.arange(0, np.shape(age)[0]):
    T1=data_dir + 'NL_sub-' + age.ID[i] + "_" + age.SES[i] + '_T1w.nii.gz'
    T2=data_dir + 'NL_sub-' + age.ID[i] + "_" + age.SES[i] + '_T2w.nii.gz'
    Warp=data_dir + 'NL_sub-' + age.ID[i] + "_" + age.SES[i] + '1InverseWarp.nii.gz'
    T1_in = nib.load(T1)
    T2_in = nib.load(T2)
    Warp_in = nib.load(Warp)

    image_data[i,:,0:1]=np.float32(np.reshape(T1_in.get_data(),(vector_size,1))[mask,0:1])
    image_data[i,:,1:2]=np.float32(np.reshape(T2_in.get_data(),(vector_size,1))[mask,0:1])
    image_data[i,:,2:5]=np.float32(np.reshape(Warp_in.get_data(),(vector_size,3))[mask,:])
    print(i)


#Get averages and standard deviations
image_data_mean=np.zeros((mask_size,5),dtype=np.float32)
image_data_std=np.zeros((mask_size,5),dtype=np.float32)
for i in np.arange(0,5):
    image_data_mean[:,i:(i+1)]=np.mean(image_data[:,:,i:(i+1)],axis=0)
    image_data_std[:,i:(i+1)]=np.std(image_data[:,:,i:(i+1)],axis=0)

#and standardise
for i in np.arange(0,5):
    image_data[:,:,i:(i+1)]=np.reshape(preprocessing.scale(image_data[:,:,i],axis=0),(n_subs,mask_size,1))
    print(i)


#save data
np.savez('NormImage_alt.npz',image_data=image_data,age_demean=age_demean,age_time1_demean=age_time1_demean,age_time2_demean=age_time2_demean,age_growth_demean=age_growth_demean,age_premature_demean=age_premature_demean,age_injury_demean=age_injury_demean,mask_coords=mask_coords,image_data_mean=image_data_mean,image_data_std=image_data_std)


