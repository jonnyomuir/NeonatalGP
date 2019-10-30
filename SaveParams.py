#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:07:04 2018

@author: Jonathan
"""


import numpy as np
import GPy
import glob
import nibabel as nib

#np.savez('/data/project/cdb/NewNeonatal/MultiGP/RunData/CV_0331', pred=pred_image_all, var=var_image_all, start=start, finish=finish)
#Name : sparse gp
#Objective : 3102.1404976913677
#Number of Parameters : 219
#Number of Optimization Parameters : 169
#Updates : True
#Parameters:
#  sparse_gp.                             |                   value  |  constraints  |  priors
#  inducing_inputs                        |                 (50, 4)  |    {fixed}    |        
#  ICM.sum.rbf.variance                   |   6.058185868146009e-05  |      +ve      |        
#  ICM.sum.rbf.lengthscale                |       6.613815475915674  |      +ve      |        
#  ICM.sum.white.variance                 |  2.3304231004866006e-06  |      +ve      |        
#  ICM.sum.linear.variances               |      0.2795634758407185  |      +ve      |        
#  ICM.B.W                                |                  (5, 1)  |               |        
#  ICM.B.kappa                            |                    (5,)  |      +ve      |        
#  mixed_noise.Gaussian_noise_0.variance  |      0.9550345407167109  |      +ve      |        
#  mixed_noise.Gaussian_noise_1.variance  |      0.9544614314331804  |      +ve      |        
#  mixed_noise.Gaussian_noise_2.variance  |      0.9236903791016804  |      +ve      |        
#  mixed_noise.Gaussian_noise_3.variance  |      0.9529499092773532  |      +ve      |        
#  mixed_noise.Gaussian_noise_4.variance  |      0.8848780298736509  |      +ve      |    


#Load Intensity Images and mask
img=nib.load('/mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz')
mask=img.get_data()
mask_size=np.shape(mask)
mask_coords=np.argwhere(mask)
mask_index=np.ndarray.flatten(mask[mask>0])
vector_size=np.size(mask)
mask=np.ndarray.flatten(mask>0)

#Prep for cross validation loop (for overall image)
param_size=np.hstack((mask_size, 219))
param_output=np.zeros(param_size,dtype='float32')
model_output=np.zeros((np.shape(mask_index)[0],219))


bmatrix_size=np.hstack((mask_size, 25))
bmatrix_output_image=np.zeros(bmatrix_size,dtype='float32')
bmatrix_output=np.zeros((np.shape(mask_index)[0],25))


#Find the CVs and loop the mapping, reshape output
cv_list = sorted(glob.glob('/mnt/lustre/users/kkth0282/dHCPModel_update/Model_2_Scale/C*npz'))
model_filelist = sorted(glob.glob('/mnt/lustre/users/kkth0282/dHCPModel_update/Model_2_Scale/M*npz'))

for i in range(0, np.shape(cv_list)[0]):
    #Load results file
    loaded=np.load(cv_list[i])
    start=loaded['start']
    finish=loaded['finish']
    
    #Load model file
    loaded=np.load(model_filelist[i])
    #model_output[start:finish,:]=loaded['model_list'][:,200:] #Ignore the 200 inducing inputs
    model_output[start:finish,:]=loaded['model_list']
    bmatrix_output[start:finish,:]=loaded['bmatrix']
    
    
    print(i)
    init=start
    for j in np.arange(start,finish):
        x=mask_coords[j,0]
        y=mask_coords[j,1]
        z=mask_coords[j,2]
        param_output[x,y,z,:]=model_output[j,:]
        bmatrix_output_image[x,y,z,:]=bmatrix_output[j,:]
        

    



#Output model params
out=['params']
img_ts=nib.load('/mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz')
header=img_ts.header
affine=img_ts.get_affine()
header.set_data_shape((4,145,145,101,219))
outname= '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_2_Scale/highres_params_iter1.nii.gz'
outimage= nib.Nifti1Image(param_output,affine,header,file_map=None)
nib.loadsave.save(outimage, outname)


#Output model params
out=['bmatrix']
img_ts=nib.load('/mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz')
header=img_ts.header
affine=img_ts.get_affine()
header.set_data_shape((4,145,145,101,25))
outname= '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_2_Scale/highres_bmatrix_iter1.nii.gz'
outimage= nib.Nifti1Image(bmatrix_output_image,affine,header,file_map=None)
nib.loadsave.save(outimage, outname)


