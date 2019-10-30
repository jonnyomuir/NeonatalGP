#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:30:27 2019

@author: kkth0282
"""

import numpy as np
import GPy
import os
import time


start=INIT
finish=END
N_ITERS=finish-start


#Load up saved prepped data
data=np.load('NORM_DATA')
image_data=np.float32(data['image_data'])
age_demean=np.float32(data['age_demean'])
age_time1_demean=np.float32(data['age_time1_demean'])
age_time2_demean=np.float32(data['age_time2_demean'])
age_injury_demean=np.float32(data['age_injury_demean'])
age_growth_demean=np.float32(data['age_growth_demean'])
age_premature_demean=np.float32(data['age_premature_demean'])

mask_coords=data['mask_coords']

#Prep for cross validation (per voxel), 10-fold
xdim=np.shape(image_data)[0]
ydim=np.shape(image_data)[1]

n_time1=np.shape(age_time1_demean)[0]
n_time2=np.shape(age_time2_demean)[0]
n_injury=np.shape(age_injury_demean)[0]
n_growth=np.shape(age_growth_demean)[0]
n_premature=np.shape(age_premature_demean)[0]


#Prep outputs
pred_image_all=np.zeros(((xdim*5),N_ITERS),dtype='float32')
var_image_all=np.zeros(((xdim*5),N_ITERS),dtype='float32')
pred_image_time1=np.zeros(((n_time1*5),N_ITERS),dtype='float32')
var_image_time1=np.zeros(((n_time1*5),N_ITERS),dtype='float32')
pred_image_time2=np.zeros(((n_time2*5),N_ITERS),dtype='float32')
var_image_time2=np.zeros(((n_time2*5),N_ITERS),dtype='float32')
pred_image_injury=np.zeros(((n_injury*5),N_ITERS),dtype='float32')
var_image_injury=np.zeros(((n_injury*5),N_ITERS),dtype='float32')
pred_image_growth=np.zeros(((n_growth*5),N_ITERS),dtype='float32')
var_image_growth=np.zeros(((n_growth*5),N_ITERS),dtype='float32')
pred_image_premature=np.zeros(((n_premature*5),N_ITERS),dtype='float32')
var_image_premature=np.zeros(((n_premature*5),N_ITERS),dtype='float32')



#Prep prem params for prediction (could be a loop but lazy)
test_age=np.float32(age_demean)
test_age_T1w=np.hstack([test_age,np.zeros_like(test_age[:,0:1])])
test_age_T2w=np.hstack([test_age,np.ones_like(test_age[:,0:1])])
test_age_x=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*2])
test_age_y=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*3])
test_age_z=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*4])
test_indices_prem=np.vstack((test_age_T1w,test_age_T2w,test_age_x,test_age_y,test_age_z))
noise_dict_prem = {'output_index':test_indices_prem[:,3:].astype(int)}

test_age=np.float32(age_time1_demean)
test_age_T1w=np.hstack([test_age,np.zeros_like(test_age[:,0:1])])
test_age_T2w=np.hstack([test_age,np.ones_like(test_age[:,0:1])])
test_age_x=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*2])
test_age_y=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*3])
test_age_z=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*4])
test_indices_hold1=np.vstack((test_age_T1w,test_age_T2w,test_age_x,test_age_y,test_age_z))
noise_dict_hold1 = {'output_index':test_indices_hold1[:,3:].astype(int)}

test_age=np.float32(age_time2_demean)
test_age_T1w=np.hstack([test_age,np.zeros_like(test_age[:,0:1])])
test_age_T2w=np.hstack([test_age,np.ones_like(test_age[:,0:1])])
test_age_x=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*2])
test_age_y=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*3])
test_age_z=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*4])
test_indices_hold2=np.vstack((test_age_T1w,test_age_T2w,test_age_x,test_age_y,test_age_z))
noise_dict_hold2 = {'output_index':test_indices_hold2[:,3:].astype(int)}

test_age=np.float32(age_injury_demean)
test_age_T1w=np.hstack([test_age,np.zeros_like(test_age[:,0:1])])
test_age_T2w=np.hstack([test_age,np.ones_like(test_age[:,0:1])])
test_age_x=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*2])
test_age_y=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*3])
test_age_z=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*4])
test_indices_injury=np.vstack((test_age_T1w,test_age_T2w,test_age_x,test_age_y,test_age_z))
noise_dict_injury = {'output_index':test_indices_injury[:,3:].astype(int)}

test_age=np.float32(age_growth_demean)
test_age_T1w=np.hstack([test_age,np.zeros_like(test_age[:,0:1])])
test_age_T2w=np.hstack([test_age,np.ones_like(test_age[:,0:1])])
test_age_x=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*2])
test_age_y=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*3])
test_age_z=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*4])
test_indices_growth=np.vstack((test_age_T1w,test_age_T2w,test_age_x,test_age_y,test_age_z))
noise_dict_growth = {'output_index':test_indices_growth[:,3:].astype(int)}

test_age=np.float32(age_premature_demean)
test_age_T1w=np.hstack([test_age,np.zeros_like(test_age[:,0:1])])
test_age_T2w=np.hstack([test_age,np.ones_like(test_age[:,0:1])])
test_age_x=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*2])
test_age_y=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*3])
test_age_z=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*4])
test_indices_premature=np.vstack((test_age_T1w,test_age_T2w,test_age_x,test_age_y,test_age_z))
noise_dict_premature = {'output_index':test_indices_premature[:,3:].astype(int)}




model_list=[]
bmatrix=[]

for voxel in range(start,finish):
    print(voxel)
    location=voxel-start

    base_T1w=image_data[:,voxel:(voxel+1),0]
    base_T2w=image_data[:,voxel:(voxel+1),1]
    base_x=image_data[:,voxel:(voxel+1),2]
    base_y=image_data[:,voxel:(voxel+1),3]
    base_z=image_data[:,voxel:(voxel+1),4]
 
    pred_image=np.zeros(((xdim*5),1),dtype='float32')
    var_image=np.zeros(((xdim*5),1),dtype='float32')

    random_index=np.arange(xdim)
    random_index_long=np.hstack((random_index,(random_index+xdim),(random_index+(xdim*2)),(random_index+(xdim*3)),(random_index+(xdim*4))))
    start_time=time.time()  
   
    #Fit and save the model from the whole data
    train_image=[base_T1w, base_T2w, base_x, base_y, base_z]
    train_age=[age_demean,age_demean,age_demean,age_demean,age_demean]
    k1 = GPy.kern.RBF(3,active_dims=(0,1,2), lengthscale=2 )
    k2 = GPy.kern.White(3,active_dims=(0,1,2))
    k3 = GPy.kern.Linear(3)
    k_add = k1 + k2 + k3
    icm = GPy.util.multioutput.ICM(input_dim=3,num_outputs=5,kernel=k_add)
    m = GPy.models.SparseGPCoregionalizedRegression(train_age,train_image,kernel=icm)
    m.optimize('bfgs', max_iters=100)
    model_list.append(m.param_array)
    bmatrix.append(np.reshape(icm.B.B,(25,)))

    #Predict Holdout Data (time1)
    pred_image_time1[:,location:(location+1)],var_image_time1[:,location:(location+1)]=m.predict(test_indices_hold1,Y_metadata=noise_dict_hold1)
    #Predict Holdout Data (time1)
    pred_image_time2[:,location:(location+1)],var_image_time2[:,location:(location+1)]=m.predict(test_indices_hold2,Y_metadata=noise_dict_hold2)
    pred_image_injury[:,location:(location+1)],var_image_injury[:,location:(location+1)]=m.predict(test_indices_injury,Y_metadata=noise_dict_injury)
    #Predict Premature Output (synthetic data)
    pred_image_premature[:,location:(location+1)],var_image_premature[:,location:(location+1)]=m.predict(test_indices_premature,Y_metadata=noise_dict_premature)
    #Predict Growth Output (synthetic data)
    pred_image_growth[:,location:(location+1)],var_image_growth[:,location:(location+1)]=m.predict(test_indices_growth,Y_metadata=noise_dict_growth)
  

    #Leave one out loop to predict intensities
    for i in np.arange(0,5):
        selection=np.arange(i,xdim,5)
        #Make prediction indices
        test_age=age_demean[selection,:]
        test_age_T1w=np.hstack([test_age,np.zeros_like(test_age[:,0:1])])
        test_age_T2w=np.hstack([test_age,np.ones_like(test_age[:,0:1])])
        test_age_x=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*2])
        test_age_y=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*3])
        test_age_z=np.hstack([test_age,(np.ones_like(test_age[:,0:1]))*4])
        test_indices=np.vstack((test_age_T1w,test_age_T2w,test_age_x,test_age_y,test_age_z))
        noise_dict = {'output_index':test_indices[:,3:].astype(int)} #The four being indiced means the last column
        
        #Extract out dependent variables
        train_age=np.delete(age_demean, [selection], axis=0)
        train_age_single=train_age
        train_age=[train_age,train_age,train_age,train_age,train_age]
        
        #Split imaging outputs to train and test
        test_T1w=base_T1w[selection,:]
        test_T2w=base_T2w[selection,:]
        test_x=base_x[selection,:]
        test_y=base_y[selection,:]
        test_z=base_z[selection,:]
        
        train_T1w=np.delete(base_T1w, selection, axis=0)
        train_T2w=np.delete(base_T2w, selection, axis=0)
        train_x=np.delete(base_x, selection, axis=0)
        train_y=np.delete(base_y, selection, axis=0)
        train_z=np.delete(base_z, selection, axis=0)
        train_image=[train_T1w, train_T2w, train_x, train_y, train_z]
        
        
        #This line hurts but is just so pred_image gets filled in the right order
        target_indices=np.hstack((selection,(selection+(xdim*1)),(selection+(xdim*2)),(selection+(xdim*3)),(selection+(xdim*4))))
        
        #Now the modelling bit
        k1 = GPy.kern.RBF(3,active_dims=(0,1,2), lengthscale=2 )
        k2 = GPy.kern.White(3,active_dims=(0,1,2))
        k3 = GPy.kern.Linear(3)
        k_add = k1 + k2 + k3
        icm = GPy.util.multioutput.ICM(input_dim=3,num_outputs=5,kernel=k_add)
        m = GPy.models.SparseGPCoregionalizedRegression(train_age, train_image, kernel=icm)
        m.optimize('bfgs', max_iters=100)
        print(i)
        
        pred_image[target_indices], var_image[target_indices] = m.predict(test_indices,Y_metadata=noise_dict)
    pred_image_all[:,location:(location+1)]=pred_image
    var_image_all[:,location:(location+1)]=var_image




warpx_demean=None
warpy_demean=None
warpz_demean=None
train_T1w=None 
train_T2w=None
mask_index=None
mask=None

np.savez('PATH/CV_ORDER', pred=pred_image_all, var=var_image_all, pred_prem=pred_image_premature, pred_time1=pred_image_time1, pred_time2=pred_image_time2, pred_growth=pred_image_growth, pred_injury=pred_image_injury, var_prem=var_image_premature, var_time1=var_image_time1, var_time2=var_image_time2, var_growth=var_image_growth, var_injury=var_image_injury, start=start, finish=finish)
np.savez('PATH/MODEL_ORDER', model_list=model_list, bmatrix=bmatrix)






