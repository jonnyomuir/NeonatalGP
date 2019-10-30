import SimpleITK as sitk
import numpy as np
import glob
import pandas as pd
import GPy
import transforms3d as t3d
import nibabel as nib
import os as os


#Load the Model
age=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos/AllSubjects_unique_model.csv',header=None)
age=pd.DataFrame.as_matrix(age); n_age=np.shape(age)[0]
age_mean=np.mean(age,axis=0); age_std=np.std(age,axis=0)
age=(age-age_mean)/age_std

#And holdout data
premature_age=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos/Premature_simple.csv',header=None)
premature_age=pd.DataFrame.as_matrix(premature_age); n_prem=np.shape(premature_age)[0]
premature_age=(premature_age-age_mean)/age_std
growth_age=pd.read_csv('/mnt/lustre/users/kkth0282/dHCPModel_update/Demos/Growth_simple.csv',header=None)
growth_age=pd.DataFrame.as_matrix(growth_age); n_growth=np.shape(growth_age)[0]
growth_age=(growth_age-age_mean)/age_std
n_growth=np.shape(growth_age)[0]

#Prep_output
Aff_pred=np.zeros((n_age,6))
Aff_var=np.zeros((n_age,6))
Aff_prem_pred=np.zeros((n_prem,6))
Aff_prem_var=np.zeros((n_prem,6))
Aff_growth_pred=np.zeros((n_growth,6))
Aff_growth_var=np.zeros((n_growth,6))



#Load the affines and reshape to 4*4
affine_init = np.zeros((4,4,n_age))
affine=np.zeros((12,n_age))
affine_list1 = sorted(glob.glob('/mnt/lustre/users/kkth0282/dHCPModel_update/OptiReg/CC*GenericAffine.mat'))
affine_list2 = sorted(glob.glob('/mnt/lustre/users/kkth0282/dHCPModel_update/OptiReg/CC*rigid*GenericAffine.mat'))
affine_list=sorted(list(set(affine_list1)-set(affine_list2)))

affine_file = sitk.ReadTransform(affine_list[0])
affine[:,0:1] = np.reshape((np.asarray(affine_file.GetParameters())),(12,1))
for i in np.arange(1,n_age):
    affine_file = sitk.ReadTransform(affine_list[i])
    affine[:,i:(i+1)] = np.reshape((np.asarray(affine_file.GetParameters())),(12,1))

affine=np.reshape(affine,(4,3,n_age))
affine_init[0:3,0:3,:]=affine[0:3,0:3]
#Add the translations back in though it pains me to do it...
affine_init[0:3,3,:]=affine[3,:,:]

#affine_init[3,:,:]=np.zeros((1,4,190))
affine_init[3,3,:]=np.ones(n_age)


#Split up the affines
#[T, R, Z, S] = t3d.affines.decompose(test)
#T is translation (set to zero); R is rotation (ignore)
#Z is scale (diagonal); S is shear (upper triangular)
Aff_params=np.zeros((n_age,6))
for i in np.arange(0,n_age):
    [ T, R, Aff_params[i,:3], Aff_params[i,3:] ] = t3d.affines.decompose(affine_init[:,:,i])

Aff_params_mean=np.mean(Aff_params,axis=0); Aff_params_std=np.std(Aff_params,axis=0)
Aff_params=(Aff_params-Aff_params_mean)/Aff_params_std



#The loop goes through the 3 zooms (first 3 columns) and 3 shears separately
for row in np.arange(0,6):
    train_age=age
    Aff_params_row=np.reshape(Aff_params[:,row],(n_age,1))
    
    for i in np.arange(0,5):
        selection=np.arange(i,n_age,5)
        test_age=age[selection,:]
        train_age=np.delete(age, [selection], axis=0)
        test_shear=Aff_params_row[selection]
        train_shear=np.delete(Aff_params_row, selection, axis=0)


        k1 = GPy.kern.RBF(3,active_dims=(0,1,2), lengthscale=2 )
        k2 = GPy.kern.White(3,active_dims=(0,1,2))
        k_add = k1 + k2
        m = GPy.models.GPRegression(train_age, train_shear, kernel=k_add)
        m.optimize('bfgs', max_iters=100)
        print(i)
        Aff_pred[selection,row:(row+1)], Aff_var[selection,row:(row+1)] = m.predict(test_age)

    
    #Train and predict full model
    k1 = GPy.kern.RBF(3,active_dims=(0,1,2), lengthscale=2 )
    k2 = GPy.kern.White(3,active_dims=(0,1,2))
    k3 = GPy.kern.Linear(3)
    k_add = k1 + k2 + k3
    m = GPy.models.GPRegression(age, Aff_params_row, kernel=k_add)
    m.optimize('bfgs', max_iters=100)
    Aff_growth_pred[:,row:(row+1)], Aff_growth_var[:,row:(row+1)] = m.predict(growth_age)
    Aff_prem_pred[:,row:(row+1)], Aff_prem_var[:,row:(row+1)] = m.predict(premature_age)
    
        

##Write out for loocv
#Write Predictions
Aff_pred=(Aff_pred*Aff_params_std)+Aff_params_mean
Output=np.zeros((4,4,n_age))
Output[0,0,:]=Aff_pred[:,0]
Output[1,1,:]=Aff_pred[:,1]
Output[2,2,:]=Aff_pred[:,2]
Output[0,1,:]=Aff_pred[:,3]
Output[0,2,:]=Aff_pred[:,4]
Output[1,2,:]=Aff_pred[:,5]
Output[3,3,:]=np.ones((n_age))

#The middle of the brain in template coords - Shift is the midline difference (for flipped x)
#Roughly the middle of the template
Translations_template=np.array((72,65,53))


#Output the affines
affine_list_out = [word.replace('.mat','_pred_fsl.mat') for word in affine_list]
affine_list_itk = [word.replace('.mat','_pred_itk.txt') for word in affine_list]

for i in np.arange(0,n_age):
    #Shift by difference in centre-of-gravity
    Output[1:3,3,i]=((Translations_template-np.dot(Translations_template,Output[:3,:3,i]))[1:])/2
    np.savetxt(affine_list_out[i], Output[:,:,i])
    os.system("/users/kkth0282/software/c3d/c3d_affine_tool -ref /mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz -src /mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz " +  affine_list_out[i] + " -oitk " + affine_list_itk[i])
    os.system("rm " + affine_list_out[i])
    print(i)



##Write out for growth
#Write Predictions
Aff_growth_pred=(Aff_growth_pred*Aff_params_std)+Aff_params_mean
Output=np.zeros((4,4,n_growth))
Output[0,0,:]=Aff_growth_pred[:,0]
Output[1,1,:]=Aff_growth_pred[:,1]
Output[2,2,:]=Aff_growth_pred[:,2]
Output[0,1,:]=Aff_growth_pred[:,3]
Output[0,2,:]=Aff_growth_pred[:,4]
Output[1,2,:]=Aff_growth_pred[:,5]
Output[3,3,:]=np.ones((n_growth))

#The middle of the brain in template coords - Shift is the midline difference (for flipped x)
#Roughly the middle of the template
Translations_template=np.array((72,65,53))


#Output the affines
for i in np.arange(0,n_growth):
    outname = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_3_Affine/growth' + str(i).zfill(4) + '_fsl.mat'
    print(outname)
    outname_itk = outname.replace('.mat','_pred_itk.txt')
    #Shift by difference in centre-of-gravity
    Output[1:3,3,i]=((Translations_template-np.dot(Translations_template,Output[:3,:3,i]))[1:])/2
#    Output[0:3,3,i]=np.zeros(3)
    np.savetxt(outname, Output[:,:,i])
    os.system("/users/kkth0282/software/c3d/c3d_affine_tool -ref /mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz -src /mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz " +  outname + " -oitk " + outname_itk)
    os.system("rm " + outname)
    print(i)




##Write out for growth
#Write Predictions
Aff_prem_pred=(Aff_prem_pred*Aff_params_std)+Aff_params_mean
Output=np.zeros((4,4,n_prem))
Output[0,0,:]=Aff_prem_pred[:,0]
Output[1,1,:]=Aff_prem_pred[:,1]
Output[2,2,:]=Aff_prem_pred[:,2]
Output[0,1,:]=Aff_prem_pred[:,3]
Output[0,2,:]=Aff_prem_pred[:,4]
Output[1,2,:]=Aff_prem_pred[:,5]
Output[3,3,:]=np.ones((n_prem))

#The middle of the brain in template coords - Shift is the midline difference (for flipped x)
#Roughly the middle of the template
Translations_template=np.array((72,65,53))


#Output the affines
for i in np.arange(0,n_prem):
    outname = '/mnt/lustre/users/kkth0282/dHCPModel_update/Model_3_Affine/prem' + str(i).zfill(4) + '_fsl.mat'
    print(outname)
    outname_itk = outname.replace('.mat','_pred_itk.txt')
    #Shift by difference in centre-of-gravity
    Output[1:3,3,i]=((Translations_template-np.dot(Translations_template,Output[:3,:3,i]))[1:])/2
#    Output[0:3,3,i]=np.zeros(3)
    np.savetxt(outname, Output[:,:,i])
    os.system("/users/kkth0282/software/c3d/c3d_affine_tool -ref /mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz -src /mnt/lustre/users/kkth0282/dHCPModel_update/mask_brain.nii.gz " +  outname + " -oitk " + outname_itk)
    os.system("rm " + outname)
    print(i)


