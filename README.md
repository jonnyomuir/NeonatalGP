# NeonatalGP

Basic set of scripts to run multi-output regression using structural MRI data.
Reliant on GPy, scipy, numpy, pandas and python 3.7 or above.

Code reproduces work in this paper: 
O'Muircheartaigh, J., Robinson, E. C., Pietsch, M., Wolfers, T., Aljabar, P., Grande, L. C., ... & Edwards, A. D. (2020). 
Modelling brain development to detect white matter injury in term and preterm born neonates. 
Brain, 143(2), 467-479.
https://academic.oup.com/brain/article/143/2/467/5707354

Using freely available neonatal magnetic resonance imaging from the developing Human Connectome Project (dHCP)
http://www.developingconnectome.org/


Currently makes a lot of assumptions about the structure of your data (e.g. filenames and demographics are kept in a pandas friendly csv file).
Bash script is designed to split the generic 5D script into n-jobs for use on an SGE array.
If you have less tasks / outputs, then the 5D GPy and Recon scripts would need to be adapted - but the change isn't dramatic. 

Follows a 3 step structure:
1. Load_and_normalise reads in your nifti images and (comma separated) design matrices, standardises them by voxel / column and save the whole thing into a numpy compliant npz. 

2. GPyModel_1D_affine.py performs GPs (single output) on the scale and shear components of the affine matrix. This assumes the image being registered has already been *rigidly* registered to a template.
or
2. GPyModel_5D_alt.py performs GPs (multi output) on the T1w and T2w intensities and the 3 components of the warp field output from ants registration. Switching one method / modality out and another in is straightforward but the choice of kernel(s) may change.

3. Recon_alt.py reconstructs nifti images from saved numpy npz files (output of GPyModel_5D_alt.py). Again this needs to be adapted according to the number of output dimensions, predictions and size of training data.

(4. Save_params.py performs a similar function to Recon_alt.py but saves the model parameters from the GPs as images e.g. lengthscale, variance etc - lengthscale images can be particularly informative)
