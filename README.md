# NeonatalGP

Basic set of scripts to run multi-output regression using structural MRI data.
Reliant on GPy, scipy, numpy, pandas and python 3.7 or above.


Currently makes a lot of assumptions about the structure of your data (e.g. filenames and demographics are kept in a pandas friendly csv file).
Bash script is designed to split the generic 5D script into n-jobs for use on an SGE array.
If you have less tasks / outputs, then the 5D GPy and Recon scripts would need to be adapted - but the change isn't dramatic. 

Load_and_normalise reads in your nifti images and design matrices, standardises them by voxel / column and save the whole thing into a numpy compliant npz. 

GPyModel_1D_affine.py performs GPs (single output) on the scale and shear components of the affine matrix. This assumes the image being registered has already been *rigidly* registered to a template.

GPyModel_5D_alt.py performs GPs (multi output) on the T1w and T2w intensities and the 3 components of the warp field output from ants registration. Switching one method / modality out and another in is straightforward but the choice of kernel(s) may change.

Recon_alt.py reconstructs nifti images from saved numpy npz files (output of GPyModel_5D_alt.py).

Save_params.py performs a similar function to Recon_alt.py but saves the model parameters from the GPs as images (e.g. lengthscale, variance etc).
