# NeonatalGP

Basic set of scripts to run multi-output regression using structural MRI data.
Reliant on GPy, scipy, numpy, pandas and python 3.7 or above.


Currently makes a lot of assumptions about the structure of your data (e.g. filenames and demographics are kept in a pandas friendly csv file).
Bash script is designed to split the generic 5D script into n-jobs for use on an SGE array.
If you have less tasks / outputs, then the 5D GPy and Recon scripts would need to be adapted - but the change isn't dramatic. 

