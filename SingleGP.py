import numpy as np
import GPy

#start = INIT (designed for a job that's split up for a HPC)
start = 0
#finish = END
finish = 32
N_ITERS = finish - start

# Load up saved prepped data (Again aimed at data saved elsewhere)
#data = np.load('NORM_DATA')
#image_data = np.float32(data['image_data'])
#age_demean = np.float32(data['age_demean'])
#age_growth_demean = np.float32(data['age_growth_demean'])

#Assumes imaging data is already demeaned - guess that's a safe guess for you!
image_data = np.genfromtxt('/Users/Jonathan/Projects/GOSH/GPMay/DataFS/train_FLAIR.csv',delimiter=',')
age= np.genfromtxt('/Users/Jonathan/Projects/GOSH/GPMay/DataFS/Design/DesignTrain.csv',delimiter=',')
age_growth = np.genfromtxt('/Users/Jonathan/Projects/GOSH/GPMay/DataFS/Design/DesignAge.csv',delimiter=',')

age_mean=np.mean(age,axis=0)
age_std=np.std(age,axis=0)

age_demean=(age-age_mean)/age_std
age_growth_demean=(age_growth-age_mean)/age_std

# Prep for cross validation (per voxel), 10-fold
xdim = np.shape(image_data)[0]
ydim = np.shape(image_data)[1]
n_growth = np.shape(age_growth_demean)[0]


# Prep outputs - for the subsets you want to run (
pred_image_all = np.zeros(((xdim), N_ITERS), dtype='float32')
var_image_all = np.zeros(((xdim), N_ITERS), dtype='float32')
pred_image_growth = np.zeros(((n_growth), N_ITERS), dtype='float32')
var_image_growth = np.zeros(((n_growth), N_ITERS), dtype='float32')

# Prep prem params for prediction (could be a loop but lazy)
model_list = []

for voxel in range(start, finish):
    print(voxel)
    location = voxel - start

    base_T1w = image_data[:, voxel:(voxel + 1)]

    pred_image = np.zeros(((xdim), 1), dtype='float32')
    var_image = np.zeros(((xdim), 1), dtype='float32')

    random_index = np.arange(xdim)
    random_index_long = np.hstack((random_index, (random_index + xdim), (random_index + (xdim * 2)),
                                   (random_index + (xdim * 3)), (random_index + (xdim * 4))))

    # Fit and save the model from the whole data
    train_image = base_T1w
    train_age = age_demean
    k1 = GPy.kern.RBF(2, active_dims=(0, 1), lengthscale=2)
    k2 = GPy.kern.White(2, active_dims=(0, 1))
    k3 = GPy.kern.Linear(2)
    k_add = k1 + k2 + k3
    m = GPy.models.SparseGPRegression(train_age, train_image, kernel=k_add)
    m.optimize('bfgs', max_iters=100)
    model_list.append(m.param_array)
    # Predict Growth Output (synthetic data)
    pred_image_growth[:, location:(location + 1)], var_image_growth[:, location:(location + 1)] = m.predict(
        age_growth_demean)



    # Ten Fold
    for i in np.arange(0, 10):
        selection = np.arange(i, xdim, 5)
        # Make prediction indices
        test_age = age_demean[selection, :]

        # Extract out dependent variables
        train_age = np.delete(age_demean, [selection], axis=0)

        # Split imaging outputs to train and test
        test_image = base_T1w[selection, :]
        train_image = np.delete(base_T1w, [selection], axis=0)

        # Now the modelling bit
        k1 = GPy.kern.RBF(2, active_dims=(0, 1), lengthscale=2)
        k2 = GPy.kern.White(2, active_dims=(0, 1))
        k3 = GPy.kern.Linear(2)
        k_add = k1 + k2 + k3
        m = GPy.models.SparseGPRegression(train_age, train_image, kernel=k_add)
        m.optimize('bfgs', max_iters=100)
        print(i)

        pred_image[selection], var_image[selection] = m.predict(test_age)
    pred_image_all[:, location:(location + 1)] = pred_image
    var_image_all[:, location:(location + 1)] = var_image

