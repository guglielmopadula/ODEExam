from typing import Any
import torch.nn as nn
import meshio
import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch
import pickle
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from tqdm import trange
from sklearn.gaussian_process import GaussianProcessRegressor
points=meshio.read("data/bunny_0.ply").points
triangles=meshio.read("data/bunny_0.ply").cells_dict["triangle"]
all_points=np.zeros((600,len(points),3))
for i in trange(600):
    all_points[i]=meshio.read("data/bunny_"+str(i)+".ply").points

all_points=all_points.reshape(600,-1)
average_bunny=np.mean(all_points,axis=0)
meshio.write_points_cells("bunny_avg.stl",average_bunny.reshape(-1,3),{"triangle":triangles})

all_points_diff=all_points-average_bunny.reshape(1,-1)
latent_space=np.load("data/latent_space.npy")

gpr=GaussianProcessRegressor()
gpr.fit(latent_space[:500],all_points_diff[:500])
print(np.linalg.norm(gpr.predict(latent_space[:500])-all_points_diff[:500])/np.linalg.norm(average_bunny+all_points_diff[:500]))
print(np.linalg.norm(gpr.predict(latent_space[500:])-all_points_diff[500:])/np.linalg.norm(average_bunny+all_points_diff[500:]))

def check_gaussian(dataset, alpha, n_components=None):
    n=dataset.shape[0]
    #dataset: numpy array of size n_samples x n_features
    #alpha: float, the level of significance
    #n_components: int, the number of principal components to keep
    #Returns: True if the dataset is Gaussian, False otherwise
    if n_components is None:
        n_components = np.linalg.matrix_rank(dataset)
    mu=np.mean(dataset,axis=0)
    centered_data=dataset-mu
    barX=centered_data/np.sqrt(dataset.shape[0]-1)
    u,s,vh=randomized_svd(barX.T,n_components=n_components)
    barBplus=np.diag((1/s))@(u.T)
    Z=centered_data@barBplus.T
    eps=np.sqrt(np.log(2/alpha)/(2*n))
    flag=True
    for i in range(n_components):
        Z_i=Z[:,i]
        y=np.linspace(Z_i.min()-eps,Z_i.max()+eps,10000)
        true=scipy.stats.norm.cdf(y)
        pred=np.sum(Z_i.reshape(-1,1)<=y.reshape(1,-1),axis=0)/600
        if not np.prod(np.abs(pred-true)<eps):
            flag=False  
            plt.plot(y,pred)
            plt.plot(y,true+eps)
            plt.plot(y,true-eps)
            plt.plot(y,true)
            plt.show()
            break
    return flag

print(check_gaussian(latent_space,0.05))

with open('gpr.pkl','wb') as f:
    pickle.dump(gpr,f)