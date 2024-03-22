import meshio
import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
from torchdiffeq import odeint
import pickle
T=1
points=meshio.read("data/bunny_0.ply").points
avg_bunny=meshio.read("bunny_avg.stl").points

triangles=meshio.read("data/bunny_0.ply").cells_dict["triangle"]

BATCH_SIZE=1


new_latent=np.load("new_latent_space.npy")



print(new_latent.shape)

with open('gpr.pkl', 'rb') as f:
    gpr = pickle.load(f)


all_points_rec=gpr.predict(new_latent).reshape(1000,-1,3)
for i in range(1000):
    points=all_points_rec[i]+avg_bunny.reshape(-1,3)
    points=points-np.mean(points,axis=0)
    meshio.write_points_cells("data/bunny_gen_"+str(i)+".ply",points,{"triangle":triangles})

