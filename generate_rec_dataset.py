import meshio
import numpy as np
import pickle

triangles=meshio.read("data/bunny_0.ply").cells_dict["triangle"]
latent_space=np.load("data/latent_space.npy")
avg_bunny=meshio.read("bunny_avg.stl").points

with open('gpr.pkl', 'rb') as f:
    gpr = pickle.load(f)

all_points_rec=gpr.predict(latent_space).reshape(600,-1,3)

for i in range(600):
    points=all_points_rec[i]+avg_bunny.reshape(-1,3)
    points=points-np.mean(points,axis=0)
    meshio.write_points_cells("data/bunny_rec_"+str(i)+".ply",points,{"triangle":triangles})
