import matplotlib.pyplot as plt
import numpy as np

explicit=["euler_expl","mid22","heun33","rk44","ab2","amb2","dec48"]
explicit_vmap=["euler_expl_vmap","mid22_vmap","heun33_vmap","rk44_vmap", "am2_vmap","amb2_vmap","dec48_vmap","dec48_impl_vmap"]
implicit=["euler_impl","sdirk34","sdirk54","gl2","gl3","radauIIA3","lobattoIIIA3","lobattoIIIC3","dec48_impl"]

allnames=explicit+explicit_vmap+implicit

for name in allnames:
    arr=np.load("results/"+name+".npy",allow_pickle=True)[()]
    fig,ax=plt.subplots()
    ax.loglog(arr["num_timesteps"],arr["rel"])
    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("Relative Error")
    ax.set_title(name+": Timesteps vs RelError")
    fig.savefig("plots/"+name+"_se",bbox_inches='tight')
    plt.close()


fig,ax=plt.subplots()

for name in explicit_vmap:
    arr=np.load("results/"+name+".npy",allow_pickle=True)[()]
    ax.scatter(arr["times"][-1],arr["rel"][-1],label=name)

ax.set_xlabel("Execution Time per sample")
ax.set_yscale('log')
ax.set_ylabel("Rel Error")
ax.set_title(" ExTime vs RelError (n=10000)")

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.savefig("plots/vmap.png",dpi=400)
plt.close()


fig,ax=plt.subplots()

for name in explicit:
    arr=np.load("results/"+name+".npy",allow_pickle=True)[()]
    ax.scatter(arr["times"][-1],arr["rel"][-1],label=name)

ax.set_xlabel("Execution Time per sample")
ax.set_yscale('log')
ax.set_ylabel("Rel Error")
ax.set_title("ExTime vs RelError (n=10000)")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.savefig("plots/ex.png",dpi=400)
plt.close()

fig,ax=plt.subplots()

for name in implicit:
    arr=np.load("results/"+name+".npy",allow_pickle=True)[()]
    ax.scatter(arr["times"][-1],arr["rel"][-1],label=name)

ax.set_xlabel("Execution Time  per sample")
ax.set_yscale('log')
ax.set_ylabel("Rel Error")
ax.set_title("ExTime vs RelError (n=10000)")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.savefig("plots/im.png",dpi=400)
plt.close()

fig,ax=plt.subplots()

for name in explicit:
    arr=np.load("results/"+name+".npy",allow_pickle=True)[()]
    ax.scatter(arr["times"][-1],arr["rel"][-1],label=name)

fig.gca().set_prop_cycle(None)

for name in explicit_vmap:
    arr=np.load("results/"+name+".npy",allow_pickle=True)[()]
    ax.scatter(arr["times"][-1],arr["rel"][-1],marker='*',label=name)

ax.set_xlabel("Execution Time  per sample")
ax.set_yscale('log')
ax.set_ylabel("Rel Error")
ax.set_title("ExTime vs RelError (n=1000)")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.savefig("plots/evv.png",dpi=400)
plt.close()

fig,ax=plt.subplots()

for name in implicit+explicit:
    arr=np.load("results/"+name+".npy",allow_pickle=True)[()]
    ax.scatter(arr["times"][-1],arr["rel"][-1],label=name)

ax.set_xlabel("Execution Time  per sample")
ax.set_yscale('log')
ax.set_ylabel("Rel Error")
ax.set_title("ExTime vs RelError (n=10000)")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.close()
