
import torch
import jax
from reference_method import ContinousScore
from flax import linen as nn
import numpy as np
from functools import partial
import jax.numpy as jnp
from jaxopt import GaussNewton, BFGS
from jax import jit
import time
from tqdm import trange
from jax.scipy import optimize
model=torch.load("model.pt")
from nodepy import rk


fc0w= jnp.array(model.state_dict()["score_model.0.weight"].detach().numpy())
fc0b= jnp.array(model.state_dict()["score_model.0.bias"].detach().numpy())
fc2w= jnp.array(model.state_dict()["score_model.2.weight"].detach().numpy())
fc2b= jnp.array(model.state_dict()["score_model.2.bias"].detach().numpy())
fc4w= jnp.array(model.state_dict()["score_model.4.weight"].detach().numpy())
fc4b= jnp.array(model.state_dict()["score_model.4.bias"].detach().numpy())
fc6w= jnp.array(model.state_dict()["score_model.6.weight"].detach().numpy())
fc6b= jnp.array(model.state_dict()["score_model.6.bias"].detach().numpy())
#print(jnp.linalg.norm(fc0w)*jnp.linalg.norm(fc2w)*jnp.linalg.norm(fc4w)*jnp.linalg.norm(fc6w))


def score(x,t):
   x=jnp.concatenate((x,jnp.array([t])),axis=0).reshape(1,-1)
   x=x@fc0w.T+fc0b
   x=jax.nn.log_sigmoid(x)
   x=x@fc2w.T+fc2b
   x=jax.nn.log_sigmoid(x)
   x=x@fc4w.T+fc4b
   x=jax.nn.log_sigmoid(x)
   x=x@fc6w.T+fc6b
   return x.reshape(-1)

a=0.1
b=20
def beta(t):
    return a+(b-a)*t

def int_beta(self,t):
    return (self.a+1/2*(b-a)*t)*t


def f(x,t):
    return -1/2*beta(t)*x

def g(x,t):
    return beta(t)**0.5

def ode(x,t):
    t=1-t
    fxt =f(x,t)
    gt= g(x,t)
    return -(fxt-1/2*gt**2*score(x,t))

jac=jax.jacobian(ode)

def dec48_inner_loop(x,u_a,t,delta_t):
    for k in range(8):
        rhs0=ode(x,t+beta5[0]*delta_t)
        rhs1=ode(u_a[:,0],t+beta5[1]*delta_t)
        rhs2=ode(u_a[:,1],t+beta5[2]*delta_t)
        rhs3=ode(u_a[:,2],t+beta5[3]*delta_t)
        rhs4=ode(u_a[:,3],t+beta5[4]*delta_t)
        u_a=jnp.ones((3,5))*(x.reshape(3,-1))+delta_t*(
            theta5[0].reshape(1,-1)*rhs0.reshape(-1,1)+
            theta5[1].reshape(1,-1)*rhs1.reshape(-1,1)+
            theta5[2].reshape(1,-1)*rhs2.reshape(-1,1)+
            theta5[3].reshape(1,-1)*rhs3.reshape(-1,1)+
            theta5[4].reshape(1,-1)*rhs4.reshape(-1,1)
        )
        u_a=u_a[:,1:]
    return u_a

def dec48_impl_inner_loop(x,u_a,t,delta_t):
    SS=jac(x,t)
    u_a=jnp.ones((3,4))*(x.reshape(3,-1))
    invJac=jnp.linalg.inv(jnp.eye(3).reshape(1,3,3)-delta_t*beta5[1:].reshape(-1,1,1)*SS.reshape(1,3,3))
    for k in range(8):
        rhs0=ode(x,t+beta5[0]*delta_t)
        rhs1=ode(u_a[:,0],t+beta5[1]*delta_t)
        rhs2=ode(u_a[:,1],t+beta5[2]*delta_t)
        rhs3=ode(u_a[:,2],t+beta5[3]*delta_t)   
        rhs4=ode(u_a[:,3],t+beta5[4]*delta_t)
        sum_term=theta5[0].reshape(1,-1)*rhs0.reshape(-1,1)+theta5[1].reshape(1,-1)*rhs1.reshape(-1,1)+theta5[2].reshape(1,-1)*rhs2.reshape(-1,1)+theta5[3].reshape(1,-1)*rhs3.reshape(-1,1)+theta5[4].reshape(1,-1)*rhs4.reshape(-1,1)
        u_a=u_a+delta_t*(jnp.einsum("Bij,jB->iB",invJac,(-(u_a-x.reshape(3,-1))/delta_t+sum_term[:,1:])))
    return u_a


dec48_inner_loop=jax.jit(dec48_inner_loop)
dec48_impl_inner_loop=jax.jit(dec48_impl_inner_loop)


ode=jax.jit(ode)

variables = {'params': 
             {'fc0': {'kernel': fc0w, 'bias': fc0b},
             'fc2': {'kernel': fc2w, 'bias': fc2b},
            'fc4': {'kernel': fc4w, 'bias': fc4b},
            'fc6': {'kernel': fc6w, 'bias': fc6b},
             }}

x0=np.load("start.npy")
true=np.load("new_latent_space.npy")
def explicit_euler_method(x0,timesteps):
    tspan=jnp.linspace(0,1,timesteps)
    x=x0
    for i in range(timesteps-1):
        x=x+(tspan[i+1]-tspan[i])*ode(x,tspan[i])
    return x

def explicitRK(y_0, N_time, A, b, c):
    tspan=jnp.linspace(0,1,N_time)
    dim=len(y_0)       
    y=jnp.zeros((dim,N_time))    
    y=y.at[:,0].set(y_0)                 
    S=jnp.shape(A)[0]
    u=jnp.zeros((dim,S)) 
    Fu=jnp.zeros((dim,S))
    for n in range(N_time-1):
        delta_t=tspan[n+1]-tspan[n]
        for k in range(S):
            u=u.at[:,k].set(y[:,n] )
            for j in range(k):
                u=u.at[:,k].add(delta_t*A[k,j]*Fu[:,j])
            Fu=Fu.at[:,k].set(ode(u[:,k],tspan[n]+delta_t*c[k]))
        y=y.at[:,n+1].set(y[:,n])
        for j in range(S):
            y=y.at[:,n+1].add(delta_t*b[j]*Fu[:,j])
    return y[:,-1]  

def rk44(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    x=x0
    delta_t=tspan[1]-tspan[0]
    for i in range(timesteps-1):
        k1=ode(x,tspan[i])
        k2=ode(x+delta_t/2*k1,tspan[i]+delta_t/2)
        k3=ode(x+delta_t/2*k2,tspan[i]+delta_t/2)
        k4=ode(x+delta_t*k3,tspan[i]+delta_t)
        x=x+delta_t/6*(k1+2*k2+2*k3+k4)
    return x

def heun33(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    x=x0
    delta_t=tspan[1]-tspan[0]
    for i in range(timesteps-1):
        k1=ode(x,tspan[i])
        k2=ode(x+delta_t/3*k1,tspan[i]+delta_t/3)
        k3=ode(x+2*delta_t/3*k2,tspan[i]+2*delta_t/3)
        x=x+delta_t/4*(k1+3*k3)
    return x

def mid22(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    x=x0
    delta_t=tspan[1]-tspan[0]
    for i in range(timesteps-1):
        k1=ode(x,tspan[i])
        k2=ode(x+delta_t/2*k1,tspan[i]+delta_t/2)
        x=x+delta_t*k2
    return x

def amb2(x0,timesteps):
    tspan=jnp.linspace(0,1,timesteps)
    x=x0
    delta_t=(tspan[1]-tspan[0])
    x1=x0+delta_t*ode(x,tspan[0])
    x2s=x1+3/2*delta_t*ode(x1,tspan[1])-1/2*delta_t*ode(x0,tspan[0])
    x2=x1+1/2*delta_t*(ode(x1,tspan[1])+ode(x2s,tspan[2]))
    for i in range(2,timesteps-1):
        x0=x1.copy()
        x1=x2.copy()
        x2s=x1+3/2*delta_t*ode(x1,tspan[i])-1/2*delta_t*ode(x0,tspan[i-1])
        x2=x1+1/2*delta_t*(ode(x1,tspan[i])+ode(x2s,tspan[i+1]))
    return x2

def am2(x0,timesteps):
    tspan=jnp.linspace(0,1,timesteps)
    x=x0
    delta_t=(tspan[1]-tspan[0])
    x1=x0+delta_t*ode(x,tspan[0])
    x2=x1+3/2*delta_t*ode(x1,tspan[1])-1/2*delta_t*ode(x0,tspan[0])
    for i in range(2,timesteps-1):
        x0=x1.copy()
        x1=x2.copy()
        x2=x1+3/2*delta_t*ode(x1,tspan[i])-1/2*delta_t*ode(x0,tspan[i-1])
    return x2

def lglnodes(n,eps=10**-15):
    w = np.zeros((n+1,))
    x = np.zeros((n+1,))
    xold = np.zeros((n+1,))

    # The Legendre Vandermonde Matrix
    P = np.zeros((n+1,n+1))

    epss = eps

    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    for i in range(n+1): 
        x[i] = -np.cos(np.pi*i / n)
  
  
    # Compute P using the recursion relation
    # Compute its first and second derivatives and 
    # update x using the Newton-Raphson method.
    
    xold = 2.0
    
    for i in range(100):
        xold = x
       
        P[:,0] = 1.0 
        P[:,1] = x
       
        for k in range(2,n+1):
            P[:,k] = ( (2*k-1)*x*P[:,k-1] - (k-1)*P[:,k-2] ) / k
       
        x = xold - ( x*P[:,n] - P[:,n-1] )/( (n+1)*P[:,n]) 
        
        if (max(abs(x - xold).flatten()) < epss ):
            break 
    
    w = 2.0 / ( (n*(n+1))*(P[:,n]**2))
    
    return x, w


def lagrange_basis(nodes,x,k):
    y=np.zeros(x.size)
    for ix, xi in enumerate(x):
        tmp=[(xi-nodes[j])/(nodes[k]-nodes[j])  for j in range(len(nodes)) if j!=k]
        y[ix]=np.prod(tmp)
    return y

def get_nodes(order):
    nodes, w = lglnodes(order-1,10**-15)
    nodes=nodes*0.5+0.5
    w = w*0.5
    return nodes, w
        
def compute_theta_DeC(order):
    # Polynomial nodes
    nodes, w = get_nodes(order)
    # Quadrature nodes (exact)
    int_nodes, int_w = get_nodes(order)
    # generate theta and beta coefficients 
    theta = np.zeros((order,order))
    beta = np.zeros(order)
    # Loop over coefficients
    for m in range(order):
        beta[m] = nodes[m]
        # Rescaling of quadrature points in the interval [0,beta[m]]
        nodes_m = int_nodes*(beta[m])
        w_m = int_w*(beta[m])
        # Computing integrals of int_0^beta[m] phi_r 
        for r in range(order):
            theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
    return theta, beta

theta5,beta5=compute_theta_DeC(5)

theta5=jnp.array(theta5)
beta5=jnp.array(beta5)

def dec48(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    delta_t=tspan[1]-tspan[0]
    x=x0
    for i in range(timesteps-1):
        u_a=jnp.ones((3,4))*(x.reshape(3,-1))
        u_a=dec48_inner_loop(x,u_a,tspan[i],delta_t)
        x=u_a[:,-1]
    return x

def dec48_impl(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    delta_t=tspan[1]-tspan[0]
    x=x0
    for i in range(timesteps-1):
        u_a=jnp.ones((3,3))*(x.reshape(3,-1))
        u_a=dec48_impl_inner_loop(x,u_a,tspan[i],delta_t)
        x=u_a[:,-1]
    return x


explicit_euler_method=jax.vmap(explicit_euler_method, (0, None), 0)
mid22=jax.vmap(mid22, (0, None), 0)
heun33=jax.vmap(heun33, (0, None), 0)
rk44=jax.vmap(rk44, (0, None), 0)
am2=jax.vmap(am2, (0, None), 0)
amb2=jax.vmap(amb2, (0, None), 0)
dec48=jax.vmap(dec48, (0, None), 0)
dec48_impl=jax.vmap(dec48_impl, (0, None), 0)

if __name__=="__main__":
    methods={
       # "euler_expl_vmap":explicit_euler_method,
       #    "mid22_vmap":mid22,
       #    "heun33_vmap":heun33,
        #   "rk44_vmap":rk44,
        #   "am2_vmap": am2,
         #  "amb2_vmap": amb2,
        #"dec48_vmap": dec48,
        "dec48_impl_vmap": dec48_impl
        }


    x0=np.load("start.npy")
    x1=np.load("new_latent_space.npy")
    x0=jnp.array(x0)
    x1=jnp.array(x1)
    num_timesteps=np.array([100
                            ,1000
                            ,10000
                            ])
    n=len(num_timesteps)
    for method_label in methods.keys():
        print(method_label)
        current_method=methods[method_label]
        times_array=np.zeros(n)
        rel_array=np.zeros(n)
        for i in range(n):
            start=time.time()
            pred=current_method(x0,num_timesteps[i])
            end=time.time()
            rel_array[i]=np.linalg.norm(x1[0]-pred[0])/(0.5*np.linalg.norm(pred[0])+0.5*np.linalg.norm(x1[0]))
            times_array[i]=end-start
        print(rel_array[-1])
        times_array=times_array/1000
        np.save("results/"+method_label+".npy",{"num_timesteps":num_timesteps,"times":times_array,"rel":rel_array})

