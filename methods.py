import numpy as np
import torch
from reference_method import ContinousScore
import time
from scipy import optimize
from nodepy import rk, ivp
import nodepy.linear_multistep_method as lm

from sympy import sqrt, Rational
from torch.func import jacfwd
model=torch.load("model.pt")
model.eval()
def ode(x,t):
    with torch.no_grad():
        return model.reverse_equivalent_ode(torch.tensor(t,dtype=torch.float32),torch.tensor(x,dtype=torch.float32).unsqueeze(0)).reshape(-1).numpy()

def ode_tfirst(t,x):
    return ode(x,t)


torchjac=jacfwd(model.reverse_equivalent_ode,argnums=1)
def jac(x,t):
    return torchjac(torch.tensor(t,dtype=torch.float32),torch.tensor(x,dtype=torch.float32).unsqueeze(0)).detach().numpy().reshape(3,3)


def explicit_euler_method(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    x=x0
    for i in range(timesteps-1):
        x=x+(tspan[i+1]-tspan[i])*ode(x,tspan[i])
    return x


def implicit_euler_method(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    x=x0
    for i in range(timesteps-1):
        func=lambda y: y-x-(tspan[i+1]-tspan[i])*ode(y,tspan[i+1]) 
        x=optimize.newton(func, x)
    return x

def explicitRK(flux, tspan, y_0, A, b, c):
    N_time=len(tspan)  
    dim=len(y_0)       
    y=np.zeros((dim,N_time))    
    y[:,0]=y_0                 
    S=np.shape(A)[0]
    u=np.zeros((dim,S)) 
    Fu=np.zeros((dim,S))
    for n in range(N_time-1):
        delta_t=tspan[n+1]-tspan[n]
        for k in range(S):
            u[:,k]=y[:,n] 
            for j in range(k):
                u[:,k] =u[:,k]+ delta_t*A[k,j]*Fu[:,j]
            Fu[:,k] = flux(u[:,k],tspan[n]+delta_t*c[k])
        y[:,n+1]=y[:,n]
        for j in range(S):
            y[:,n+1]=y[:,n+1]+delta_t*b[j]*Fu[:,j]
    return y[:,-1]  


def implicitRK(func, jac_func, A, b, tspan, y_0):
    '''
    Implicit RK method with a nonlinear solver
    Input:
    func (nonlinear) function of the ODE, takes input u, t
    jac_func jacobian wrt to u of func, takes input u, t
    tspan vector of timesteps (t^0,...,t^N)
    y_0 initial value    
    '''
    N_time=len(tspan)  # N+1
    dim=len(y_0)          # S
    y=np.zeros((dim,N_time))    # initializing the variable of solutions   
    RKdim = np.shape(A)[0]     # RK dimension 
    c = np.sum(A,axis=1)       # c RK
    y[:,0]=y_0                 # first timestep
    un = np.zeros((dim * RKdim)) # vector of previous value for RK
    t_loc = np.zeros(RKdim)


    for n in range(N_time-1):    # loop through timesteps n=0,..., N-1
        dt = tspan[n+1]-tspan[n] #timestep
        for i in range(RKdim):  # creating vector dimension dim*NRK with un everywhere
            un[i*dim:(i+1)*dim] = y[:,n]
            t_loc[i] = tspan[n] + dt*c[i]  # times of stages

        def res_RK(u):
            """residual equation of implicit RK """
            res = np.zeros(len(u))
            ff = np.zeros(len(u)) # [func(u^{(1)}),...,func(u^{(NRK)})]
            for i in range(RKdim):
                ff[i*dim:(i+1)*dim] = func(u[i*dim:(i+1)*dim],t_loc[i])
            for i in range(RKdim): # U-y^n-A F(U)
                res[i*dim:(i+1)*dim] = u[i*dim:(i+1)*dim] - y[:,n] 
                for j in range(RKdim):
                    res[i*dim:(i+1)*dim] = res[i*dim:(i+1)*dim]\
                        -dt*A[i,j]*ff[j*dim:(j+1)*dim]
            return res
        
        def jac_res_RK(u):
            """jacobian of the residual equation"""
            jac = np.zeros((len(u),len(u))) # dimension (dim*NRK)^2
            jj = np.zeros((len(u),dim))
            for i in range(RKdim): # jacobian of rhs in each variable u^{(i)}
                jj[i*dim:(i+1)*dim,:] = jac_func(u[i*dim:(i+1)*dim],t_loc[i])
            for i in range(RKdim): 
                # jacobian in cell [i,j] of dimension dim x dim
                # is \delta_{ij} I_dim -dt A_ij J_u F(u^{(j)})
                jac[i*dim:(i+1)*dim,i*dim:(i+1)*dim] = np.eye(dim)
                for j in range(RKdim):
                    jac[i*dim:(i+1)*dim,j*dim:(j+1)*dim]=\
                        jac[i*dim:(i+1)*dim,j*dim:(j+1)*dim] \
                        -dt*A[i,j]*jj[j*dim:(j+1)*dim,:]
            return jac

        # finding the solution of the residual equation 
        z = optimize.root(res_RK, un, jac=jac_res_RK, method="lm")
        # reconstructing at new timestep
        y[:,n+1] = y[:,n]
        for i in range(RKdim):
            y[:,n+1] = y[:,n+1] + dt*b[i]*func(z.x[i*dim:(i+1)*dim],t_loc[i])
    return y[:,-1] 

def rk44(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('RK44')
    return explicitRK(ode,tspan,x0,myrk.A,myrk.b,myrk.c)

def mid22(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('Mid22')
    return explicitRK(ode,tspan,x0,myrk.A,myrk.b,myrk.c)


def heun33(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('Heun33')
    return explicitRK(ode,tspan,x0,myrk.A,myrk.b,myrk.c)


def sdirk34(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('SDIRK34')
    return implicitRK(ode,jac,myrk.A,myrk.b,tspan,x0)

def sdirk54(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('SDIRK54')
    return implicitRK(ode,jac,myrk.A,myrk.b,tspan,x0)

def gl2(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('GL2')
    return implicitRK(ode,jac,myrk.A,myrk.b,tspan,x0)

def gl3(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('GL3')
    return implicitRK(ode,jac,myrk.A,myrk.b,tspan,x0)

def radauIIA3(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('RadauIIA3')
    return implicitRK(ode,jac,myrk.A,myrk.b,tspan,x0)

def lobattoIIIA3(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('LobattoIIIA3')
    return implicitRK(ode,jac,myrk.A,myrk.b,tspan,x0)

def lobattoIIIC3(x0,timesteps):
    tspan=np.linspace(0,1,timesteps)
    myrk=rk.loadRKM('LobattoIIIC3')
    return implicitRK(ode,jac,myrk.A,myrk.b,tspan,x0)

def multiAB(flux, tspan, y_0, b):
    N_time=len(tspan)
    dim=y_0.shape[0]     
    y=np.zeros((dim,N_time))
    k = len(b)-1               
    if y_0.shape[1] < k:
        raise ValueError("Input vector is too small")
    y[:,:k]=y_0               
    n0=k-1                    
    Fu=np.zeros((dim,k))
    for j in range(k):
        Fu[:,j]=flux(y[:,j],tspan[j])
    for n in range(n0,N_time-1):    # n=0,..., N-1
        delta_t=tspan[n+1]-tspan[n]
        y[:,n+1]=y[:,n]
        for j in range(k):
            y[:,n+1]=y[:,n+1]+delta_t*b[j]*Fu[:,j]
        Fu[:,:k-1] = Fu[:,1:]
        Fu[:,k-1] = flux(y[:,n+1],tspan[n+1])
    return y[:,-1] 

def multiAMB(flux, tspan, y_0, bAB, bAM):
    # Solving u'=F(u,t)
    # input: flux=F, tspan is a vector of times determining the RK steps
    # input: y_0 the initial condition with the first k values of the solution
    # input: bAB are k+1 b_j coefficients where the last one is 0 of Adam Bashforth
    # input: bAB are k b_j coefficients of Adams Moulton
    N_time=len(tspan)  # N+1
    dim=y_0.shape[0]          # S
    y=np.zeros((dim,N_time))    # initializing the variable of solutions
    k = len(bAB)-1                # size of AB
    if y_0.shape[1] < k:
        raise ValueError("Input vector is too small")
    y[:,:k]=y_0                  # first timesteps 
    n0=k-1                       # last index assigned
    Fu=np.zeros((dim,k))         # Flux at internal stages
    for j in range(k):
        Fu[:,j]=flux(y[:,j], tspan[j])
    for n in range(n0,N_time-1):    # n=0,..., N-1
        delta_t=tspan[n+1]-tspan[n]
        y[:,n+1]=y[:,n]
        for j in range(k):
            y[:,n+1]=y[:,n+1]+delta_t*bAB[j]*Fu[:,j]
        Fu[:,:k-1] = Fu[:,1:]
        Fu[:,k-1] = flux(y[:,n+1], tspan[n+1])
        y[:,n+1] = y[:,n]
        for j in range(k):
            y[:,n+1] =y[:,n+1] +delta_t*bAM[j]*Fu[:,j]
        Fu[:,k-1] =flux(y[:,n+1], tspan[n+1])
    return y[:,-1]

def ab2(x0,timesteps):
    AB=lm.Adams_Bashforth(2)
    tspan=np.linspace(0,1, timesteps)
    x1=x0+(tspan[1]-tspan[0])*ode(x0,tspan[0])
    x1=x1.reshape(3,1)
    x0=x0.reshape(3,1)
    x0=np.concatenate((x0,x1),axis=1)
    return multiAB(ode,tspan,x0,AB.beta)

def amb2(x0,timesteps):
    AB=lm.Adams_Bashforth(2)
    AM=lm.Adams_Moulton(1)
    tspan=np.linspace(0,1, timesteps)
    x1=x0+(tspan[1]-tspan[0])*ode(x0,tspan[0])
    x1=x1.reshape(3,1)
    x0=x0.reshape(3,1)
    x0=np.concatenate((x0,x1),axis=1)
    return multiAMB(ode,tspan,x0,AB.beta, AM.beta)

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


def dec(func, tspan, y_0, M_sub, K_corr):
    '''
    Deferred correction algorithm with the formalism of Abgrall 2017
    Input:
    func: (lambda function) RHS of the ODE
    tspan: (array (N_time)) the timesteps
    y_0: (array (dim)) is the initial value
    M_sub: (int) number of subtimeintervals [t^0,t^m], m=0,\dots, M_sub
    K_corr: (int) number of iterations of the algorithm (order=K_corr)
    distribution: (str) distribution of the subtimenodes between "equispaced", "gaussLobatto"

    Output:
    tspan (array) timestpes
    U (array)     solutions at the different timesteps
    '''
    N_time=len(tspan)
    dim=len(y_0)
    U=np.zeros((dim, N_time))
    u_a=np.zeros((dim, M_sub+1))
    rhs= np.zeros((dim,M_sub+1))
    Theta, beta = compute_theta_DeC(M_sub+1)
    U[:,0]=y_0
    for it in range(1, N_time):
        delta_t=(tspan[it]-tspan[it-1])
        for m in range(M_sub+1):
            u_a[:,m]=U[:,it-1]
        for k in range(1,K_corr+1):
            for r in range(M_sub+1):
                rhs[:,r]=func(u_a[:,r],tspan[it-1]+beta[r]*delta_t)
            for m in range(1,M_sub+1):
                u_a[:,m]= U[:,it-1]+delta_t*sum([Theta[r,m]*rhs[:,r] for r in range(M_sub+1)])
        U[:,it]=u_a[:,M_sub]
    return U[:,-1]



def dec48(x0,timesteps):
    tspan=np.linspace(0,1, timesteps)
    return dec(ode,tspan,x0,4,8)
    

def decImplicit(func,jac_stiff, tspan, y_0, M_sub, K_corr):
    """
    The decImplicit function implements the DeC (Difference-Corrected) method for solving
    initial value problems (IVPs) in ordinary differential equations (ODEs).

    Inputs:
    func: callable      Right-hand side of the ODE system.
    jac_stiff: callable Jacobian matrix of the right-hand side of the ODE system, used for stiffness correction.
    tspan: numpy array  Time steps for the simulation.
    y_0: numpy array    Initial conditions for the ODE system.
    M_sub: int          Number of sub-intervals.
    K_corr: int         Number of correction steps.
    distribution: str   nodes distribution

    Outputs:
    tspan: numpy array  Time steps for the simulation.
    U: numpy array      Numerical solution to the ODE system.

    Note:
    The inputs func and jac_stiff should be function handles that return the right-hand side and
    Jacobian matrix, respectively, for the given input argument(s).
    """
    
    N_time=len(tspan) # Compute the number of time steps
    dim=len(y_0) # Compute the dimension of the ODE system
    U=np.zeros((dim, N_time)) # Initialize the solution array
    
    # Initialize arrays for predictor and corrector
    u_p=np.zeros((dim, M_sub+1))
    u_a=np.zeros((dim, M_sub+1))
    
    # Initialize temporary arrays
    u_help= np.zeros(dim)
    rhs= np.zeros((dim,M_sub+1))
    
    # Compute the coefficients for the DeC method
    Theta, beta = compute_theta_DeC(M_sub+1)
    
    # Initialize the inverse Jacobian matrix
    invJac=np.zeros((M_sub+1,dim,dim))
    
    # Set the initial conditions
    U[:,0]=y_0
    
    # Loop over each time step
    for it in range(1, N_time):
        # Compute the time step size
        delta_t=(tspan[it]-tspan[it-1])
        
        # Initialize the predictor and corrector arrays
        for m in range(M_sub+1):
            u_a[:,m]=U[:,it-1]
        
        # Compute the Jacobian matrix at the start of the time step
        SS=jac_stiff(u_a[:,0],tspan[it-1])
        
        # Compute the inverse Jacobian matrix for each sub-interval
        for m in range(1,M_sub+1):
            invJac[m,:,:]=np.linalg.inv(np.eye(dim) - delta_t*beta[m]*SS)

        # Iterative loop for correction
        for k in range(1, K_corr + 1):
            # Copy the previous solution
            u_p = np.copy(u_a)
            # Compute the right-hand side of the ODE
            for r in range(M_sub + 1):
                rhs[:, r] = func(u_a[:, r],tspan[it-1]+beta[r]*delta_t)
            # Update the intermediate solution
            for m in range(1, M_sub + 1):
                u_a[:, m] = u_a[:, m] + delta_t * np.matmul(invJac[m, :, :], (
                    -(u_a[:, m] - u_a[:, 0]) / delta_t + sum(
                        [Theta[r, m] * rhs[:, r] for r in range(M_sub + 1)])))
        # Update the solution
        U[:, it] = u_a[:, M_sub]
    return U[:,-1]


def dec48_impl(x0,timesteps):
    return decImplicit(ode,jac,np.linspace(0,1,timesteps),x0,4,8)



if __name__=="__main__":

    methods={
          # "euler_expl":explicit_euler_method,
          # "euler_impl":implicit_euler_method,
          # "mid22": mid22,
          # "heun33":heun33,
          # "rk44":rk44,
           #"sdirk34":sdirk34,
           #"sdirk54":sdirk54,
           #"gl2": gl2,
           #"gl3": gl3,
           #"radauIIA3": radauIIA3,
           #"lobattoIIIA3": lobattoIIIA3,
           #"lobattoIIIC3": lobattoIIIC3,
           #"ab2": ab2,
           #"amb2":amb2,
           #"dec48":dec48,
           #"dec48_impl":dec48_impl
           }



    x0=np.load("start.npy")[0]
    x1=np.load("new_latent_space.npy")[0]
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
            rel_array[i]=np.linalg.norm(x1-pred)/(0.5*np.linalg.norm(pred)+0.5*np.linalg.norm(x1))
            times_array[i]=end-start
        print(rel_array[-1])
        np.save("results/"+method_label+".npy",{"num_timesteps":num_timesteps,"times":times_array,"rel":rel_array})
