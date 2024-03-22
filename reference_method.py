
from sklearn import datasets
import torch
from tqdm import trange
import numpy as np
from torch import nn
import torchdiffeq
import time
import matplotlib.pyplot as plt
class ContinousScore(nn.Module):

    def __init__(self,a,b,latent_dim):
        #beta=a+(b-a)*t
        super().__init__()
        self.latent_dim=latent_dim
        self.a=a
        self.b=b
        self.score_model = torch.nn.Sequential(
        torch.nn.Linear(latent_dim+1, 64),
        torch.nn.LogSigmoid(),
        torch.nn.Linear(64, 64),
        torch.nn.LogSigmoid(),
        torch.nn.Linear(64, 64),
        torch.nn.LogSigmoid(),
        torch.nn.Linear(64, latent_dim)
        )

    def beta(self,t):
        return self.a+(self.b-self.a)*t
    
    def int_beta(self,t):
        return (self.a+1/2*(self.b-self.a)*t)*t

    def mu(self,x0,t):
        coeff=torch.exp(-1/2*self.int_beta(t))
        return x0*coeff

    def var(self,x0,t):
        coeff=-torch.expm1(-self.int_beta(t))
        return coeff

    def x(self,x0,t):
        mu=self.mu(x0,t)
        std=self.var(x0,t)**0.5
        return torch.randn_like(x0)*std+mu

    def grad_(self,x,x0,t):
        mu=self.mu(x0,t)
        var=self.var(x0,t)
        return -(x-mu)/var

    def compute_score(self,x,t):
        tmp=torch.cat((x, t), dim=-1)
        return self.score_model(tmp)

    def get_loss(self,x0,t):
        x=self.x(x0,t)
        return torch.mean(self.var(x0,t)*(self.grad_(x,x0,t)-self.compute_score(x,t))**2)
    
    def train_model(self,dataloader,num_epochs=15000,eps=1e-4):
        opt=torch.optim.Adam(self.score_model.parameters(), lr=3e-4)
        for i_epoch in trange(num_epochs):
            total_loss = 0
            for data, in dataloader:
                opt.zero_grad()
                t = torch.rand((data.shape[0], 1)) * (1 - eps) + eps
                loss = self.get_loss(data,t)
                loss.backward()
                opt.step()
                total_loss = total_loss + loss.detach().item() * data.shape[0]
            if i_epoch % 5000 == 0:
                print(f"{total_loss / len(dataloader)}")

    def f(self,x,t):
        return -1/2*self.beta(t)*x
    
    def g(self,x,t):
        return self.beta(t)**0.5
    
    def equivalent_ode(self,t,x):
        #t is a scalar, not a tensor
        t=t.unsqueeze(0).unsqueeze(0).repeat(x.shape[0],1)
        xt = torch.cat((x, t), dim=-1) 
        fxt =self.f(x,t)
        gt= self.g(x,t)
        return fxt-1/2*gt**2*self.score_model(xt) 



    def reverse_equivalent_ode(self,t,x):
        #t is a scalar, not a tensor
        t=1+-t.unsqueeze(0).unsqueeze(0).repeat(x.shape[0],1)
        xt = torch.cat((x, t), dim=-1) 
        fxt =self.f(x,t)
        gt= self.g(x,t)
        return -(fxt-1/2*gt**2*self.score_model(xt)) 


    def sample(self,num_samples):
        xt=torch.randn(num_samples,self.latent_dim)
        samples=torchdiffeq.odeint(self.equivalent_ode,xt,torch.tensor([1,0],dtype=torch.float32),method="rk4", options=dict(step_size=0.001))
        return samples,xt


if __name__ == "__main__":
    mydata,t=datasets.make_swiss_roll(n_samples=1500, noise=0.01, random_state=0)
    mydata=mydata-np.mean(mydata)
    mydata=mydata/np.std(mydata)
    fig,ax=plt.subplots(1,3,figsize=(24,8))
    ax[0].scatter(mydata[:,0],mydata[:,1])
    ax[1].scatter(mydata[:,0],mydata[:,2])
    ax[2].scatter(mydata[:,1],mydata[:,2])
    fig.savefig("data.png")
    np.save("data.npy",mydata)
    mydata=torch.tensor(mydata).float()
    torch.random.manual_seed(0)
    dataset = torch.utils.data.TensorDataset(mydata)
    model=ContinousScore(0.1,20,3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model.train_model(dataloader,300000)
    torch.save(model,"model.pt")
    mystart=time.time()
    samples,start=model.sample(1000)
    myend=time.time()
    np.save("start.npy",start.detach().numpy())
    new_samples=samples.clone()[-1]
    new_samples=new_samples.detach()

    print(torch.cov(new_samples.T))
    print(torch.cov(mydata.T))
    print(torch.mean(new_samples,axis=0))
    print(torch.mean(mydata,axis=0))
    print(torch.min(new_samples))
    print(torch.max(torch.sum(new_samples,axis=1)))
    new_samples=new_samples.numpy()
    fig,ax=plt.subplots(1,3,figsize=(24,8))
    ax[0].scatter(new_samples[:,0],new_samples[:,1])
    ax[1].scatter(new_samples[:,0],new_samples[:,2])
    ax[2].scatter(new_samples[:,1],new_samples[:,2])
    fig.savefig("newdata.png")
    np.save("new_latent_space.npy",new_samples)
   # np.save("new_latent_space",new_samples.numpy())

