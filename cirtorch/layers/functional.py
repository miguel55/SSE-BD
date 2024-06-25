import numpy as np

import torch

import random

# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


# --------------------------------------
# loss
# --------------------------------------

def my_aggregate_exponential_gamma_loss(x, label, Lw=None, gamma=1.0, alpha=1.0, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()

    numPos = torch.sum(label!=0).item()
    numNeg = 1.*torch.sum(label==0).item()
    xp = x[:, 0:numPos]
    xn = x[:, numPos::]
    
    # compute all unique pairwise positive distances    
    iup,jup = np.triu_indices(numPos, k=1)
    inp = np.hstack((iup,jup))
    jnp = np.hstack((jup,iup))
    
    i = torch.LongTensor(inp).cuda()
    j = torch.LongTensor(jnp).cuda()
    
    difp = xp[:,i] - xp[:,j]
    dpos = torch.pow(difp+eps, 2).sum(dim=0).sqrt()
    
    # compute all unique pairwise negative distances, and delete one non 
    # hard negative per positive
    ilo=np.repeat(np.arange(numPos),numNeg-1)
    jlo=np.repeat(np.arange(int(numNeg))[None,:],numPos,axis=0).tolist()
    for iel in range(len(jlo)):
        jlo[iel].remove(random.sample(set(jlo[iel]) - set([iel]),1)[0])
    jlo=np.reshape(np.array(jlo),(-1))
    i = torch.LongTensor(ilo).cuda()
    j = torch.LongTensor(jlo).cuda()
    
    
    difn = xp[:,i] - xn[:,j]
    dneg = torch.pow(difn+eps, 2).sum(dim=0).sqrt()
    
    Dp = -torch.log(torch.sum(torch.exp(-gamma * dpos)))
    Dn = -torch.log(torch.sum(torch.exp(-gamma * dneg)))

    dda = Dn - alpha*Dp
    y1 = torch.exp(-dda)
    y1 = torch.sum(y1)
    return y1

def my_aggregate_exponential_gamma_loss_id(x, label, Lw=None, gamma=1.0, alpha=1.0, beta=0.0, eps=1e-6):
    
    if( Lw is not None ):
        x = torch.mm(Lw['P'], x-Lw['m'])
        x = l2n(x.t()).t()

    numPos = torch.sum(label!=0).item()
    numNeg = 1.*torch.sum(label==0).item()
    xp = x[:, 0:numPos]
    xn = x[:, numPos::]
    
    # compute all unique pairwise positive distances    
    iup,jup = np.triu_indices(numPos, k=1)
    inp = np.hstack((iup,jup))
    jnp = np.hstack((jup,iup))
    
    i = torch.LongTensor(inp).cuda()
    j = torch.LongTensor(jnp).cuda()
    
    difp = xp[:,i] - xp[:,j]
    dpos = torch.pow(difp+eps, 2).sum(dim=0).sqrt()
    
    # compute all unique pairwise negative distances, and delete one non 
    # hard negative per positive
    ilo=np.repeat(np.arange(numPos),numNeg-1)
    jlo=np.repeat(np.arange(int(numNeg))[None,:],numPos,axis=0).tolist()
    for iel in range(len(jlo)):
        jlo[iel].remove(random.sample(set(jlo[iel]) - set([iel]),1)[0])
    jlo=np.reshape(np.array(jlo),(-1))
    i = torch.LongTensor(ilo).cuda()
    j = torch.LongTensor(jlo).cuda()
    
    
    difn = xp[:,i] - xn[:,j]
    dneg = torch.pow(difn+eps, 2).sum(dim=0).sqrt()
    
    Dp = -torch.log(1/(numPos**2-numPos)*torch.sum(torch.exp(-gamma * dpos)))
    Dn = -torch.log(1/(numPos*numNeg-numPos)*torch.sum(torch.exp(-gamma * dneg)))

    dda = Dn - alpha*Dp
    y1 = torch.exp(-dda)
    y1 = torch.sum(y1)
    return y1