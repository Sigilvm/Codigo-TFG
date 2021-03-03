import random as rd
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit, float64, prange
from methods import restrict

@jit(nopython=True)
def dist(a1, a2):
    new = np.empty_like(a1)
    diffs = a1-a2
    for i in range(a1.shape[0]):
        new[i] = diffs[i]**2
    result = np.sum(new)
    return result
############################################
def initF(quantity, dim, Range):
    agents = []
    for i in range(quantity):
        agents.append(np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)) #faltaria el dtype=np.float64)
    return np.array(agents)

@jit(nopython=True)
def update(agents, alfa, quantity, dim, Range, gamma, beta0, f, gran):
    result = np.empty_like(agents)
    for k in prange(gran):
        for i in range(int(quantity/gran)):
            new_agent = np.copy(agents[int(quantity/gran)*k + i])
            for j in range(int(quantity/gran)):
                if j != i:
                    if f(agents[int(quantity/gran)*k + j])<f(new_agent):
                        r2 = dist(new_agent, agents[int(quantity/gran)*k + j])
                        new_pos = new_agent + beta0*np.exp(-gamma*r2)*(agents[int(quantity/gran)*k + j]-new_agent) + alfa*np.array([np.random.uniform(-Range,Range) for k in range(dim)])
                        if f(restrict(new_pos,Range))<f(new_agent):
                            new_agent = np.copy(restrict(new_pos,Range))
            result[int(quantity/gran)*k + i] = new_agent
    return result

def FA(quantity,it_number,dim,Range,alfa0,reduct,gamma,beta0,f,gran,lapso):
    start=time.time()
    agents = initF(quantity, dim, Range)
    best = min([f(agents[i]) for i in range(quantity)])
    alfa = alfa0
    for i in range(it_number):
        if i == 0:
            performance = np.array([[[i,best],[time.time()-start,best]]])
        if i != 0:
            performance = np.append(performance,[[[i,best],[time.time()-start,best]]], axis=0)
        if i%lapso == 0:
            np.random.shuffle(agents)        
        
        agents = update(agents, alfa, quantity, dim, Range, gamma, beta0, f, gran)
        alfa = alfa * reduct
        best = min([f(agents[i]) for i in range(quantity)])

    end=time.time()
    return (performance,best,end-start)


























