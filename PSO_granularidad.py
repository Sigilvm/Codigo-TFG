import random as rd
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit, float64, prange
from methods import restrict

@jit(nopython=True)
def initAG(quantity,dim,Range,VRange):   #Genera la lista inicial de agentes, formato: [(posicion, velocidad, best)]
    agents = np.zeros((quantity,3,dim))
    for i in range(quantity):
        position0 = np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)
        velocity0 = np.array([np.random.uniform(-VRange,VRange) for j in range(dim)], dtype=np.float64)
        best0 = np.copy(position0)
        agents[i] = np.stack((position0,velocity0,best0))
    return agents

@jit(nopython=True, parallel=True)
def update(agents,global_best,quantity,dim,inertia,indiv,social,f,gran,Range):
    for k in prange(gran):
        global_best = best(agents[int(quantity/gran)*k : int(quantity/gran)*(k+1)],quantity,f)
        for i in range(int(quantity/gran)):
            agents[int(quantity/gran)*k + i] = new_agent(agents[int(quantity/gran)*k + i],global_best,dim,inertia,indiv,social,f,Range)
    return agents

@jit(nopython=True)
def new_agent(agent,global_best,dim,inertia,indiv,social,f,Range):
    pos = agent[0]
    v = agent[1]
    p_best = agent[2]
    new_v = np.empty(dim, dtype=np.float64)
    new_pos = np.empty(dim, dtype=np.float64)
    for j in prange(dim):
        new_v[j] = inertia*v[j] + indiv*rd.random()*(p_best[j]-pos[j]) + social*rd.random()*(global_best[j]-pos[j])
        new_pos[j] = pos[j] + new_v[j]
    new_pos = restrict(new_pos,Range)
    if f(new_pos) < f(p_best):
        return np.stack((new_pos,new_v,new_pos))
    else:
        return np.stack((new_pos,new_v,p_best))
@jit(nopython=True)
def best(agents,quantity,f):   #Devuelve la mejor posicion global hasta el momento
    candidates = [f(agents[i][2]) for i in range(agents.shape[0])]#lista de personal bests
    agent_index = candidates.index(min(candidates))
    return np.copy(agents[agent_index][2])

def PSOGran(quantity, it_number, dim, Range, VRange, inertia, indiv, social, f, gran, lapso):
    start = time.time()
    agents = initAG(quantity,dim,Range,VRange)
    global_best = best(agents,quantity,f)
    for i in range(it_number):
        if i == 0:
            performance = np.array([[[i,f(global_best)],[time.time()-start,f(global_best)]]])
        if i != 0:
            performance = np.append(performance,[[[i,f(global_best)],[time.time()-start,f(global_best)]]], axis=0)
        if i%lapso == 0:
            np.random.shuffle(agents)        
        
        agents = update(agents,global_best,quantity,dim,inertia,indiv,social,f,gran,Range)
        global_best = best(agents,quantity,f)
        
    end = time.time()
    return (performance,f(global_best),end-start)







