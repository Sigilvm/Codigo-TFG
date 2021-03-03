import numpy as np
from numba import jit, float64, prange
import time
from methods import restrict

def DEinitAg(quantity, dim, Range):
    agents = []
    for i in range(quantity):
        agents.append(np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)) #faltaria el dtype=np.float64)
    return np.array(agents)

@jit(nopython=True, parallel=True)
def DEupdate(agents, quantity, dim, CP, F, f, gran, Range):
    result = np.empty_like(agents)
    for k in prange(gran):
        for i in range(int(quantity/gran)):
            a0 = np.copy(agents[int(quantity/gran)*k+i]) #no se si hace falta copiarlo en realidad
            idxs = np.array([int(quantity/gran)*k+idx for idx in range(int(quantity/gran)) if idx != i])
            parents = np.random.choice(idxs, 3, replace=False)
            a1 = agents[parents[0]]
            a2 = agents[parents[1]]
            a3 = agents[parents[2]]
            donor = a1 + F*(a2-a3)
            probs = np.array([np.random.uniform(0., 1.) for j in range(dim)])
            special = np.random.randint(0,dim)
            trial = np.empty_like(a0)
            for j in range(dim):
                if probs[j]<CP or j == special:
                    trial[j] = donor[j]
                else:
                    trial[j] = a0[j]
            if f(restrict(trial,Range))<f(a0):
                result[int(quantity/gran)*k+i] = restrict(trial,Range)
            else:
                result[int(quantity/gran)*k+i] = a0
    return result

def DE(quantity, it_number, dim, CP, F, Range, f, gran,lapso):
    start = time.time()
    agents = DEinitAg(quantity, dim, Range)
    best = min([f(agents[i]) for i in range(quantity)])
    for i in range(it_number):
        if i == 0:
            performance = np.array([[[i,best],[time.time()-start,best]]])
        if i != 0:
            performance = np.append(performance,[[[i,best],[time.time()-start,best]]], axis=0)
        if i%lapso == 0:
            np.random.shuffle(agents)

        agents = DEupdate(agents, quantity, dim, CP, F, f, gran, Range)
        best = min([f(agents[i]) for i in range(quantity)])
    end = time.time()
    return (performance,best,end-start)



        