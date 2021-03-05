import random as rd
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit, float64, prange

quantity = 100 #cantidad de agentes, habitualmente entre 2dim y 40dim
it_number = 1000 #numero de iteraciones
dim = 30 #dimension

CP = 0.3 #crossover probability
F = 0.8 #factor de mutación
Range = 10 #rango sobre el que se generan (positivo y negativo)
@jit(float64(float64[:]), nopython=True)
def f(p):
    a = 0
    for i in prange(dim):
        a += p[i]**2
    return a
############################################
def initAg():
    agents = []
    for i in range(quantity):
        agents.append(np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64))
    return np.array(agents)

@jit(nopython=True, parallel = True)
def update(agents):
    result = np.empty_like(agents)
    for i in prange(quantity):
        a0 = np.copy(agents[i])
        idxs = np.array([idx for idx in range(quantity) if idx != i])
        parents = np.random.choice(idxs, 4, replace=False)
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
        if f(trial)<f(a0):
            result[i] = trial
        else:
            result[i] = a0
    return result

if __name__ == "__main__":
    agents = initAg()
    for i in range(it_number):
        best = min([f(agents[i]) for i in range(quantity)])
        print(best)
        print(i)
        agents = update(agents)



        






        
