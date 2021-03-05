import random as rd
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit, float64, prange
#Lista de parametros:
inertia = 0.1 #peso de la velocidad anterior
indiv = 1.9  #peso de la mejor posicion hasta el momento del agente
social = 2.1  #peso de la mejor posicion hasta el momento 
              #en el vecindario del agente
quantity = 1000 #cantidad de agentes
it_number = 1000 #numero de iteraciones
dim = 30 #dimension
@jit(float64(float64[:]), nopython=True)
def f(p):
    a = 0
    for i in prange(dim):
        a += p[i]**2
    return a
Range = 100 #rango sobre el que se generan (positivo y negativo)
VRange = 100 #maxima velocidad inicial en cada coordenada (positivo y negativo)
###############################################################################
@jit(nopython=True)
def initAG():   #Genera la lista inicial de agentes, formato: [(posicion, velocidad, best)]
    agents = np.zeros((quantity,3,dim))
    for i in range(quantity):
        position0 = np.array([rd.randrange(-Range,Range+1) for j in range(dim)], dtype=np.float64)
        velocity0 = np.array([rd.randrange(-VRange,VRange+1) for j in range(dim)], dtype=np.float64)
        best0 = np.copy(position0)
        agents[i] = np.stack((position0,velocity0,best0))
    return agents

@jit(nopython=True, parallel=True)
def update(agents,global_best):
    for i in prange(quantity):
        agents[i] = new_agent(agents[i],global_best)
    return agents
@jit(nopython=True)
def new_agent(agent,global_best):
    pos = agent[0]
    v = agent[1]
    p_best = agent[2]
    new_v = np.empty(dim, dtype=np.float64)
    new_pos = np.empty(dim, dtype=np.float64)
    for j in prange(dim):
        new_v[j] = inertia*v[j] + indiv*rd.random()*(p_best[j]-pos[j]) + social*rd.random()*(global_best[j]-pos[j])
        new_pos[j] = pos[j] + new_v[j]
    if f(new_pos) < f(p_best):
        return np.stack((new_pos,new_v,new_pos))
    else:
        return np.stack((new_pos,new_v,p_best))

@jit(nopython=True)
def best(agents):   #Devuelve la mejor posicion global hasta el momento
    candidates = [f(agents[i][2]) for i in range(quantity)]   #lista de personal bests
    agent_index = candidates.index(min(candidates))
    return np.copy(agents[agent_index][2])

if __name__ == "__main__":
    agents = initAG()
    global_best = best(agents)
    for i in range(it_number):
        print("iteracion: ", i)
        print("valor de f: ", f(global_best))
        agents = update(agents,global_best)
        global_best = best(agents)














