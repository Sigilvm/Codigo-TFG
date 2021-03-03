import random as rd
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit, float64, prange

@jit(nopython=True)
def restrict(p, Range):
    for i in prange(p.shape[0]-2):
        if p[i] > Range:
            p[i] = Range
        if p[i] < -Range:
            p[i] = -Range
    return p

@jit(nopython=True)
def fitness(p,f):
    fvalue = f(p)
    if fvalue >= 0:
        return 1/(1+fvalue)
    else:
        return 1+np.abs(fvalue)
@jit(nopython=True)
def random_choice(half,probability_values):
    u = np.sum(probability_values)
    for i in prange(half):
        r = np.random.uniform(0,u)
        if r <= probability_values[i]:
            result = i
            break
        u = u - probability_values[i]
    return result
############################################
@jit(nopython=True)
def initSols(half,dim,Range):
    sols = np.zeros((half,dim+2))
    for i in range(half):
        position0 = np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)
        sols[i][:dim] = position0
    return sols

@jit(nopython=True, parallel=True)
def EBphase(sols,half,dim,f,gran,Range):
    for k in prange(gran):
        for i in range(int(half/gran)):
            sols[int(half/gran)*k + i] = employed_bee(sols[int(half/gran)*k + i], sols, half,dim,f,gran,k,Range)
    return sols
@jit(nopython=True)
def employed_bee(sol, sols, half,dim,f,gran,k,Range):
    j = rd.randrange(0,dim) #coordenada a alterar
    k = rd.randrange(int(half/gran)*k,int(half/gran)*(k+1)) #solucion auxiliar a usar
    trial_sol = np.copy(sol)
    trial_sol[j] = sol[j] + np.random.uniform(-1,1)*(sol[j]-sols[k][j])
    trial_sol[dim] = 0
    trial_sol[dim+1] = 0
    if f(trial_sol[:dim])<f(sol[:dim]):
        sol = np.copy(restrict(trial_sol,Range))
    else:
        sol[dim] = sol[dim] + 1 #aumentamos el conteo de intentos de mejoria
    return sol

@jit(nopython=True, parallel=True)
def OBphase(sols,half,dim,f,gran,Range):
    for k in prange(gran):
        fitness_values = np.array([fitness(sols[int(half/gran)*k+i][:dim],f) for i in range(int(half/gran))])
        total_fitness = np.sum(fitness_values)
        probability_values = np.array([fitness_values[i]/total_fitness for i in range(int(half/gran))])
        for i in range(int(half/gran)):
            selected_solution = random_choice(int(half/gran),probability_values) + int(half/gran)*k
            sols[selected_solution][dim+1] += 1
        for i in range(int(half/gran)):
            sols[int(half/gran)*k+i] = onlooker_group(sols[int(half/gran)*k+i], sols, half,dim,f,gran,Range)
    return sols
@jit(nopython=True)
def onlooker_group(sol, sols, half,dim,f,gran,Range):
    for i in prange(int(sol[dim+1])):
        j = rd.randrange(0,dim) #coordenada a alterar
        k = rd.randrange(0,half) #solucion auxiliar a usar
        trial_sol = np.copy(sol)
        trial_sol[j] = sol[j] + np.random.uniform(-1,1)*(sol[j]-sols[k][j])
        trial_sol[dim] = 0
        trial_sol[dim+1] = 0
        if f(trial_sol[:dim])<f(sol[:dim]):
            sol = np.copy(restrict(trial_sol,Range))
        else:
            sol[dim] = sol[dim] + 1 #aumentamos el conteo de intentos de mejoria
    return sol

@jit(nopython=True)
def SCphase(sols,half,dim,Range,limit):
    for i in range(half):
        if sols[i][dim]>limit:
            position0 = np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)
            sols[i][:dim] = position0
            sols[i][dim] = 0
            sols[i][dim+1] = 0
    return sols

def ABC(quantity, it_number, dim, Range, limit, f, gran,lapso):
    start = time.time()
    half = int(quantity/2)
    sols = initSols(half,dim,Range)
    candidates = [f(sols[i][:dim]) for i in range(half)]
    fbest = min(candidates)
    best = sols[candidates.index(fbest)]
    for i in range(it_number):
        if i == 0:
            performance = np.array([[[i,fbest],[time.time()-start,fbest]]])
        if i != 0:
            performance = np.append(performance,[[[i,fbest],[time.time()-start,fbest]]], axis=0)
        if i%lapso == 0:
            np.random.shuffle(sols)
        
        sols = EBphase(sols,half,dim,f,gran,Range)
        sols = OBphase(sols,half,dim,f,gran,Range)
        sols = SCphase(sols,half,dim,Range,limit)
        candidates = [f(sols[i][:dim]) for i in range(half)]
        if min(candidates)<fbest:
            best = sols[candidates.index(min(candidates))]
            fbest = min(candidates)
    end = time.time()
    return (performance,fbest,end-start)





















