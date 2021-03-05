import random as rd
import numpy as np
from numba import jit, float64, prange

quantity = 2000 #cantidad de agentes, ha de ser un número par
it_number = 100 #numero de iteraciones
dim = 30 #dimension

half = int(quantity/2)
Range = 100 #rango sobre el que se generan (positivo y negativo)
limit = dim*quantity/2
@jit(float64(float64[:]), nopython=True)
def f(p):
    a = 0
    for i in prange(dim):
        a += p[i]**2
    return a
@jit(nopython=True)
def fitness(p):
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
def initSols():
    sols = np.zeros((half,dim+2))
    for i in range(half):
        position0 = np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)
        sols[i][:dim] = position0
    return sols

@jit(nopython=True, parallel=True)
def EBphase(sols):
    for i in prange(half):
        sols[i] = employed_bee(sols[i], sols)
    return sols
@jit(nopython=True)
def employed_bee(sol, sols):
    j = rd.randrange(0,dim) #coordenada a alterar
    k = rd.randrange(0,half) #solucion auxiliar a usar
    trial_sol = np.copy(sol)
    trial_sol[j] = sol[j] + np.random.uniform(-1,1)*(sol[j]-sols[k][j])
    trial_sol[dim] = 0
    trial_sol[dim+1] = 0
    if f(trial_sol[:dim])<f(sol[:dim]):
        sol = np.copy(trial_sol)
    else:
        sol[dim] = sol[dim] + 1 #aumentamos el conteo de intentos de mejoria
    return sol

@jit(nopython=True, parallel=True)
def OBphase(sols):
    fitness_values = np.array([fitness(sols[i][:dim]) for i in range(half)])
    total_fitness = np.sum(fitness_values)
    probability_values = np.array([fitness_values[i]/total_fitness for i in range(half)])
    for i in range(half):
        selected_solution = random_choice(half,probability_values)
        sols[selected_solution][dim+1] += 1
    for i in prange(half):
        sols[i] = onlooker_group(sols[i], sols)
    return sols
@jit(nopython=True)
def onlooker_group(sol, sols):
    for i in prange(int(sol[dim+1])):
        j = rd.randrange(0,dim) #coordenada a alterar
        k = rd.randrange(0,half) #solucion auxiliar a usar
        trial_sol = np.copy(sol)
        trial_sol[j] = sol[j] + np.random.uniform(-1,1)*(sol[j]-sols[k][j])
        trial_sol[dim] = 0
        trial_sol[dim+1] = 0
        if f(trial_sol[:dim])<f(sol[:dim]):
            sol = np.copy(trial_sol)
        else:
            sol[dim] = sol[dim] + 1 #aumentamos el conteo de intentos de mejoria
    return sol

@jit(nopython=True)
def SCphase(sols):
    for i in range(half):
        if sols[i][dim]>limit:
            position0 = np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)
            sols[i][:dim] = position0
            sols[i][dim] = 0
            sols[i][dim+1] = 0
    return sols

if __name__ == "__main__":
    sols = initSols()
    candidates =[f(sols[i][:dim]) for i in range(half)]
    best = sols[candidates.index(min(candidates))]
    for i in range(it_number):
        print("iteracion numero :", i, ", mejor valor: ", f(best))
        sols = EBphase(sols)
        sols = OBphase(sols)
        sols = SCphase(sols)
        candidates = [f(sols[i][:dim]) for i in range(half)]
        if min(candidates)<f(best):
            best = sols[candidates.index(min(candidates))]











