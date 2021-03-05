import random as rd
import numpy as np
from numba import jit, float64, prange
#Lista de parámetros:
quantity = 300   #Cantidad de agentes
it_number = 100 #Cantidad de iteraciones
dim = 30 #dimension

Range = 100       #Margen de búsqueda en cada dimensión
alfa0 = 0.01*(2*Range)
reduct = 0.9
gamma = 1/(2*Range)**2
beta0 = 1
#################################
@jit(float64(float64[:]), nopython=True)
def f(p):
    a = 0
    for i in prange(dim):
        a += p[i]**2
    return a
@jit(nopython=True)
def dist(a1, a2):
    new = np.empty_like(a1)
    diffs = a1-a2
    for i in range(a1.shape[0]):
        new[i] = diffs[i]**2
    result = np.sum(new)
    return result

def initF():
    agents = []
    for i in range(quantity):
        agents.append(np.array([rd.randrange(-Range,Range+1) for j in range(dim)], dtype=np.float64))
    return np.array(agents)

@jit(nopython=True, parallel=True)
def update(agents, alfa):
    result = np.empty_like(agents)
    for i in prange(quantity):
        new_agent = np.copy(agents[i])
        for j in range(quantity):
            if j != i:
                if f(agents[j])<f(new_agent):
                    r2 = dist(new_agent, agents[j])
                    new_pos = new_agent + beta0*np.exp(-gamma*r2)*(agents[j]-new_agent) + alfa*np.array([np.random.uniform(-Range,Range) for k in range(dim)])
                    if f(new_pos)<f(new_agent):
                        new_agent = np.copy(new_pos)
        result[i] = new_agent
    return result

if __name__ == "__main__":
    agents = initF()
    alfa = alfa0
    for i in range(it_number):
        agents = update(agents, alfa)
        alfa = alfa * reduct
        best = min([f(agents[i]) for i in range(quantity)])
        print(best)


























