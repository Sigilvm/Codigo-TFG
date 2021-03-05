import numpy as np
from numba import jit, float64, prange
from funciones import *
import time
import random as rd
##################################################### FUNCIONES AUXILIARES:
@jit(nopython=True)
def dist(a1, a2):
    new = np.empty_like(a1)
    diffs = a1-a2
    for i in range(a1.shape[0]):
        new[i] = diffs[i]**2
    result = np.sum(new)
    return result
@jit(nopython=True)
def restrict(p, Range):
    for i in prange(p.shape[0]):
        if p[i] > Range:
            p[i] = Range
        if p[i] < -Range:
            p[i] = -Range
    return p
@jit(nopython=True)
def fitness(p, f):
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

######################################################
#                    FA:
######################################################
    
def FAinitF(quantity, dim, Range):
    agents = []
    for i in range(quantity):
        agents.append(np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)) #faltaria el dtype=np.float64)
    return np.array(agents)
@jit(nopython=True, parallel=True)
def FAupdate(agents, alfa, quantity, dim, Range, gamma, beta0, f):
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
        result[i] = restrict(new_agent, Range)
    return result
@jit(nopython=True)
def FANPupdate(agents, alfa, quantity, dim, Range, gamma, beta0, f):
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
        result[i] = restrict(new_agent, Range)
    return result
def FA(quantity,it_number,dim,Range,alfa0,reduct,gamma,beta0,f):
    start = time.time()
    agents = FAinitF(quantity, dim, Range)
    alfa = alfa0
    for i in range(it_number):
        agents = FAupdate(agents, alfa, quantity, dim, Range, gamma, beta0, f)
        alfa = alfa * reduct
        best = min([f(agents[i]) for i in range(quantity)])
        #print(i, best, alfa)
    end = time.time()
    return end-start
def FANP(quantity,it_number,dim,Range,alfa0,reduct,gamma,beta0,f):
    start = time.time()
    agents = FAinitF(quantity, dim, Range)
    alfa = alfa0
    for i in range(it_number):
        agents = FANPupdate(agents, alfa, quantity, dim, Range, gamma, beta0, f)
        alfa = alfa * reduct
        best = min([f(agents[i]) for i in range(quantity)])
        #print(i, best, alfa)
    end = time.time()
    return end-start

######################################################
#                    DE:
######################################################

def DEinitAg(quantity, dim, Range):
    agents = []
    for i in range(quantity):
        agents.append(np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)) #faltaria el dtype=np.float64)
    return np.array(agents)
@jit(nopython=True, parallel=True)
def DEupdate(agents, quantity, dim, CP, F, f):
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
@jit(nopython=True)
def DENPupdate(agents, quantity, dim, CP, F, f):
    result = np.empty_like(agents)
    for i in prange(quantity):
        a0 = np.copy(agents[i]) #no se si hace falta copiarlo en realidad
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
def DE(quantity, it_number, dim, CP, F, Range, f):
    start = time.time()
    agents = DEinitAg(quantity, dim, Range)
    for i in range(it_number):
        best = min([f(agents[i]) for i in range(quantity)])
        #print("iteracion numero: ",i, "valor de f: ", best)
        agents = DEupdate(agents, quantity, dim, CP, F, f)
    end = time.time()
    return end-start
def DENP(quantity, it_number, dim, CP, F, Range, f):
    start = time.time()
    agents = DEinitAg(quantity, dim, Range)
    for i in range(it_number):
        best = min([f(agents[i]) for i in range(quantity)])
        #print("iteracion numero: ",i, "valor de f: ", best)
        agents = DENPupdate(agents, quantity, dim, CP, F, f)
    end = time.time()
    return end-start

######################################################
#                    PSO:
######################################################

@jit(nopython=True)
def PSOinitAG(quantity,dim,Range,VRange):   #Genera la lista inicial de agentes, formato: [(posicion, velocidad, best)]
    agents = np.zeros((quantity,3,dim))
    for i in range(quantity):
        position0 = np.array([rd.randrange(-Range,Range+1) for j in range(dim)], dtype=np.float64)
        velocity0 = np.array([rd.randrange(-VRange,VRange+1) for j in range(dim)], dtype=np.float64)
        best0 = np.copy(position0)
        agents[i] = np.stack((position0,velocity0,best0))
    return agents
@jit(nopython=True, parallel=True)
def PSOupdate(agents,global_best,quantity,dim,inertia,indiv,social,f):
    for i in prange(quantity):
        agents[i] = PSOnew_agent(agents[i],global_best,dim,inertia,indiv,social,f)
    return agents
@jit(nopython=True)
def PSONPupdate(agents,global_best,quantity,dim,inertia,indiv,social,f):
    for i in prange(quantity):
        agents[i] = PSOnew_agent(agents[i],global_best,dim,inertia,indiv,social,f)
    return agents
@jit(nopython=True)
def PSOnew_agent(agent,global_best,dim,inertia,indiv,social,f):
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
def PSObest(agents, quantity, f):   #Devuelve la mejor posicion global hasta el momento
    candidates = [f(agents[i][2]) for i in range(quantity)]   #lista de personal bests
    agent_index = candidates.index(min(candidates))
    return np.copy(agents[agent_index][2])
def PSO(quantity, it_number, dim, Range, VRange, inertia, indiv, social, f):
    start = time.time()
    agents = PSOinitAG(quantity,dim,Range,VRange)
    global_best = PSObest(agents,quantity,f)
    for i in range(it_number):
        #print("iteration: ", i, "fValue: ", f(global_best))
        agents = PSOupdate(agents,global_best,quantity,dim,inertia,indiv,social,f)
        global_best = PSObest(agents,quantity,f)
    end = time.time()
    return end-start
def PSONP(quantity, it_number, dim, Range, VRange, inertia, indiv, social, f):
    start = time.time()
    agents = PSOinitAG(quantity,dim,Range,VRange)
    global_best = PSObest(agents,quantity,f)
    for i in range(it_number):
        #print("iteration: ", i, "fValue: ", f(global_best))
        agents = PSONPupdate(agents,global_best,quantity,dim,inertia,indiv,social,f)
        global_best = PSObest(agents,quantity,f)
    end = time.time()
    return end-start

######################################################
#                    ABC:
######################################################

@jit(nopython=True)
def initSols(half,dim,Range):
    sols = np.zeros((half,dim+2))
    for i in range(half):
        position0 = np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)
        sols[i][:dim] = position0
    return sols
@jit(nopython=True, parallel=True)
def EBphase(sols,half,dim,f):
    for i in prange(half):
        sols[i] = employed_bee(sols[i], sols, half,dim,f)
    return sols
@jit(nopython=True)
def NPEBphase(sols,half,dim,f):
    for i in prange(half):
        sols[i] = employed_bee(sols[i], sols, half,dim,f)
    return sols
@jit(nopython=True)
def employed_bee(sol, sols, half,dim,f):
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
def OBphase(sols,half,dim,f):
    fitness_values = np.array([fitness(sols[i][:dim],f) for i in range(half)])
    total_fitness = np.sum(fitness_values)
    probability_values = np.array([fitness_values[i]/total_fitness for i in range(half)])
    for i in range(half):
        selected_solution = random_choice(half,probability_values)
        sols[selected_solution][dim+1] += 1
    for i in prange(half):
        sols[i] = onlooker_group(sols[i], sols, half,dim,f)
    return sols
@jit(nopython=True)
def NPOBphase(sols,half,dim,f):
    fitness_values = np.array([fitness(sols[i][:dim],f) for i in range(half)])
    total_fitness = np.sum(fitness_values)
    probability_values = np.array([fitness_values[i]/total_fitness for i in range(half)])
    for i in range(half):
        selected_solution = random_choice(half,probability_values)
        sols[selected_solution][dim+1] += 1
    for i in prange(half):
        sols[i] = onlooker_group(sols[i], sols, half,dim,f)
    return sols
@jit(nopython=True)
def onlooker_group(sol, sols, half,dim,f):
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
def SCphase(sols,half,dim,Range,limit):
    for i in range(half):
        if sols[i][dim]>limit:
            position0 = np.array([np.random.uniform(-Range,Range) for j in range(dim)], dtype=np.float64)
            sols[i][:dim] = position0
            sols[i][dim] = 0
            sols[i][dim+1] = 0
    return sols

def ABC(quantity, it_number, dim, Range, limit, f):
    start = time.time()
    half = int(quantity/2)
    sols = initSols(half,dim,Range)
    best = min([f(sols[i][:dim]) for i in range(half)])
    for i in range(it_number):
        #print("iteracion numero :", i, ", mejor valor: ", best)
        sols = EBphase(sols,half,dim,f)
        sols = OBphase(sols,half,dim,f)
        sols = SCphase(sols,half,dim,Range,limit)
        aspirante = min([f(sols[i][:dim]) for i in range(half)])
        if aspirante<best:
            best = aspirante
    end = time.time()
    return end-start
def ABCNP(quantity, it_number, dim, Range, limit, f):
    start = time.time()
    half = int(quantity/2)
    sols = initSols(half,dim,Range)
    best = min([f(sols[i][:dim]) for i in range(half)])
    for i in range(it_number):
        #print("iteracion numero :", i, ", mejor valor: ", best)
        sols = NPEBphase(sols,half,dim,f)
        sols = NPOBphase(sols,half,dim,f)
        sols = SCphase(sols,half,dim,Range,limit)
        aspirante = min([f(sols[i][:dim]) for i in range(half)])
        if aspirante<best:
            best = aspirante
    end = time.time()
    return end-start
























