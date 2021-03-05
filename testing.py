from funciones import *
from methods import *
import time
import matplotlib.pyplot as plt
from PSO_granularidad import *
from DE_granularidad import *
from FA_granularidad import *
from ABC_granularidad import *

"""
Dada una implementación como función de un algoritmo, hay dos cosas que puede interesar
que devuelva: la evolución completa de la ejecución y el tiempo de ejecución. Las funciones
que definidas en otros archivos para testear aquí devuelven una cosa u otra a fin de facilitar
los testeos, pero la adaptación para obtener lo uno o lo otro (o algo distinto, como únicamente
el vector final) son extremadamente sencillas una vez implementado el algoritmo en sí.
"""

def testVelocidades():
    """
    Esta función testea las velocidades de cada versión paralela y no paralela de cada algoritmo.
    Se puede ejecutar tal y como está.
    
    Todo está definido en el propio cuerpo de la función. Para cambiar las funciones a testear,
    hay que cambiar la lista "funs" que se define al principio. En el archivo "funciones.py" se
    encuentran definidas todas las opciones. Con funs = funTodas se prueban todas.
    Para cambiar el número de ejecuciones de donde sale la media, alterar el parámetros "numEjs",
    que es una unidad mayor que el número de ejecuciones que se harán.
    Para cambiar los parámetros, hacerlo en el cuerpo de las llamadas a los algoritmos (aunque no
    afectan demasiado al tiempo de ejecución, que es lo que se testea). 
    """
    funs = [fun2,fun3,fun8,funHimmel,funBooth]
    numEjs = 11
    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            result = PSONP(quantity=1000, it_number=1000, dim=funs[i][1], Range=funs[i][2],
                         VRange=100, inertia=0.1, indiv=1.9, social=2.1, f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El PSO (no paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")
    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            result = PSO(quantity=1000, it_number=1000, dim=funs[i][1], Range=funs[i][2],
                         VRange=100, inertia=0.1, indiv=1.9, social=2.1, f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El PSO (paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")
    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            quantity=1000
            dim=funs[i][1]
            result = ABCNP(quantity=1000, it_number=1000, dim=funs[i][1], Range=funs[i][2], limit=quantity*dim/2, f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El ABC (no paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")
    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            quantity=1000
            dim=funs[i][1]
            result = ABC(quantity=1000, it_number=1000, dim=funs[i][1], Range=funs[i][2], limit=quantity*dim/2, f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El ABC (paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")

    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            result = DENP(quantity=1000, it_number=1000, dim=funs[i][1], CP=0.5, F=0.8, Range=funs[i][2], f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El DE (no paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")
    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            result = DE(quantity=1000, it_number=1000, dim=funs[i][1], CP=0.5, F=0.8, Range=funs[i][2], f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El DE (paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")
        
    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            Range=funs[i][2]
            result = FANP(quantity=1000,it_number=1000,dim=funs[i][1],Range=funs[i][2],
                          alfa0=0.01*Range*2,reduct=0.97,gamma=1/((2*Range)**2),beta0=1,f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El FA (no paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")
    for i in range(len(funs)):
        t = 0
        for j in range(numEjs):
            Range=funs[i][2]
            result = FA(quantity=1000,it_number=1000,dim=funs[i][1],Range=funs[i][2],
                          alfa0=0.01*Range*2,reduct=0.97,gamma=1/((2*Range)**2),beta0=1,f=funs[i][0])
            if j!=0:  #para evitar contar la compilación inicial de Numba
                t += result
            #print("La ejecucion ", j, "-esima ha tardado ", result)
        print("El FA (paralelizado) aplicado a la funcion numero ", i, " ha tardado de media ", t/(numEjs-1), " segundos")


def testGranularidadesPSO(reps,g1,l1,g2,l2,g3,l3,fun,args):
    """
    Esta función testea las distintas convergencias de las granularidades que se le indiquen, para el
    algoritmo PSO.
    Una llamada posible sería testGranularidadesPSO(10,1,1000,4,250,20,50,fun3,(100,10000,0.1,1.9,2.1))
    La primera coordenada indica el número de ejecuciones del que obtener la media, las seis siguientes,
    los tres pares de valores (cantidad de subproblemas, intervalos de comunicación) que definen la
    granularidad, la octava debe ser una estructura "fun" de las definidas en "funciones.py" y la
    novena es una terna con los parámetros del algoritmo.
    """
    quantity, it_number, inertia, indiv, social = args
    o = PSOGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],VRange=fun[2],
            inertia=inertia, indiv=indiv, social=social, f=fun[0], gran=g1, lapso=l1) #primera llamada para descartar el tiempo de compilacion inicial
    for i in range(reps):
        a = PSOGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],VRange=fun[2],
            inertia=inertia, indiv=indiv, social=social, f=fun[0], gran=g1, lapso=l1) 
        a = a[0]
        b = PSOGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],VRange=fun[2],
            inertia=inertia, indiv=indiv, social=social, f=fun[0], gran=g2, lapso=l2) #repetimos pa que los tiempos de la primera compilacion no importen
        b = b[0]
        c = PSOGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],VRange=fun[2],
            inertia=inertia, indiv=indiv, social=social, f=fun[0], gran=g3, lapso=l3) #repetimos pa que los tiempos de la primera compilacion no importen
        c = c[0]
        if i ==0:
            p,q,r = a,b,c
        else:
            p += a
            q += b
            r += c
    p,q,r = (p/reps,q/reps,r/reps)
    a_its = p[:,0]
    a_tiempos = p[:,1]
    b_its = q[:,0]
    b_tiempos = q[:,1]
    c_its = r[:,0]
    c_tiempos = r[:,1]
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()

def testGranularidadesDE(reps,g1,l1,g2,l2,g3,l3,fun,args):
    """
    Esta función testea las distintas convergencias de las granularidades que se le indiquen, para el
    algoritmo DE.
    Una llamada posible sería testGranularidadesDE(10,1,1000,4,250,20,50,fun3,(100,10000,0.8,0.8))
    La primera coordenada indica el número de ejecuciones del que obtener la media, las seis siguientes,
    los tres pares de valores (cantidad de subproblemas, intervalos de comunicación) que definen la
    granularidad, la octava debe ser una estructura "fun" de las definidas en "funciones.py" y la
    novena es una terna con los parámetros del algoritmo.
    """
    quantity, it_number, CP, F = args
    o = DEGran(quantity=quantity, it_number=it_number, dim=fun[1], CP=CP, F=F,
           Range=fun[2], f=fun[0], gran=g1,lapso=l1) #primera llamada para descartar el tiempo de compilacion inicial
    for i in range(reps):
        a = DEGran(quantity=quantity, it_number=it_number, dim=fun[1], CP=CP, F=F,
               Range=fun[2], f=fun[0], gran=g1,lapso=l1)  
        a = a[0]
        b = DEGran(quantity=quantity, it_number=it_number, dim=fun[1], CP=CP, F=F,
               Range=fun[2], f=fun[0], gran=g2,lapso=l2)  
        b = b[0]
        c = DEGran(quantity=quantity, it_number=it_number, dim=fun[1], CP=CP, F=F,
               Range=fun[2], f=fun[0], gran=g3,lapso=l3)  
        c = c[0]
        if i ==0:
            p,q,r = a,b,c
        else:
            p += a
            q += b
            r += c
    p,q,r = (p/reps,q/reps,r/reps)
    a_its = p[:,0]
    a_tiempos = p[:,1]
    b_its = q[:,0]
    b_tiempos = q[:,1]
    c_its = r[:,0]
    c_tiempos = r[:,1]
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()
    
def testGranularidadesFA(reps,g1,l1,g2,l2,g3,l3,fun,args):
    """
    Esta función testea las distintas convergencias de las granularidades que se le indiquen, para el
    algoritmo FA.
    Una llamada posible sería testGranularidadesFA(10,1,1000,4,250,20,50,fun3,(100,400,0.01*200,0.98,1/(200**2),1))
    La primera coordenada indica el número de ejecuciones del que obtener la media, las seis siguientes,
    los tres pares de valores (cantidad de subproblemas, intervalos de comunicación) que definen la
    granularidad, la octava debe ser una estructura "fun" de las definidas en "funciones.py" y la
    novena es una terna con los parámetros del algoritmo.
    """
    quantity, it_number, alfa0,reduct,gamma,beta0 = args
    o = FAGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
           alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=fun[0], gran=g1,lapso=l1)
    for i in range(reps):
        a = FAGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
               alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=fun[0], gran=g1,lapso=l1)
        a = a[0]
        b = FAGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
               alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=fun[0], gran=g2,lapso=l2)
        b = b[0]
        c = FAGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
               alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=fun[0], gran=g3,lapso=l3)
        c = c[0]
        if i ==0:
            p,q,r = a,b,c
        else:
            p += a
            q += b
            r += c
    p,q,r = (p/reps,q/reps,r/reps)
    a_its = p[:,0]
    a_tiempos = p[:,1]
    b_its = q[:,0]
    b_tiempos = q[:,1]
    c_its = r[:,0]
    c_tiempos = r[:,1]
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()

def testGranularidadesABC(reps,g1,l1,g2,l2,g3,l3,fun,args):
    """
    Esta función testea las distintas convergencias de las granularidades que se le indiquen, para el
    algoritmo ABC.
    Una llamada posible sería testGranularidadesABC(10,1,1000,4,250,20,50,fun3,(100,5000,50*30))
    La primera coordenada indica el número de ejecuciones del que obtener la media, las seis siguientes,
    los tres pares de valores (cantidad de subproblemas, intervalos de comunicación) que definen la
    granularidad, la octava debe ser una estructura "fun" de las definidas en "funciones.py" y la
    novena es una terna con los parámetros del algoritmo.
    """
    quantity, it_number, limit = args
    o = ABCGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
           limit=limit, f=fun[0], gran=g1,lapso=l1)
    for i in range(reps):
        a = ABCGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
               limit=limit, f=fun[0], gran=g1,lapso=l1)
        a = a[0]
        b = ABCGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
               limit=limit, f=fun[0], gran=g2,lapso=l2)
        b = b[0]
        c = ABCGran(quantity=quantity, it_number=it_number, dim=fun[1], Range=fun[2],
               limit=limit, f=fun[0], gran=g3,lapso=l3)
        c = c[0]
        if i ==0:
            p,q,r = a,b,c
        else:
            p += a
            q += b
            r += c
    p,q,r = (p/reps,q/reps,r/reps)
    a_its = p[:,0]
    a_tiempos = p[:,1]
    b_its = q[:,0]
    b_tiempos = q[:,1]
    c_its = r[:,0]
    c_tiempos = r[:,1]
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()


























