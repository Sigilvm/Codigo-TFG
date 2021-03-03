from funciones import *
####################from methods import *
import time
import matplotlib.pyplot as plt
from PSO16granularidad import *
from DE12granularidad import *
from FA12granularidad import *
from ABC12granularidad import *

def testVelocidades():
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


def testGranularidadesPSO(g1,l1,g2,l2,g3,l3,fun_index,args):
    funs = [fun3]
    quantity, it_number, inertia, indiv, social = args
    o = PSO(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],VRange=funs[fun_index][2],
            inertia=inertia, indiv=indiv, social=social, f=funs[fun_index][0], gran=g1, lapso=l1)
    
    a = PSO(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],VRange=funs[fun_index][2],
            inertia=inertia, indiv=indiv, social=social, f=funs[fun_index][0], gran=g1, lapso=l1) #repetimos pa que los tiempos de la primera compilacion no importen
    a_its = a[0][:,0]
    a_tiempos = a[0][:,1]
    print("tiempo total con la primera configuracion: ", a[2], ", best: ", a[1])
    
    b = PSO(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],VRange=funs[fun_index][2],
            inertia=inertia, indiv=indiv, social=social, f=funs[fun_index][0], gran=g2, lapso=l2)
    b_its = b[0][:,0]
    b_tiempos = b[0][:,1]
    print("tiempo total con la segunda configuracion: ", b[2], ", best: ", b[1])
    
    c = PSO(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],VRange=funs[fun_index][2],
            inertia=inertia, indiv=indiv, social=social, f=funs[fun_index][0], gran=g3, lapso=l3)
    c_its = c[0][:,0]
    c_tiempos = c[0][:,1]
    print("tiempo total con la tercera configuracion: ", c[2], ", best: ", c[1])
    

    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    #plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    #plt.show()
    return (a,b,c)

def media():
    for i in range(15):
        if i ==0:
            a,b,c=testGranularidadesPSO(1,1000,4,250,20,50,0,(100,1000,0.1,1.9,2.1))
        else:
            k=testGranularidadesPSO(1,1000,4,250,20,50,0,(100,1000,0.1,1.9,2.1))
            a += k[0]
            b += k[1]
            c += k[2]
    a,b,c = (a/15,b/15,c/15)
    a_its = a[0][:,0]
    a_tiempos = a[0][:,1]
    b_its = b[0][:,0]
    b_tiempos = b[0][:,1]
    c_its = c[0][:,0]
    c_tiempos = c[0][:,1]
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()
    
    
def testGranularidadesDE(g1,l1,g2,l2,g3,l3,fun_index,args):
    funs = funTodas
    quantity, it_number, CP, F = args
    o = DE(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], CP=CP, F=F,
           Range=funs[fun_index][2], f=funs[fun_index][0], gran=g1,lapso=l1)
    
    a = DE(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], CP=CP, F=F,
           Range=funs[fun_index][2], f=funs[fun_index][0], gran=g1,lapso=l1) #repetimos pa que los tiempos de la primera compilacion no importen
    a_its = a[0][:,0]
    a_tiempos = a[0][:,1]
    print("tiempo total con la primera configuracion: ", a[2], ", best: ", a[1])
    
    b = DE(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], CP=CP, F=F,
           Range=funs[fun_index][2], f=funs[fun_index][0], gran=g2,lapso=l2)
    b_its = b[0][:,0]
    b_tiempos = b[0][:,1]
    print("tiempo total con la segunda configuracion: ", b[2], ", best: ", b[1])
    
    c= DE(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], CP=CP, F=F,
           Range=funs[fun_index][2], f=funs[fun_index][0], gran=g3,lapso=l3)
    c_its = c[0][:,0]
    c_tiempos = c[0][:,1]
    print("tiempo total con la tercera configuracion: ", c[2], ", best: ", c[1])
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()

def testGranularidadesFA(g1,l1,g2,l2,g3,l3,fun_index,args):
    funs = funTodas
    quantity, it_number, alfa0,reduct,gamma,beta0 = args
    o = FA(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=funs[fun_index][0], gran=g1,lapso=l1)
    
    a = FA(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=funs[fun_index][0], gran=g1,lapso=l1) #repetimos pa que los tiempos de la primera compilacion no importen
    a_its = a[0][:,0]
    a_tiempos = a[0][:,1]
    print("tiempo total con la primera configuracion: ", a[2], ", best: ", a[1])
    
    b = FA(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=funs[fun_index][0], gran=g2,lapso=l2)
    b_its = b[0][:,0]
    b_tiempos = b[0][:,1]
    print("tiempo total con la segunda configuracion: ", b[2], ", best: ", b[1])

    c = FA(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           alfa0=alfa0,reduct=reduct,gamma=gamma,beta0=beta0, f=funs[fun_index][0], gran=g3,lapso=l3)
    c_its = c[0][:,0]
    c_tiempos = c[0][:,1]
    print("tiempo total con la tercera configuracion: ", c[2], ", best: ", c[1])
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()


def testGranularidadesABC(g1,l1,g2,l2,g3,l3,fun_index,args):
    funs = funTodas
    quantity, it_number, limit = args
    limit = quantity*funs[fun_index][1]/2
    o = ABC(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           limit=limit, f=funs[fun_index][0], gran=g1,lapso=l1)
    
    a = ABC(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           limit=limit, f=funs[fun_index][0], gran=g1,lapso=l1)
    a_its = a[0][:,0]
    a_tiempos = a[0][:,1]
    print("tiempo total con la primera configuracion: ", a[2], ", best: ", a[1])
    
    b = ABC(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           limit=limit, f=funs[fun_index][0], gran=g2,lapso=l2)
    b_its = b[0][:,0]
    b_tiempos = b[0][:,1]
    print("tiempo total con la segunda configuracion: ", b[2], ", best: ", b[1])

    c = ABC(quantity=quantity, it_number=it_number, dim=funs[fun_index][1], Range=funs[fun_index][2],
           limit=limit, f=funs[fun_index][0], gran=g3,lapso=l3)
    c_its = c[0][:,0]
    c_tiempos = c[0][:,1]
    print("tiempo total con la tercera configuracion: ", c[2], ", best: ", c[1])
    
    plt.scatter(*zip(*a_its[:]),s=0.7)
    plt.scatter(*zip(*b_its[:]),s=0.7)
    plt.scatter(*zip(*c_its[:]),s=0.7)
    plt.show()
    plt.scatter(*zip(*a_tiempos[:]),s=0.7)
    plt.scatter(*zip(*b_tiempos[:]),s=0.7)
    plt.scatter(*zip(*c_tiempos[:]),s=0.7)
    plt.show()

















