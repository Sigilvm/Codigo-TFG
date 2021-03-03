from numba import jit, float64, prange
import numpy as np

@jit(float64(float64[:]), nopython=True)
def f1(p):
    a = 0
    for i in prange(p.shape[0]):
        a += p[i]**2
    return a
@jit(float64(float64[:]), nopython=True)
def f2(p):
    suma = 0
    prod = 1
    for i in prange(p.shape[0]):
        suma += np.abs(p[i])
        prod *= np.abs(p[i])
    return suma+prod
@jit(float64(float64[:]), nopython=True)
def f3(p):
    a = 0
    for i in prange(p.shape[0]):
        b = 0
        for j in prange(i+1):
            b += p[j]
        a += b**2
    return a
@jit(float64(float64[:]), nopython=True)
def f4(p):
    return np.max(np.abs(p))
@jit(float64(float64[:]), nopython=True)
def f5(p):
    a = 0
    for i in prange(p.shape[0]-1):
        b = p[i+1]-(p[i]**2)
        c = (p[i]-1)**2
        a += 100*b**2+c
    return a
@jit(float64(float64[:]), nopython=True)
def fBeale(p):
    x = p[0]
    y = p[1]
    return (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2
@jit(float64(float64[:]), nopython=True)
def fGold(p):
    x = p[0]
    y = p[1]
    a = 1 + (x+y+1)**2 * (19-14*x+3*x**2-14*y+6*x*y+3*y**2)
    b = 30 + (2*x-3*y)**2 * (18-32*x+12*x**2+48*y-36*x*y+27*y**2)
    return a*b
@jit(float64(float64[:]), nopython=True)
def fBooth(p):
    return (p[0]+2*p[1]-7)**2+(2*p[0]+p[1]-5)**2
@jit(float64(float64[:]), nopython=True)
def fMatyas(p):
    x = p[0]
    y = p[1]
    return 0.26*(x**2+y**2)-0.48*x*y
@jit(float64(float64[:]), nopython=True)
def fHimmel(p):
    return (p[0]**2+p[1]-11)**2+(p[0]+p[1]**2-7)**2
@jit(float64(float64[:]), nopython=True)
def fCamel(p):
    x = p[0]
    y = p[1]
    return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2
@jit(float64(float64[:]), nopython=True)
def fEasom(p):
    x = p[0]
    y = p[1]
    a = -((x-np.pi)**2 + (y-np.pi)**2)
    return -np.cos(x) * np.cos(y) * np.exp(a)

@jit(float64(float64[:]), nopython=True)
def f8(p):
    a = 0
    for i in prange(p.shape[0]):
        a += -p[i]*np.sin(np.sqrt(np.abs(p[i])))
    return a
@jit(float64(float64[:]), nopython=True)
def fRas(p):
    dim = p.shape[0]
    suma = 0
    for i in prange(dim):
        suma += (p[i]**2) - 10*np.cos(2*np.pi*p[i])
    return 10*dim + suma
@jit(float64(float64[:]), nopython=True)
def fAck(p):
    x = p[0]
    y = p[1]
    a = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
    b = -np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))
    return a + b + np.e + 20
@jit(float64(float64[:]), nopython=True)
def fLevi(p):
    x = p[0]
    y = p[1]
    b = (x-1)**2 * (1+(np.sin(3*np.pi*y)**2))
    c = (y-1)**2 * (1+(np.sin(2*np.pi*y)**2))
    return np.sin(3*np.pi*x)**2 + b + c
@jit(float64(float64[:]), nopython=True)
def fEgg(p):
    x = p[0]
    y = p[1]
    a = -(y+47)*np.sin(np.sqrt(np.abs((x/2)+y+47)))
    b = -x*np.sin(np.sqrt(np.abs(x-y-47)))
    return a+b
@jit(float64(float64[:]), nopython=True)
def fTable(p):
    x = p[0]
    y = p[1]
    a = np.abs(1 - np.sqrt(x**2+y**2)/np.pi)
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(a))

#nomenclatura: referencia = (funcion,dim,Range,vectorBest,valueBest,unimodal/multimodal)
fun1 = (f1,30,100,[0],0,'u')
fun2 = (f2,30,10,[0],0,'u')
fun3 = (f3,30,100,[0],0,'u')
fun4 = (f4,30,100,[0],0,'u')
fun5 = (f5,30,30,[0],0,'u')
funBeale = (fBeale,2,4.5,[3,0.5],0,'u')
funGold = (fGold,2,2,[0,-1],3,'u')
funBooth = (fBooth,2,10,[1,3],0,'u')
funMatyas = (fMatyas,2,10,[0,0],0,'u')
funHimmel = (fHimmel,2,5,[[3,2],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126]],0,'u')
funCamel = (fCamel,2,5,[0,0],0,'u')
funEasom = (fEasom,2,100,[np.pi,np.pi],-1,'u')

fun8 = (f8,30,500,[420.9687],-12569.5,'m')
funRas = (fRas,30,5.12,[0],0,'m')
funAck = (fAck,2,500,[0,0],0,'m')
funLevi = (fLevi,2,10,[1,1],0,'m')
funEgg = (fEgg,2,512,[512,404.2319],-959.6407,'m')
funTable = (fTable,2,10,[[8.05502,9.66459],[-8.05502,9.66459],[8.05502,-9.66459],[-8.05502,-9.66459]],-19.2085)

funTodas = [fun1,fun2,fun3,fun4,fun5,funBeale,funGold,funBooth,funMatyas,funHimmel,funCamel,funEasom,fun8,funRas,funAck,funLevi,funEgg,funTable]