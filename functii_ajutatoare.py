import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from scipy.optimize import minimize
 

######## FUNCTII PT GENERARE DI ##########

def exponentiala(lamb):
    u = random.random()
    
    return - 1/lamb * np.log(u)


def norm(miu, sigma):

    while True:
        y = exponentiala(1)

        u1 = random.random()
        u2 = random.random()

        if u1 <= np.exp(- ((y-1)**2)/2 ): 
            x = y
            if u2 <= 1/2:
                x = -np.abs(x)
                return miu + sigma*x
            else:
                x = np.abs(x)
                return miu + sigma* x



def verificare_vector_di(di):

    for elem in di:
        if elem < 0:
            return False
    
    return True

def generare_di(x, ai):

    r = norm(0, 10)

    di = [np.linalg.norm(x-a) - r + norm(0, 1) for a in ai]
    di = np.array(di)

    while verificare_vector_di(di) == False or r < 0:
        
        r = norm(0, 10)
        di = [np.linalg.norm(x-a) - r + norm(0, 1) for a in ai]
        di = np.array(di)

    return di

def generare_di_ce_reurneaza_si_r(x, ai):

    r = norm(0, 10)

    di = [np.linalg.norm(x-a) - r + norm(0, 1) for a in ai]
    di = np.array(di)

    while verificare_vector_di(di) == False or r < 0:
        
        r = norm(0, 10)
        di = [np.linalg.norm(x-a) - r + norm(0, 1) for a in ai]
        di = np.array(di)

    return di, r



######### FUNCTII PT GPS_LS si CF_LS ##########


def r(x, ai, di):
    m = len(ai)
    
    vec = [np.linalg.norm(x - a) for a in ai]

    vec = vec - di

    rez = 1/m * sum(vec)
    

    #rez = np.abs(rez)
    #rez = np.floor(rez)
    rez = max(0, rez)

    return rez


def T(x, ai, di):
    
    m = len(ai)

    vec = [np.linalg.norm(x - a) for a in ai]

    vec = [ix / a for ix,a in zip(x-ai, vec)]

    factor_stang = r(x,ai,di) + di

    vec = [a * b for a,b in zip(factor_stang,vec)]

    return 1/m * sum(ai) + 1/m * sum(vec)



def generare_A_caciula(a):

    A_caciula = list()

    for elem in a:    
        linie = [2* elem[0] , 2*elem[1]  , -1]
        linie = np.array(linie)
        A_caciula.append(linie)

    A_caciula = np.array(A_caciula)
    return A_caciula



def verificare_asumption_matrice(a):

    A_caciula = generare_A_caciula(a)

    A_caciula = np.transpose(A_caciula)

    iter1 = -1

    for linie1 in A_caciula:
        iter1 += 1

        iter2 = -1
        for linie2 in A_caciula:
            iter2 += 1

            if iter1 != iter2:
                
                aux = linie1 / linie2

                flag = True
                for elem1 in aux:
                    for elem2 in aux:
                        if elem1 != elem2:
                            flag = False

                if flag == True:
                    return False
    
    return True




############ FUNCTII PT GPS_SLS #############



def generare_B(a, d):

    B = list()

    for ai, di in zip(a, d):
        aux = [2* ai[0] , 2*ai[1], -1 , 2* di]
        B.append(aux)
    
    B= np.array(B)

    return B


def generare_b(a, d):
    
    b = list()

    b = [np.linalg.norm(ai)**2 - di**2 for ai, di in zip(a,d)]
    b = np.array(b)

    return b



def generare_In(n):

    In = generare_On(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                In[i][j] = 1

    return In

def generare_On(n):
    
    return [[0 for i in range(n)] for i in range(n)]


def generare_D(n):
    
    D = generare_In(n)

    for linie in D:
        linie.append(0)
        linie.append(0)

    D.append([0 for i in range(n+2)])
    D.append([0 if i < n+1 else -1 for i in range(n+2)])

    D = np.array(D)

    return D

def generare_g(n):
    

    linie = [0 for i in range(n)]
    linie.append(1)
    linie.append(0)

    linie = np.array(linie)
    linie = 1/2 * linie
    
    return linie





def generare_E(n):

    E = generare_In(n)

    for linie in E:
        linie.append(0)
    
    E.append([0 for i in range(n+1)])
    
    E = np.array(E)

    return E



def sistem_compatibil(matrice, b):

    pass
    max_rang_extins = 0

    
    copie = copy.deepcopy(matrice)


def generare_beta(n, a):
    E = generare_E(n)
    A_caciula = generare_A_caciula(a)

    rez = E@( np.transpose(A_caciula) @ A_caciula)      # Cine naiba s-a gandit ca poate inmulti matrici de genul?


def cautare_lambda_caciula(B, D):

    lambda_caciula = None

    while True:
        G = np.transpose(B)@B + lambda_caciula * D

        flag = True
        for elem in np.eigvalues(G):
            if elem == False:
                flag = False
        
        if flag == True:
            return lambda_caciula
        
        lambda_caciula = lambda_caciula / 2







############### FUNCTII PT x0 OPTIM PENTRU GPS_LS FOLOSIND GPS_SLS ##################

"""alg pag 22 +  lema 5.1 pagina 21 
Find x0 satisfying : f(x0) < min {f(a1), ..f(an), fliminf }
f_liminf = np.
"""

def h(x,ai,di,j):
    
    copy_ai = np.copy(ai)
    copy_ai= np.delete(ai,j)

    m = len(copy_ai) - 1

    vec = [np.linalg.norm(x - a) for a in copy_ai]
    vec = np.array(vec)
    vec = sum(vec)
    vec = vec/m
    return vec

def g(x,ai,di,j):
       
    copy_ai = np.copy(ai)
    copy_ai= np.delete(ai,j)
    copy_di = np.copy(di)
    copy_di = np.delete(di,j)

    vec = [(np.linalg.norm(x-a) - d)**2 for a,d in zip(copy_ai,copy_di)]
    vec = np.array(vec)
    vec = sum(vec)
    return vec

def f(x,ai,di):
    m = len(ai)
    
    vec = [np.linalg.norm(x - a) - d for a,d in zip(ai,di)]

  
    vec = np.square(vec)
    vec = sum(vec)

    vec = vec - m * r(x,ai,di)**2
   
    return vec

""" Gaseste liminf """


""" transforma in GTRS:  ec. (4.8)   """
def function_liminf(z, a,d):
       


   m = len(a)
   print('this is m',m)
   A = np.transpose(a[0])
   for i in range(1,m):
     aux = np.transpose(a[i])
     A = np.vstack([A,aux])
  
   print(A.shape)
   print(d.shape)
   id_m = [1]
   for i in range(1,m):
     id_m = np.vstack([id_m,[1]])
   
   t3 =  np.add(np.matmul(A,z) , d )
   t1 = np.transpose(t3)
   t2 = id_m - np.matmul(id_m, np.transpose(id_m))/m

   
   
   return np.matmul(np.matmul(t1,t2),t3)

""" conditie min """
def constraint_liminf(z):
  return np.linalg.norm(z) - 1 



"""Gaseste x0 optim pentru GSP LS : """

from scipy.optimize import basinhopping, approx_fprime

def find_x0(ai,di):
    m = len(ai)
    
    z0 = np.random.randn(*ai[0].shape)
    norm = np.linalg.norm(z0)
    z0 = z0/norm 


    cons = {'type':'eq', 'fun': constraint_liminf}
    minimizer_kwargs = {"args":(ai,di),"constraints":cons}
    #f_liminf = minimize(fun = function_liminf,x0 = z0,args = (ai,di), constraints = cons)
    #f_liminf = f_liminf.x
    # pentru functia nonconvexa gaseste toate minimele locale, apoi alege minimul global
    f_liminf = basinhopping(func = function_liminf,x0 = z0, minimizer_kwargs = minimizer_kwargs )
    f_liminf = f_liminf.fun
    vec = [f(a,ai,di) for a in ai]

    minn = min(vec)
    print('minim',minn)
    if f_liminf < minn:
        x0 = np.zeros_like(ai[0]) ### modifica x0 = x_sls 
        return x0
    
    p = vec.index(minn)
    a_p = ai[p]
    if r(ai[p],ai,di) > 0 :
        print('save me')
        print(approx_fprime(a_p,g,(ai,di,p)))
        z1 =  np.multiply(approx_fprime(a_p,g,ai,di,p), (-1)) + np.multiply(approx_fprime(a_p,g,ai,di,p), 2 * m * r(a_p,ai,di) )
        #z1 = np.array(z1)
        zero  = np.zeros_like(z1)
        if z1 != zero:
            v = z1/np.linalg.norm(z1)
        else:
            v = np.random.randn(*z1.shape)
            print("aici1",z1)
            norm = np.linalg.norm(v)
            v = v/norm # v = any normalized vector
    else:
        z = np.gradient(g(a_p,ai,di,p))
        z = np.array(z)
        zero = np.zeros_like(z)
        if z != zero :
            v= (-1) * z/np.linalg.norm(z)
        else:
            v = np.random.randn(*z.shape)
            print('aici2', z)
            norm = np.linalg.norm(v)
            v = v/norm # v = any normalized vector
    
    s = 1 

    while f(a_p + s*v , ai,di) >= f(a_p,ai,di):
        s = s/2
    
    x0 = a_p + s*v
    return x0
