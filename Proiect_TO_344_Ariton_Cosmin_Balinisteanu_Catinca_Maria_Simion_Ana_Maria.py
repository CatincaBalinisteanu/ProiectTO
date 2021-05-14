
from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt
import random

"""Repartitie exponentiala """

def exponentiala(lamb):
    u = random.random()
    
    return - 1/lamb * np.log(u)

"""Repartitie normala """

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

"""Verifica di > 0: """

def verificare_vector_di(di):

    for elem in di:
        if elem < 0:
            return False
    
    return True

"""Generare di"""

def generare_di(x, ai):

    r = norm(0, 10)

    di = [np.linalg.norm(x-a) - r + norm(0, 1) for a in ai]
    di = np.array(di)

    while verificare_vector_di(di) == False or r < 0:
        
        r = norm(0, 10)
        di = [np.linalg.norm(x-a) - r + norm(0, 1) for a in ai]
        di = np.array(di)

    return di

"""Functia r => eroarea cauzata de ceas"""

def r(x, ai, di):
    m = len(ai)
    
    vec = [np.linalg.norm(x - a) for a in ai]

    vec = vec - di

    rez = 1/m * sum(vec)
    

    rez = np.abs(rez)
    rez = np.floor(rez)

    return rez

"""alg pag 22 +  lema 5.1 pagina 21 
Find x0 satisfying : f(x0) < min {f(a1), ..f(an), fliminf }

f_liminf = np.
"""

def h(x,ai,di,j):
    m = len(ai) -1
    copy_ai = ai
    copy_ai= np.delete(ai,j)
    vec = [np.linalg.norm(x - a) for a in copy_ai]
    vec = sum(vec)
    vec = vec/m
    return m

def g(x,ai,di,j):
    m = len(ai) -1
    copy_ai = ai
    copy_ai= np.delete(ai,j)
    copy_di = di
    copy_di = np.delete(di,j)
    vec = [np.linalg.norm(x-a) for a in copy_ai]
    vec = vec - copy_di
    vec =  vec^2
    vec = sum(vec)
    return vec

def f(x,ai,di):
    m = len(ai)
    
    vec = [np.linalg.norm(x - a) for a in ai]

    vec = (vec - di)
    vec = np.square(vec)
    vec = sum(vec)

    vec = vec - m * r(x,ai,di)**2
   
    return vec

"""Gaseste xSLS optim """

# ??

"""Gaseste x0 optim pentru GSP LS : """

def find_x0(ai,di):
    m = len(ai)
    f_liminf = np.zeros_like(ai[0])  ## gresit!!! nu stiu cum 
    vec = [f(a,ai,di) for a in ai]
    minn = min(vec)
    if np.less(f_liminf,minn).all():
        x0 = f_liminf  ### gresit !!!!  x0 = x_sls  => nu stiu cine e 
        return x0
    
    p = vec.index(minn)
    a_p = vec[p]
    if r(ai[p]) > 0 :
        z1 =  - np.gradient(g(a_p,ai,di,p)) + 2 * m * r(a_p) * np.gradiend(h(a_p,ai,di,p))
        if z1 != 0:
            v = z1/np.linalg.norm(z1)
        else:
            v = np.random.randn(*z1.shape)
            norm = np.linalg.norm(v)
            v = v/norm # v = any normalized vector
    else:
        z = np.gradient(g(a_p,ai,di,p))
        if z != 0 :
            v= - z/np.linalg.norm(z)
        else:
            v = np.random.randn(*z.shape)
            norm = np.linalg.norm(v)
            v = v/norm # v = any normalized vector
    
    s = 1 
    while f(a_p + s*v) >= f(a_p):
        s = s/2
    
    xo = a_p + s*v
    return x0

"""metoda GPS LS : pentru un set de coordonate ai si distante di ilustram influenta alegerii punctului x initial"""



def fixed_point_GPS_LS_random_influenta_punctului_de_start():

    nr_figura = 0

    NR_MAXIM_DIMENSIUNI = 10
    NR_MAXIM_SATELITI = 20
    INTERVAL_STANGA = -10
    INTERVAL_DREAPTA = 10

    n = random.randint(2, NR_MAXIM_DIMENSIUNI)

    nr_sateliti = 0

    while nr_sateliti < n+1:
        nr_sateliti = random.randint(0,NR_MAXIM_SATELITI)

    x = np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)])

    a = list()
    for elem in range(nr_sateliti):
        a.append(np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)]))
    
    a= np.array(a)

    x_true = np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)])
    pasi_acuratete = 10**2
    d = generare_di(x_true, a)


    plt.figure(nr_figura)
    fixed_point_GPS_LS_histograma_erorilor_influentat_de_x_initial(x, a, d, pasi_acuratete, x_true, INTERVAL_STANGA, INTERVAL_DREAPTA, n)

    print("################REZULTATE PENTRU SPATIU SI SATELITI GENERATI RANDOM AVAND IN VEDERE SCOATEREA IN EVIDENTA A RELEVANTEI ALEGERII PUNCTULUI INITIAL#############")
    print("Nr sateliti: " + str(nr_sateliti))
    print("Spatiul n = " + str(n))


    plt.show()

def T(x, ai, di):
    
    m = len(ai)

    vec = [np.linalg.norm(x - a) for a in ai]

    vec = [ix / a for ix,a in zip(x-ai, vec)]

    factor_stang = r(x,ai,di) + di

    vec = [a * b for a,b in zip(factor_stang,vec)]

    return 1/m * sum(ai) + 1/m * sum(vec)

"""#A fixed point metod for solving GPS LS"""

def fixed_point_GPS_LS(xk, ai, di, pasi_acuratete):

    assert verificare_asumption_matrice(ai) == True , "a1...an se afla intr-un spatiu dimensional afin mai mic[de exmplu: lucram in plan(2D) si satelitii sunt coliniari]"
    assert len(ai) >= len(xk) + 1, "Consecinta directa a faptului ca a1...an nu se afla intr-un spatiu dimensional afin mai mic nu este satisfacuta"


    for i in range(pasi_acuratete):
        xk = T(xk, ai, di)
    return xk

"""convergenta metodei GPS LS"""

def fixed_point_GPS_LS_afisare_convergenta(xk, ai, di, pasi_acuratete, x_true):

    plt.title("Convergenta metodei GPS_LS pentru o rulare la intamplare")
    lista_pt_plot = list()
    for i in range(pasi_acuratete):
        xk = T(xk, ai, di)
        lista_pt_plot.append(np.linalg.norm(xk - x_true) )

    plt.plot(range(pasi_acuratete) , lista_pt_plot, linewidth = 3, label = "evolutia diferentei fata de x-ul real", color = 'magenta')
    plt.legend()

def fixed_point_GPS_LS_histograma_erorilor(x, a,  pasi_acuratete, x_true):
    generari = 10**3
    B = list()

    for i in range(generari):
        #x_true = [random.random() * (10 - (-10)) + -10 , random.random() * (10 - (-10)) + -10 ]
        #x_true = np.array(x_true)

        d = generare_di(x_true, a)
        B.append(np.linalg.norm(fixed_point_GPS_LS(x, a, d, pasi_acuratete) - x_true))


    
    plt.hist(B, bins = 13, ec='black', color='magenta')
    plt.title("Histograma erorilor GPS_LS")

def fixed_point_GPS_LS_histograma_erorilor_influentat_de_x_initial(x, a,d,  pasi_acuratete, x_true, INTERVAL_STANGA, INTERVAL_DREAPTA, n):
    generari = 10**3
    B = list()

    for i in range(generari):
        B.append(np.linalg.norm(fixed_point_GPS_LS(x, a, d, pasi_acuratete) - x_true))
        x = np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)])


    
    plt.hist(B, bins = 13, ec='black', color='magenta')
    plt.title("Histograma erorilor GPS_LS in functie de x-ul initial")

"""
Fixed point method for solving (CF LS) : general step"""

def CF_LS_step(xk, ai, di):
    m = len(ai)
    vec = [np.linalg.norm(xk - a) for a in ai]


    vec = [elem / elemvec for elem, elemvec in zip(xk-ai, vec)]

    vec = sum(vec)

    vec = 1/m * vec

    vec = np.floor(vec)

    vec = vec * r(xk, ai, di)

    vec = vec + sum(ai) * 1/m
   
    return vec

"""Circle fitting least squares """

def fixed_point_CF_LS(xk, ai , di, pasi_acuratete):
    
    for i in range(pasi_acuratete):

        xk = CF_LS_step(xk, ai, di)
    
    return xk

def verificare_asumption_matrice(a):
    for elem in a:
        for elem2 in (np.transpose(elem) * 2 - 1):
            if elem2 == 0 :
                return False
    
    return True

"""ilustrarea metodei GPS LS pentru datele oferite in exemplul 5.2 din lucrare (pagina 23)"""

def exemplul_5_2():

    nr_figura = 0

    x = np.array([-10, 5])
    a = np.array([[-29, -18], [7, -24], [-19, -27], [10, -27], [-9, 3], [-33, -34]])
    x_true = np.array([-8, -2])
    pasi_acuratete = 10**2
    d = generare_di(x_true, a)
    

 

    plt.figure(nr_figura)
    nr_figura += 1
    fixed_point_GPS_LS_afisare_convergenta(x, a, d, pasi_acuratete, x_true)
    plt.figure(nr_figura)
    nr_figura += 1
    fixed_point_GPS_LS_histograma_erorilor(x, a, pasi_acuratete, x_true)




    a = np.linspace(-30, 20, 10**1)
    b = np.linspace(-30, 20, 10**1)

    x, y = np.meshgrid(a,b)

    z = list()
    for liniex, liniey in zip(x,y):
        aux = list()
        for elemx, elemy in zip (liniex, liniey):
            aux.append([elemx, elemy])
        
        daux = generare_di(x_true, aux)

        z.append(daux)
    z= np.array(z)


    plt.figure(nr_figura)
    nr_figura += 1
    ax = plt.axes(projection = '3d')
    plt.contour(x,y,z, 50)
    ax.scatter(x_true[0], x_true[1], 0, color='red', label='Punctul de minim x_true')
    plt.title("Functia care trebuie minimizata GPS_LS")
    plt.legend()
    plt.show()

"""ilustrarea metodei CF LS pentru datele oferite in exemplul 5.3 din lucruare(pagina 24)"""

def exemplul_5_3():
    x = np.array([3, -7])
    a = np.array([[1, 9], [ 2, 7], [5, 8], [7, 7], [9, 5], [3, 7]])
    x_true = np.array([0, -8])
    pasi_acuratete = 5
    d = generare_di(x_true, a)
    #d = [0 for i in a]
    #d = np.array(d)


    fig, axes = plt.subplots()
    axes.set_aspect(1)

    axes.set_xlim([-50, 50])
    axes.set_ylim([-50, 50])

    for satelit in a:
        plt.plot(satelit[0], satelit[1], 'ro')


    x_gasit_optim = fixed_point_CF_LS(x, a, d, pasi_acuratete)


    plt.plot(x_gasit_optim[0], x_gasit_optim[1],'bo',  label = 'Centrul cercului gasit prin CF_LS' ,markersize = 3)
    plt.plot(x_true[0], x_true[1], 'mo', label = 'Centrul real al cercului', markersize = 5)

    draw_circle = plt.Circle((x_gasit_optim[0], x_gasit_optim[1]), r(x_gasit_optim, a, d),fill=False, lw = 3)
    axes.add_artist(draw_circle)

    plt.title('Circle fitting LS\n ' + 'distanta fata de punctul de optim :' + str( np.linalg.norm( fixed_point_CF_LS(x, a, d, pasi_acuratete)- x_true)))

    plt.legend()
    plt.show()

"""metoda GPS LS : alegem random coordonatele ai , distantele di si punctul initial x"""

def fixed_point_GPS_LS_random():
    nr_figura = 0

    NR_MAXIM_DIMENSIUNI = 10
    NR_MAXIM_SATELITI = 20
    INTERVAL_STANGA = -10
    INTERVAL_DREAPTA = 10

    n = random.randint(2, NR_MAXIM_DIMENSIUNI)

    nr_sateliti = 0

    while nr_sateliti < n+1:
        nr_sateliti = random.randint(0,NR_MAXIM_SATELITI)

    x = np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)])

    a = list()
    for elem in range(nr_sateliti):
        a.append(np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)]))
    
    a= np.array(a)

    x_true = np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)])
    pasi_acuratete = 10**2
    d = generare_di(x_true, a)

    


    plt.figure(nr_figura)
    fixed_point_GPS_LS_histograma_erorilor(x, a, pasi_acuratete, x_true)
    nr_figura +=1
    plt.figure(nr_figura)
    fixed_point_GPS_LS_afisare_convergenta(x, a, d, pasi_acuratete, x_true)

    print("################REZULTATE PENTRU SPATIU SI SATELITI GENERATI RANDOM#############")
    print("Nr sateliti: " + str(nr_sateliti))
    print("Spatiul n = " + str(n))
    
    plt.legend()
    plt.show()

    ################## aceleasi date ai,di si x0 optim 



    x = find_x0(a,d)


    plt.figure(nr_figura)
    fixed_point_GPS_LS_histograma_erorilor(x, a, pasi_acuratete, x_true)
    nr_figura +=1
    plt.figure(nr_figura)
    fixed_point_GPS_LS_afisare_convergenta(x, a, d, pasi_acuratete, x_true)

    print("################REZULTATE PENTRU SPATIU SI SATELITI GENERATI RANDOM si X_0 OPTIM#############")
    print("Nr sateliti: " + str(nr_sateliti))
    print("Spatiul n = " + str(n))
    
    plt.legend()
    plt.show()

"""metoda GPS LS : alegem random coordonatele ai si distantele di si gasim punctul initial x optim """

def fixed_point_GPS_LS_random_optim():
    nr_figura = 0

    NR_MAXIM_DIMENSIUNI = 10
    NR_MAXIM_SATELITI = 20
    INTERVAL_STANGA = -10
    INTERVAL_DREAPTA = 10

    n = random.randint(2, NR_MAXIM_DIMENSIUNI)

    nr_sateliti = 0

    while nr_sateliti < n+1:
        nr_sateliti = random.randint(0,NR_MAXIM_SATELITI)

    #x = np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)])

    a = list()
    for elem in range(nr_sateliti):
        a.append(np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)]))
    
    a= np.array(a)

    x_true = np.array([random.random() * (INTERVAL_DREAPTA - INTERVAL_STANGA) + INTERVAL_STANGA for i in range(n)])
    pasi_acuratete = 10**2
    d = generare_di(x_true, a)

    x = find_x0(a,d)


    plt.figure(nr_figura)
    fixed_point_GPS_LS_histograma_erorilor(x, a, pasi_acuratete, x_true)
    nr_figura +=1
    plt.figure(nr_figura)
    fixed_point_GPS_LS_afisare_convergenta(x, a, d, pasi_acuratete, x_true)

    print("################REZULTATE PENTRU SPATIU SI SATELITI GENERATI RANDOM si X_0 OPTIM#############")
    print("Nr sateliti: " + str(nr_sateliti))
    print("Spatiul n = " + str(n))
    
    plt.legend()
    plt.show()



if __name__ == '__main__':

    exemplul_5_2()
    exemplul_5_3()
    fixed_point_GPS_LS_random()
    #fixed_point_GPS_LS_random_optim()
    fixed_point_GPS_LS_random_influenta_punctului_de_start()

