import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import time
def jacobi(S, D, L, r, U, tol, k):
    start = time.time()
    D_i = np.linalg.inv(D)
    Uold = U.copy()
    error = np.infty
    i = 0
    while error > tol and i < k:
        i += 1
        U = np.dot(np.dot(-1*D_i, L + S), Uold) + np.dot(D_i, r)
        error = np.linalg.norm(U - Uold)
#         print(error)
        Uold = U.copy()
    end = time.time()
    return (U, i, error, end-start)


def gauss_seidel(S, D, L, r, U, tol, k):
    start = time.time()
    L_D_i = np.linalg.inv(L+D)
    Gs = np.dot(-1*L_D_i, S)
    D_i = np.linalg.inv(D)

    t = np.dot(D_i, r)

    Uold = U.copy()
    error = np.infty
    i = 0
    while error > tol and i < k:
        i += 1
        U = np.dot(Gs, Uold) + np.dot(np.dot(L_D_i, D), t)
        error = np.linalg.norm(U - Uold)
#         print(error)
        Uold = U.copy()
    end = time.time()
    return (U, i, error, end-start)

def SOR(S, D, L, r, U, w, tol, k):
    start = time.time()
    D_L_i = np.linalg.inv(D+w*L)
    Gsor = np.dot(D_L_i, (1-w)*D - w*S)
    D_i = np.linalg.inv(D)
#     t = np.dot(Di, r)

    Uold = U.copy()
    error = np.infty
    i = 0
    while error > tol and i < k:
        i += 1
        U = np.dot(Gsor, Uold) + np.dot(w*D_L_i, r)
        error = np.linalg.norm(U - Uold)
#         print(error)
        Uold = U.copy()
    end = time.time()
    return (U, i, error, end-start)

def gera_ticks(l, ini, end):
    ticks = ["" for x in range(l)]
    elementos = np.linspace(ini, end, int((end-ini)/0.2 + 1))
    # print(len(elementos))
    # print(len(ticks))
    # print(ticks)
    # print(elementos)
    ii = 0
    for i in range(l):
        if i % int(np.around(len(ticks)/len(elementos))) == 0:
            if ii >= len(elementos):
                break
            ticks[i] = elementos[ii].round(2)
            ii +=1
    # print(ticks)
    return ticks

def print_res(u, M, N, i, a, b, h, k, metodo="", print_annot=False, title=False):
    x_ticks = gera_ticks(int((a-i)/h + 1), i, a)
    y_ticks = gera_ticks(int((a-i)/k + 1), i, b)
    # y_ticks = np.linspace(i, b, int((b-i)/k + 1)).round(2)
    # x_ticks = np.linspace(i, a, int((a-i)/0.2 + 1)).round(2)
    # y_ticks = np.linspace(i, b, int((b-i)/0.2 + 1)).round(2)
    # print(y_ticks)
    if title:
        title = 'Solução Numérica '+ metodo + ' h=' + str(h) + ' k=' + str(k)
    else:
        title = metodo + ' h=' + str(h) + ' k=' + str(k)

    plt.figure(dpi=200, figsize=(8, 6))
    plt.title(title)
    if print_annot:
        text = [ [ None for y in range( N+1 ) ] for x in range( M+1 ) ]
        for i in range(N+1):
            for j in range(M+1):
             text[i][j] = "U{},{}".format(i, j)
    else:
        text = False
    ax = sns.heatmap(u, cmap='jet', annot=text, fmt ='', xticklabels =1, yticklabels =1,cbar_kws={'label': 'Temperatura (ºC)'})
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    ax.invert_yaxis()
    plt.show()



# "(i="+str(i)+", j="+str(i)+") U_{"+str(i)+",1} + U_"+str(i)+","+str(i)+"} -4U_{"+str(i)+","+str(i)+"} + U_{"+str(i)+","+str(i)+"} + U_{"+str(i)+","+str(i)+"} = 0]\\\\"