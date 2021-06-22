import matplotlib.pyplot as plt 
import numpy as np 
import utils as u
import QUtils as qu
from numpy import linalg as LA 

simNames = ["Gr_r1","Gr_r2","Gr_r3","Gr_r4","Gr_r5", "Gr_r6", "Gr_r7", 
"Gr_r8", "Gr_r9", "Gr_r10"]
#simNames = ["Sin_h1_r1"]
t_dyn = 1./np.sqrt(.1*5)

class figObj(object):

    def __init__(self):
        self.fig = None 
        self.ax1 = None 
        self.ax2 = None 
fo = figObj() 

def makeFig():
    fo.fig, axs = plt.subplots(2,1,figsize = (6,12))
    fo.ax1 = axs[0]
    fo.ax2 = axs[1]

    fo.ax1.set_xticklabels([])

    fo.ax2.set_xlabel(r"$n_{tot}$")
    fo.ax1.set_ylabel(r"$t_{br} \, [ t_{d}]$")
    fo.ax2.set_ylabel(r"$r_{max}$")


def constructSq(a,aa,M):

    N = len(a[0])
    n = np.sum(np.diag(M[0]))

    xi_p = np.zeros( (len(a), N) ) + 0j
    aaS = np.zeros( len(a) ) + 0j
    baS = np.zeros( len(a) ) + 0j
    aS = np.zeros( len(a) ) + 0j

    for i in range(len(a)):
        M_ = M[i]
        eigs, psis = LA.eig(M_)
        psis = qu.sortVects(np.abs(eigs),psis)
        eigs = qu.sortE(np.abs(eigs),eigs)
        principle = psis[:,-1]
        xi_p[i,:] = principle#*np.sqrt(eigs[-1])
    
        for k in range(N):
            
            k_ = (-1*k -1)%N
            #xi_k = np.conj(xi_p[i,k_])
            xi_k = xi_p[i,k]

            aS[i] += xi_k*a[i,k]

            for j in range(N):
                j_ = (-1*j -1)%N

                #xi_j = np.conj(xi_p[i,j_])
                xi_j = xi_p[i,j]

                aaS[i] += xi_k*xi_j*aa[i,k,j]
                baS[i] += np.conj(xi_k)*xi_j*M[i,k,j]

    dbaS = baS - np.conj(aS)*aS
    daaS = aaS - aS*aS

    return 1 + 2*dbaS - 2*np.abs(daaS)


def load(name, t):
    rval = np.load(name)
    return qu.sortE(t,rval)

def PlotStuff(simName, color, ax, ax2):

    t = np.load("../Data/" + simName + "/_t.npy")
    aa = load("../Data/" + simName + "/_aa.npy",t)
    a = load("../Data/" + simName + "/_a.npy",t)
    N = load("../Data/" + simName + "/_N.npy",t)
    M = load("../Data/" + simName + "/_M.npy",t)
    n = np.abs(np.sum(N[0]))
    t = qu.sortE(t,t)

    er = np.zeros((len(t), len(N[0])))

    for i in range(len(t)):
        t_ = t[i]
        aa_ = np.diag(aa[i])
        N_ = N[i]
        a_ = a[i]

        dbda = N_ - a_*np.conj(a_)
        dada = aa_ - a_*a_

        er_ = 2*dbda - 2*np.abs(dada) + 1.
        
        er[i,:] = er_

    er = qu.sortE(t,er)
    N = qu.sortE(t,N)
    t = qu.sortE(t,t)
    sq = constructSq(a,aa,M) #er[:,2]#np.sum(er*N, axis = 1)/n

    #plt.plot(t,sq)
    #plt.plot([np.min(t), np.max(t)], [1,1])
    #plt.plot([np.min(t), np.max(t)], [0,0])
    #plt.show()

    argMin = np.argmin(sq)
    t_br = t[argMin]
    sq_min = sq[argMin]
    r_max = np.log(sq_min)/-2.

    ax.plot([n],[t_br],color)
    ax2.plot([n],[r_max],color)
    #ax2.plot([n], [np.log(n**(1/6.))], 'ko')

    return t_br, n, t_dyn, np.log(n**(1/6.))


def PlotList(names,t_br, n, color, label, ax, ax2):

    t_br_, N = [], []
    rMax = []

    t_dyn = 0.
    for i in range(len(names)):
        t_, n_, t_dyn, r_max = PlotStuff(names[i], color, ax, ax2)
        t_br.append(t_)
        t_br_.append(t_)
        n.append(n_)
        N.append(n_)
        rMax.append(r_max)

    N = np.array(N)
    t_br_ = np.array(t_br_)

    ax2.plot(N, rMax, 'k--')
    ax.plot([],[],color,label = label)
    ax.legend()

    return t_dyn


if __name__ == "__main__":
    u.orientPlot()
    makeFig()

    t_br = []
    n = []

    PlotList(simNames,t_br, n, 'bo', r'CoherentState', fo.ax1, fo.ax2)

    t_sq = .6/(5*.1)
    #r_max = np.log(n_max**(1./6))
    #y_min = np.exp(-2*r_max)
    fo.ax1.set_ylim(0, np.max(t_sq)*2)
    fo.ax1.plot([0.,np.max(n)*1.05],[t_sq, t_sq], 'k--', alpha = 1.)

    fo.ax1.set_xlim(0,np.max(n)*1.05)
    fo.ax2.set_xlim(0,np.max(n)*1.05)

    plt.subplots_adjust(wspace=.0, hspace=0.)
    plt.savefig("../Figs/breaktimesSq.pdf", bbox_inches = "tight")
    plt.show()