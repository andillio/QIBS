import utils as u
import QUtils as qu 
import numpy as np
from numpy import linalg as LA 
import matplotlib.pyplot as plt 

simName = "Attr_r12"


class figObj(object):

    def __init__(self):
        self.N = None 
        self.name = None 
        self.n = None 
fo = figObj()

def makeNFig(t, N):
    fig, ax = plt.subplots(figsize = (6,6))

    ax.set_xlabel(r'$t$')

    for i in range(fo.N):
        ax.plot(t, N[:,i], label = r'$E[\hat N_%i]$' %i)

    ax.set_xlim(0, np.max(t) )
    ax.set_ylim(0, np.max(N)*1.05 )

    ax.legend()

    fig.savefig("../Figs/" + fo.name + "_Num.pdf",bbox_inches = 'tight')

def makeMFig(t, lams):
    fig, ax = plt.subplots(figsize = (6,6))

    ax.set_xlabel(r'$t$')

    for i in range(fo.N):
        ax.plot(t, lams[:,i], label = r'$\lambda_%i$' %i)

    ax.set_xlim(0, np.max(t) )
    ax.set_ylim(0, np.max(lams)*1.05 )

    ax.legend()

    fig.savefig("../Figs/" + fo.name + "_lams.pdf",bbox_inches = 'tight')


def makePOQFig(t, eigs, Q):
    fig, ax = plt.subplots(figsize = (6,6))

    n = np.sum(fo.n)

    ax.set_xlabel(r'$t$')

    PO = 1. - (eigs[:,-1] / n)

    ax.plot(t, PO, label = r'$1 - \lambda_p/n_{tot}$')
    ax.plot(t, Q, label = r'$Q$')

    ax.set_xlim(0, np.max(t) )
    ax.set_ylim(0, 1.05)

    ax.legend()

    fig.savefig("../Figs/" + fo.name + "_POQ.pdf",bbox_inches = 'tight')


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
            xi_k = np.conj(xi_p[i,k])
            #xi_k = xi_p[i,k]

            aS[i] += xi_k*a[i,k]

            for j in range(N):
                j_ = (-1*j -1)%N

                xi_j = np.conj(xi_p[i,j])
                #xi_j = xi_p[i,j]

                aaS[i] += xi_k*xi_j*aa[i,k,j]
                baS[i] += np.conj(xi_k)*xi_j*M[i,k,j]

    dbaS = baS - np.conj(aS)*aS
    daaS = aaS - aS*aS

    return 1 + 2*dbaS - 2*np.abs(daaS)


def makeSqueezeFig(t, aa, M, a):
    sq = constructSq(a, aa, M)

    fig, ax = plt.subplots(figsize = (6,6))

    ax.set_xlabel(r'$t$')

    ax.plot(t, sq)

    ax.text(.5,.9,r'$1 + 2 E[\delta \hat a_S^\dagger \delta \hat a_S ] - 2 |Var[\hat a_S]|$', ha='center', va='center', transform= ax.transAxes, 
        bbox = {'facecolor': 'white', 'pad': 5})

    ax.plot([0, np.max(t)], [1,1], "r:")

    r_pred = np.log(fo.n**(1/6.))
    t_pred = .6/(5*.1)

    ax.plot([t_pred], [np.exp(-2*r_pred)], 'ko')

    index = np.argmin(sq)

    ax.plot([t[index]], [sq[index]], 'bo')

    if len(t[sq<2]):
        ax.set_xlim(0, np.max(t[sq<2]) )
    else:
        ax.set_xlim(0, np.max(t) )
    ax.set_ylim(0,2.)

    ax.legend(loc = 'lower right')

    fig.savefig("../Figs/" + fo.name + "_Sq.pdf",bbox_inches = 'tight')


def main(simName, *args, **kwargs):

    t = np.load("../Data/" + simName + "/_t.npy")
    N = np.load("../Data/" + simName + "/_N.npy")
    M = np.load("../Data/" + simName + "/_M.npy")
    eigs = np.load("../Data/" + simName + "/_eigs.npy")
    aa = np.load("../Data/" + simName + "/_aa.npy")
    a = np.load("../Data/" + simName + "/_a.npy")
    Q = np.load("../Data/" + simName + "/_Q.npy")

    N = qu.sortE(t, N)
    M = qu.sortE(t, M)
    eigs = qu.sortE(t, eigs)
    aa = qu.sortE(t, aa)
    a = qu.sortE(t, a)
    Q = qu.sortE(t,Q)
    t = qu.sortE(t,t)

    fo.N = len(N[0])
    fo.name = simName
    fo.n = np.real(np.sum(N[0]))

    print(fo.n)
    print(np.real(np.sum(N[-1])))

    u.orientPlot()
    makeNFig(t, N)
    makeMFig(t, eigs)
    makePOQFig(t, eigs, Q)
    makeSqueezeFig(t, aa, M, a)


if __name__ == "__main__":
    main(simName)