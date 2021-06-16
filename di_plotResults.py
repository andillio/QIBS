import utils as u
import QUtils as qu 
import numpy as np
import matplotlib.pyplot as plt 

simName = "test_r5"


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


def makeSqueezeFig(t, aa, N, a):
    er = np.zeros((len(t), fo.N))

    for i in range(len(t)):
        t_ = t[i]
        aa_ = np.diag(aa[i])
        N_ = N[i]
        a_ = a[i]

        dbda = N_ - a_*np.conj(a_)
        dada = aa_ - a_*a_

        er_ = 2*dbda - 2*np.abs(dada) + 1.
        er[i,:] = er_

    fig, ax = plt.subplots(figsize = (6,6))

    ax.set_xlabel(r'$t$')

    for i in range(fo.N):
        ax.plot(t, er[:,i], label = r'$exp(-2r_%i)$'%i)

    ax.text(.5,.9,r'$1 + 2 E[\delta \hat a^\dagger_i \delta \hat a_i ] - 2 |Var[\hat a_i]|$', ha='center', va='center', transform= ax.transAxes, 
        bbox = {'facecolor': 'white', 'pad': 5})

    ax.set_xlim(0, np.max(t) )

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
    makeSqueezeFig(t, aa, N, a)


if __name__ == "__main__":
    main(simName)