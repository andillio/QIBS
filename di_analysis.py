import utils as u
import QUtils as qu 
import numpy as np
import multiprocessing as mp 
import scipy.fftpack as sp
import time
import matplotlib.pyplot as plt
import pickle 
from numpy import linalg as LA 
import scipy.stats as st 

simName = "FNS_r1"
decimate = 2
label = ""
PLOT = True

class figObj(object):

    def __init__(self):
        self.meta = None

        self.tags = None 

        self.N = None 
        self.dt = None 
        self.framesteps = None 
        self.IC = None 
        self.phi = None 

        self.name = None 
        self.fileNames_psi = None

        self.indToTuple = None 
        self.tupleToInd = None    
fo = figObj()


def GetOffDiag(psi):
    
    M = np.zeros((fo.N, fo.N)) + 0j

    time0 = time.time()

    for j in range(len(fo.indToTuple)):
        state_j = np.array(fo.indToTuple[j])
        
        if np.abs(psi[j]) > 0:

            for b in range(fo.N):

                for a in range(b+1, fo.N):
                    state_i = state_j.copy()
                    state_i[a] = state_j[a] - 1
                    state_i[b] = state_j[b] + 1

                    if tuple(state_i) in fo.tupleToInd:
                        
                        i_ = fo.tupleToInd[ tuple(state_i) ]
                        val_ = psi[j]
                        val_ *= np.sqrt(state_j[b] + 1)
                        val_ *= np.sqrt(state_j[a])
                        val_ *= np.conj(psi[i_])

                        M[b,a] += val_ 
                        M[a,b] += np.conj(val_)
    return M


def Getaa(psi):
    aa = np.zeros((fo.N, fo.N)) + 0j

    a = np.zeros(fo.N) + 0j

    for i in range(fo.N):
        a[i] += qu.calcOp(psi, fo, a = [i])
        for j in range(i, fo.N):
            aa_ = qu.calcOp(psi, fo, a = [i,j])
            aa[i,j] += aa_ 
            if i != j:
                aa[j,i] += aa_
    
    return aa, a


def analyzeTimeStep(i):
    psi, N = qu.GetPsiAndN(i, fo)

    # time stamp
    t = fo.dt*fo.framsteps*(i+1)

    # calculate M hat
    M = np.zeros((fo.N, fo.N)) + 0j
    M += np.diag(N)
    M += GetOffDiag(psi)

    # calculate lam
    eigs, _ = LA.eig(M)
    eigs = qu.sortE(np.abs(eigs),eigs)

    # get aa op
    aa, a = Getaa(psi)

    # get Q param
    Q = np.sum( N - a*np.conj(a) ) / np.sum(fo.IC)

    return t, N, M, eigs, aa, a, Q


def analyze():
    print("Starting di_analysis...")
    pool = mp.Pool(mp.cpu_count())
    outputs = pool.map(analyzeTimeStep, range(0, len(fo.fileNames_psi), decimate) )
    print("Data analyzed...")

    n_out = len(outputs)

    t = np.zeros(n_out)
    N = np.zeros((n_out,fo.N)) + 0j
    M = np.zeros((n_out, fo.N, fo.N)) + 0j
    eigs = np.zeros((n_out,fo.N)) + 0j
    aa = np.zeros((n_out, fo.N, fo.N)) + 0j
    a = np.zeros((n_out,fo.N)) + 0j
    Q = np.zeros(n_out) + 0j

    for i in range(n_out):
        t_, N_, M_, eigs_, aa_, a_, Q_ = outputs[i]
        t[i] = t_
        N[i] = N_ 
        M[i] = M_ 
        eigs[i] = eigs_
        aa[i] = aa_ 
        a[i] = a_ 
        Q[i] = Q_ 

    return t, N, M, eigs, aa, a, Q


def setFigObj(name):
    # read in simulation parameters
    meta = u.getMetaKno(name, dir = 'Data/', N = "N", dt = "dt", frames = "frames", 
        framesteps = "framesteps", IC = "IC", omega0 = "omega0", Lambda0 = "Lambda0")
    fo.meta = meta

    # sets the figure object with these parameters
    # this is basically just so I can access them in the glocal scope
    fo.name = name

    fo.N = fo.meta["N"]
    fo.dt = fo.meta["dt"]
    fo.framsteps = fo.meta["framesteps"]
    fo.IC =  fo.meta["IC"]

    np.random.seed(1)
    fo.phi = np.random.uniform(0, 2 * np.pi, fo.N)

    # this is basically just to see how many time drops there were
    fo.fileNames_psi = u.getNamesInds('Data/' + name + "/" + "psi" + fo.tags[0])

    qu.GetDicts(fo)


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

    n = np.sum(fo.IC)

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


def SaveStuff(t, Num, M, eigs, aa, a, Q):
    np.save("../Data/" + fo.name + "/" + "_t" + ".npy", t)
    np.save("../Data/" + fo.name + "/" + "_N" + ".npy", Num)
    np.save("../Data/" + fo.name + "/" + "_M" + ".npy", M)
    np.save("../Data/" + fo.name + "/" + "_eigs" + ".npy", eigs)
    np.save("../Data/" + fo.name + "/" + "_aa" + ".npy", aa)
    np.save("../Data/" + fo.name + "/" + "_a" + ".npy", a)
    np.save("../Data/" + fo.name + "/" + "_Q" + ".npy", Q)


def main(name, tags = [], label = "", decimate = 1, plot = PLOT):
    time0 = time.time()

    fo.tags = tags

    setFigObj(name)

    t, N, M, eigs, aa, a, Q = analyze()
    
    SaveStuff(t, N, M, eigs, aa, a, Q)

    if plot:
        u.orientPlot()
        makeNFig(t, N)
        makeMFig(t, eigs)
        makePOQFig(t, eigs, Q)
        makeSqueezeFig(t, aa, N, a)

    print('completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))

if __name__ == "__main__":
    # load in the tags on the data directories
    try:
        fo.tags = np.load("../Data/" + simName + "/tags.npy")
    except IOError:
        fo.tags = [""]
    main(simName, fo.tags)