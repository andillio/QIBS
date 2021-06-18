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
import sys
import yt; yt.enable_parallelism()

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
        
        self.decimate = None
fo = figObj()


def str2sig(string):
    string = string.replace('(','')
    string = string.replace(')','')
    ints = string.split(",")
    return (int(ints[0]), int(ints[1]) )


def setFigObj(name, decimate):
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

    fo.decimate = decimate

    np.random.seed(1)
    fo.phi = np.random.uniform(0, 2 * np.pi, fo.N)

    # this is basically just to see how many time drops there were
    fo.fileNames_psi = u.getNamesInds('Data/' + name + "/" + "psi" + fo.tags[0])


def offDiag(sig, c_j, state_j, i):

    M = np.zeros((fo.N, fo.N)) + 0j

    # creation op on mode b
    for b in range(fo.N):  
        # annihilation op on mode a 
        for a in range(b+1, fo.N):

            state_i = state_j.copy()
            state_i[a] = state_j[a] - 1
            state_i[b] = state_j[b] + 1

            new_p = sig[1] + b - a
            newsig = (sig[0], new_p)

            tag_ = str(newsig)

            if tag_ in fo.tags:
                indToTuple_ = None 
                tupleToInd = None

                with open("../Data/" + fo.name + "/" + "indToTuple" + tag_ + ".pkl", 'rb') as f:    
                    indToTuple_ = pickle.load(f)
                with open("../Data/" + fo.name + "/" + "tupleToInd" + tag_ + ".pkl", 'rb') as f:    
                    tupleToInd_ = pickle.load(f)

                # load in the psi in a given special Hilbert space
                fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + tag_)
                psi_ = np.load(fileNames_psi[i])

                if tuple(state_i) in tupleToInd_:
                    i_ = tupleToInd_[ tuple(state_i) ]
                    val_ = c_j
                    val_ *= np.sqrt(state_j[b] + 1)
                    val_ *= np.sqrt(state_j[a])
                    val_ *= np.conj(psi_[i_])

                    M[b,a] += val_ 
                    M[a,b] += np.conj(val_)
    
    return M
                

def get_aa(sig, c_j, state_j, i):

    aa = np.zeros((fo.N, fo.N)) + 0j

    for a1 in range(fo.N):
        for a2 in range(a1,fo.N):
            state_i = state_j.copy()
            state_i[a1] = state_i[a1] - 1
            state_i[a2] = state_i[a2] - 1            

            new_p = sig[1] - a1 - a2
            newsig = (sig[0] - 2, new_p)

            newTag = str(newsig)

            if newTag in fo.tags:
                indToTuple_ = None 
                tupleToInd = None

                with open("../Data/" + fo.name + "/" + "indToTuple" + newTag + ".pkl", 'rb') as f:    
                    indToTuple_ = pickle.load(f)
                with open("../Data/" + fo.name + "/" + "tupleToInd" + newTag + ".pkl", 'rb') as f:    
                    tupleToInd_ = pickle.load(f)

                # load in the psi in a given special Hilbert space
                fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + newTag)
                psi_ = np.load(fileNames_psi[i])

                if tuple(state_i) in tupleToInd_:
                    i_ = tupleToInd_[ tuple(state_i) ]
                    val_ = c_j
                    if a1 != a2:
                        val_ *= np.sqrt(state_j[a1])
                        val_ *= np.sqrt(state_j[a2])
                        val_ *= np.conj(psi_[i_])

                        aa[a1,a2] += val_ 
                        aa[a1,a2] += val_

                    else:
                        val_ *= np.sqrt(state_j[a1])
                        val_ *= np.sqrt(state_j[a2]-1)
                        val_ *= np.conj(psi_[i_])

                        aa[a1,a2] += val_ 
    return aa


def get_a(sig, c_j, state_j, i):

    a = np.zeros(fo.N) + 0j

    for a1 in range(fo.N):
        state_i = state_j.copy()
        state_i[a1] = state_i[a1] - 1

        new_p = sig[1] - a1
        newsig = (sig[0] - 1, new_p)

        newTag = str(newsig)
    
        if newTag in fo.tags:
            indToTuple_ = None 
            tupleToInd = None

            with open("../Data/" + fo.name + "/" + "indToTuple" + newTag + ".pkl", 'rb') as f:    
                indToTuple_ = pickle.load(f)
            with open("../Data/" + fo.name + "/" + "tupleToInd" + newTag + ".pkl", 'rb') as f:    
                tupleToInd_ = pickle.load(f)

            # load in the psi in a given special Hilbert space
            fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + newTag)
            psi_ = np.load(fileNames_psi[i])

            if tuple(state_i) in tupleToInd_:
                i_ = tupleToInd_[ tuple(state_i) ]
                val_ = c_j
                val_ *= np.sqrt(state_j[a1])
                val_ *= np.conj(psi_[i_])
                a[a1] += val_

    return a 

def analyzeTimeStep(i):

    t = fo.dt*fo.framsteps*(i+1)
    outputs = {}

    for sto, tag_ in yt.parallel_objects( fo.tags , 0, storage=outputs):
        sys.stdout.flush()
        sto.result_id = tag_
        sig = str2sig(tag_)

        # load in the psi in a given special Hilbert space
        fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + tag_)
        psi_ = np.load(fileNames_psi[i])

        indToTuple_ = None 

        with open("../Data/" + fo.name + "/" + "indToTuple" + tag_ + ".pkl", 'rb') as f:    
            indToTuple_ = pickle.load(f)

        N_ = np.zeros(fo.N)
        M_ = np.zeros((fo.N, fo.N)) + 0j
        aa_ = np.zeros((fo.N, fo.N)) + 0j
        a_ = np.zeros(fo.N) + 0j

        #print len(psi_), len(indToTuple_), tag_

        for j in range(len(indToTuple_)):
            subState = np.array(indToTuple_[j])

            c_j = psi_[j]

            if np.abs(c_j)>0:

                N__ = subState*np.abs(c_j)**2
                N_ += N__

                M_ += np.diag(N__)
                M_ += offDiag(sig, c_j, subState, i)

                aa_ += get_aa(sig, c_j, subState, i)

                a_ += get_a(sig, c_j, subState, i)

        sto.result = (N_, M_, aa_, a_)
    
    N = np.zeros(fo.N)
    M = np.zeros((fo.N, fo.N)) + 0j
    aa = np.zeros((fo.N, fo.N)) + 0j
    a = np.zeros(fo.N) + 0j


    for i in range(len(outputs.keys())):

        key_ = outputs.keys()[i]
        N_, M_, aa_, a_ = outputs[key_]

        N += N_ 
        M += M_
        aa += aa_
        a += a_

    eigs, _ = LA.eig(M)
    eigs = qu.sortE(np.abs(eigs),eigs)

    Q = np.sum( N - a*np.conj(a) ) / np.sum(fo.IC)

    return t, N, M, eigs, aa, a, Q


def analyze():
    print("Starting di_analysis...")

    time0 = time.time()

    t, N, M, eigs, aa, a, Q = [], [], [], [], [], [], []

    steps = len(range(0,len(fo.fileNames_psi), fo.decimate))

    for i in range(0,len(fo.fileNames_psi), fo.decimate):
        t_, N_, M_, eigs_, aa_, a_, Q_ = analyzeTimeStep(i)
        t.append(t_)
        N.append(N_)
        M.append(M_)
        eigs.append(eigs_)
        aa.append(aa_)
        a.append(a_)
        Q.append(Q_)
        u.repeat_print(('%i hrs, %i mins, %i s remaining.' %u.remaining(i + 1, steps, time0)))

    t = np.array(t)
    N = np.array(N)
    M = np.array(M)
    eigs = np.array(eigs)
    aa = np.array(aa)
    a = np.array(a)
    Q = np.array(Q)

    return t, N, M, eigs, aa, a, Q


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
    Num = qu.sortE(t, Num)
    M = qu.sortE(t, M)
    eigs = qu.sortE(t, eigs)
    aa = qu.sortE(t, aa)
    a = qu.sortE(t, a)
    Q = qu.sortE(t,Q)
    t = qu.sortE(t,t)

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

    setFigObj(name, decimate)

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
