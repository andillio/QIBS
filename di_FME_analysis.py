import utils as u
import numpy as np
import QUtils as qu 
import multiprocessing as mp 
import scipy.fftpack as sp
import time
import matplotlib.pyplot as plt
import pickle 
from numpy import linalg as LA 
import scipy.stats as st 


simName = "Sin_h1_r1"
label = ""
PLOT = True

class figObj(object):

    def __init__(self):
        self.meta = None

        self.N = None 
        self.dt = None 
        self.framesteps = None 
        self.Norm = None 
        self.phi = None 

        self.name = None 
fo = figObj()

def get_aVals(approxName, tag):
    meta = u.getMetaKno(approxName,dir = 'Data/', N = "N", dt = "dt", frames = "frames", 
        framesteps = "framesteps", IC = "IC", omega0 = "omega0", Lambda0 = "Lambda0")
    fileNames_a = u.getNamesInds("Data/" + approxName + "/" + "a" + tag)

    t = np.zeros(len(fileNames_a))
    a = np.zeros((len(fileNames_a), fo.N )) + 0j

    dt = meta["dt"]
    framesteps = meta["framesteps"]

    for i in range(len(fileNames_a)):
        a_ = np.load(fileNames_a[i])

        t[i] = dt*framesteps*(i+1)

        for j in range(len(a[0])):
            a[i,j] = a_[j] 

    return t, a

def get_fmeVals(approxName, tag):
    meta = u.getMetaKno(approxName,dir = 'Data/', N = "N", dt = "dt", frames = "frames", 
        framesteps = "framesteps", IC = "IC", omega0 = "omega0", Lambda0 = "Lambda0")
    fileNames_a = u.getNamesInds("Data/" + approxName + "/" + "a" + tag)
    fileNames_aa = u.getNamesInds("Data/" + approxName + "/" + "aa" + tag)
    fileNames_ba = u.getNamesInds("Data/" + approxName + "/" + "ba" + tag)

    t = np.zeros(len(fileNames_a))
    a = np.zeros((len(fileNames_a), fo.N )) + 0j  
    aa = np.zeros((len(fileNames_aa), fo.N, fo.N )) + 0j  
    ba = np.zeros((len(fileNames_ba), fo.N, fo.N )) + 0j

    dt = meta["dt"]
    framesteps = meta["framesteps"]

    for i in range(len(fileNames_a)):
        a_ = np.load(fileNames_a[i])
        aa_ = np.load(fileNames_aa[i])
        ba_ = np.load(fileNames_ba[i])

        t[i] = dt*framesteps*(i+1)

        a[i] = a_
        aa[i] = aa_ 
        ba[i] = ba_ 

    return t, a, aa, ba

def setFigObj(name):
    # read in simulation parameters
    meta = u.getMetaKno(name, dir = 'Data/', N = "N", dt = "dt", frames = "frames", 
        framesteps = "framesteps", Norm = "Norm", omega0 = "omega0", Lambda0 = "Lambda0")
    fo.meta = meta

    # sets the figure object with these parameters
    # this is basically just so I can access them in the glocal scope
    fo.name = name

    fo.N = fo.meta["N"]
    fo.dt = fo.meta["dt"]
    fo.framsteps = fo.meta["framesteps"]
    fo.Norm =  fo.meta["Norm"]

    np.random.seed(1)
    fo.phi = np.random.uniform(0, 2 * np.pi, fo.N)


def analyze():
    t, a, aa, ba = get_fmeVals(fo.name, 'max')

    n_out = len(ba)

    n_tot = np.abs(fo.Norm)**2

    N = np.zeros((n_out,fo.N)) + 0j
    eigs = np.zeros((n_out,fo.N)) + 0j
    Q = np.zeros(n_out) + 0j

    for i in range(n_out):
        N[i] = np.diag(ba[i])
        eigs[i], _ = LA.eig(ba[i])
        eigs[i] = qu.sortE(np.abs(eigs[i]),eigs[i])
        Q[i] = np.sum(N[i] - np.abs(a[i])**2)/n_tot

    return t, N, ba, eigs, aa, a, Q


def SaveStuff(t, Num, M, eigs, aa, a, Q, label):
    np.save("../Data/" + fo.name + "/" + "_t" + label + ".npy", t)
    np.save("../Data/" + fo.name + "/" + "_N" + label + ".npy", Num)
    np.save("../Data/" + fo.name + "/" + "_M" + label + ".npy", M)
    np.save("../Data/" + fo.name + "/" + "_eigs" + label + ".npy", eigs)
    np.save("../Data/" + fo.name + "/" + "_aa" + label + ".npy", aa)
    np.save("../Data/" + fo.name + "/" + "_a" + label + ".npy", a)
    np.save("../Data/" + fo.name + "/" + "_Q" + label + ".npy", Q)




def makePOQFig(t, eigs, Q):
    fig, ax = plt.subplots(figsize = (6,6))

    n = np.abs(fo.Norm)**2

    ax.set_xlabel(r'$t$')

    PO = 1. - (eigs[:,-1] / n)

    ax.plot(t, PO, label = r'$1 - \lambda_p/n_{tot}$')
    ax.plot(t, Q, label = r'$Q$')

    ax.set_xlim(0, np.max(t) )
    ax.set_ylim(0, 1.05)

    ax.legend()

    fig.savefig("../Figs/" + fo.name + "_POQ.pdf",bbox_inches = 'tight')



def main(name, label = "", plot = PLOT):
    time0 = time.time()

    setFigObj(name) 

    tmin, amin = get_aVals(name, '')
    Nmin = np.abs(amin)**2

    t, N, M, eigs, aa, a, Q = analyze()

    SaveStuff(t, N, M, eigs, aa, a, Q, 'max')

    if plot:
        u.orientPlot()
        makePOQFig(t, eigs, Q)

    print('completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))


if __name__ == "__main__":
    main(simName)