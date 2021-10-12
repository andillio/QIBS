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
import yt; yt.enable_parallelism(); is_root = yt.is_root();

end = lambda start, id: print(f"Done with {id} in {time.time()-start:.4f} seconds")

simName = "FNS_r1"
simName = "test_r15_(0,30,30,15,0)"
decimate = 2
label = ""
PLOT = True

class figObj(object):
    '''
    This class stores all simulation metadata for figures
    '''

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
    '''
    This function takes in a string with format '(A, B)' and returns a tuple
    with (A, B)

    Parameters
    ---------------------------------------------------------------------------
    string: string
      A string with format '(A, B)'
    
    Returns
    ---------------------------------------------------------------------------
    tuple: tuple
      A tuple with (A, B)
    '''
    string = string.replace('(','')
    string = string.replace(')','')
    ints = string.split(",")
    return (int(ints[0]), int(ints[1]) )


def setFigObj(name, decimate):
    '''
    This function populates the attributes of the instance of the figObj class
    with values.
    '''
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


def offDiag(sig, psi_j, indToTuple_j, i):
    '''
    This function constructs the offdiagonal term of the (ANDREW TODO: CONFIRM)
    special Hilbert space hamiltonian.

    Parameters
    ---------------------------------------------------------------------------
    psi_j: array-like
      The (ANDREW TODO: CONFIRM) special hilbert space wavefunction
    indToTuple_j: (ANDREW TODO)
      Converts index of state to its tuple key
    i: int
      Index to state
    '''

    M = np.zeros((fo.N, fo.N)) + 0j

    # creation op on mode b
    for b in range(fo.N):  
        # annihilation op on mode a 
        for a in range(b+1, fo.N):

            new_p = sig[1] + b - a
            newsig = (sig[0], new_p)

            tag_ = str(newsig)

            if tag_ in fo.tags: 
                tupleToInd = None

                with open("../Data/" + fo.name + "/" + "tupleToInd" + tag_ + ".pkl", 'rb') as f:    
                    tupleToInd_ = pickle.load(f)

                # load in the psi in a given special Hilbert space
                fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + tag_)
                psi_ = np.load(fileNames_psi[i])

                for j in range(len(psi_j)):
                    c_j = psi_j[j]
                    state_j = np.array(indToTuple_j[j])

                    state_i = state_j.copy()
                    state_i[a] = state_j[a] - 1
                    state_i[b] = state_j[b] + 1


                    if tuple(state_i) in tupleToInd_:
                        i_ = tupleToInd_[ tuple(state_i) ]
                        val_ = c_j
                        val_ *= np.sqrt(state_j[b] + 1)
                        val_ *= np.sqrt(state_j[a])
                        val_ *= np.conj(psi_[i_])

                        M[b,a] += val_ 
                        M[a,b] += np.conj(val_)
    
    return M
                

def get_aa(sig, psi_j, indToTuple_j, i):
    '''
    This function constructs the aa operator for (ANDREW TODO)

    Parameters
    ---------------------------------------------------------------------------
    psi_j: array-like
      The (ANDREW TODO: CONFIRM) special hilbert space wavefunction
    indToTuple_j: (ANDREW TODO: what kind of thing is this?)
      Converts index of state to its tuple key
    i: int
      Index to state

    Returns
    ---------------------------------------------------------------------------
    aa: array-like
      The aa operator
    '''

    aa = np.zeros((fo.N, fo.N)) + 0j

    for a1 in range(fo.N):
        for a2 in range(a1,fo.N):

            new_p = sig[1] - a1 - a2
            newsig = (sig[0] - 2, new_p)

            newTag = str(newsig)

            if newTag in fo.tags:
                tupleToInd = None

                with open("../Data/" + fo.name + "/" + "tupleToInd" + newTag + ".pkl", 'rb') as f:    
                    tupleToInd_ = pickle.load(f)

                # load in the psi in a given special Hilbert space
                fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + newTag)
                psi_ = np.load(fileNames_psi[i])

                for j in range(len(psi_j)):
                    c_j = psi_j[j]
                    state_j = np.array(indToTuple_j[j])

                    state_i = state_j.copy()
                    state_i[a1] = state_i[a1] - 1
                    state_i[a2] = state_i[a2] - 1            


                    if tuple(state_i) in tupleToInd_:
                        i_ = tupleToInd_[ tuple(state_i) ]
                        val_ = c_j
                        if a1 != a2:
                            val_ *= np.sqrt(state_j[a1])
                            val_ *= np.sqrt(state_j[a2])
                            val_ *= np.conj(psi_[i_])

                            aa[a1,a2] += val_ 
                            aa[a2,a1] += val_

                        else:
                            val_ *= np.sqrt(state_j[a1])
                            val_ *= np.sqrt(state_j[a2]-1)
                            val_ *= np.conj(psi_[i_])

                            aa[a1,a2] += val_ 
    return aa


def get_a(sig, psi_j, indToTuple_j, i):

    '''
    This function constructs the a operator for (ANDREW TODO)

    Parameters
    ---------------------------------------------------------------------------
    psi_j: array-like
      The (ANDREW TODO: CONFIRM) special hilbert space wavefunction
    indToTuple_j: (ANDREW TODO: what kind of thing is this?)
      Converts index of state to its tuple key
    i: int
      Index to state

    Returns
    ---------------------------------------------------------------------------
    a: array-like
      The a operator
    '''

    a = np.zeros(fo.N) + 0j

    for a1 in range(fo.N):
        new_p = sig[1] - a1
        newsig = (sig[0] - 1, new_p)

        newTag = str(newsig)
    
        if newTag in fo.tags: 
            tupleToInd = None

            with open("../Data/" + fo.name + "/" + "tupleToInd" + newTag + ".pkl", 'rb') as f:    
                tupleToInd_ = pickle.load(f)

            # load in the psi in a given special Hilbert space
            fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + newTag)
            psi_ = np.load(fileNames_psi[i])

            for j in range(len(psi_j)):
                c_j = psi_j[j]
                state_j = np.array(indToTuple_j[j])

                state_i = state_j.copy()
                state_i[a1] = state_i[a1] - 1

                if tuple(state_i) in tupleToInd_:
                    i_ = tupleToInd_[ tuple(state_i) ]
                    val_ = c_j
                    val_ *= np.sqrt(state_j[a1])
                    val_ *= np.conj(psi_[i_])
                    a[a1] += val_

    return a 

def getN(psi_, indToTuple_):

    '''
    This function constructs the number operator for (ANDREW TODO)

    Parameters
    ---------------------------------------------------------------------------
    psi_: array-like
      The (ANDREW TODO: CONFIRM) special hilbert space wavefunction
    indToTuple_j: (ANDREW TODO: what kind of thing is this?)
      Converts index of state to its tuple key

    Returns
    ---------------------------------------------------------------------------
    N: array-like
      The N operator
    '''

    N = np.zeros(fo.N)

    for j in range(len(indToTuple_)):
        subState = np.array(indToTuple_[j])

        c_j = psi_[j]

        if np.abs(c_j)>0:

            N += subState*np.abs(c_j)**2

    return N


def analyzeTimeStep(i):
    '''
    This function finds all of the relevant summarizing quantities for each
    timestep, e.g. number, eigenvalues, squeezing

    Parameters
    ---------------------------------------------------------------------------
    i: integer
      Timestep to analyze

    Returns
    ---------------------------------------------------------------------------
    t: float
      Current simulation time
    N: (ANDREW TODO)
      (ANDREW TODO)
    M: array-like
      (ANDREW TODO)
    eigs: array-like
      Eigenvalues of M matrix
    aa: (ANDREW TODO)
      (ANDREW TODO)
    a: (ANDREW TODO)
      (ANDREW TODO)
    Q: float
      Classical aproximation error tracker
    '''


    t = fo.dt*fo.framsteps*(i+1)
    outputs = {}

    for sto, tag_ in yt.parallel_objects( fo.tags , 0, storage=outputs, dynamic = True):
        sys.stdout.flush()
        sto.result_id = tag_
        sig = str2sig(tag_)

        #start = time.time()
        # load in the psi in a given special Hilbert space
        fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + tag_)
        psi_ = np.load(fileNames_psi[i])
        #end(start, "Loading Psi")

        #start = time.time()
        indToTuple_ = None 
        with open("../Data/" + fo.name + "/" + "indToTuple" + tag_ + ".pkl", 'rb') as f:    
            indToTuple_ = pickle.load(f)
        #end(start, "Loading indToTuple")

        N_ = getN(psi_, indToTuple_)

        M_ = np.zeros((fo.N, fo.N)) + 0j
        aa_ = np.zeros((fo.N, fo.N)) + 0j
        a_ = np.zeros(fo.N) + 0j

        #start = time.time()
        M_ += np.diag(N_)
        #end(start, "adding diag")

        #start = time.time()
        M_ += offDiag(sig, psi_, indToTuple_, i)
        #end(start, "adding offdiag")

        #start = time.time()
        aa_ += get_aa(sig, psi_, indToTuple_, i)
        #end(start, "adding aa")

        #start = time.time()
        a_ += get_a(sig, psi_, indToTuple_, i)
        #end(start, "adding aa")
        
        sto.result = (N_, M_, aa_, a_)
    
    N = np.zeros(fo.N)
    M = np.zeros((fo.N, fo.N)) + 0j
    aa = np.zeros((fo.N, fo.N)) + 0j
    a = np.zeros(fo.N) + 0j

    for i, key_ in enumerate(outputs.keys()):

        #key_ = outputs.keys()[i]
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
    '''
    main() for analysis
    '''
    print("Starting di_analysis...")

    time0 = time.time()

    t, N, M, eigs, aa, a, Q = [], [], [], [], [], [], []

    steps = len(range(0,len(fo.fileNames_psi), fo.decimate))

    for i in range(0,len(fo.fileNames_psi), fo.decimate):

        if is_root:
            start = time.time()
        t_, N_, M_, eigs_, aa_, a_, Q_ = analyzeTimeStep(i)
        if is_root:
            end(start, f"analyzing timestep {i}")

        t.append(t_)
        N.append(N_)
        M.append(M_)
        eigs.append(eigs_)
        aa.append(aa_)
        a.append(a_)
        Q.append(Q_)
        if is_root:
            u.repeat_print(('%i hrs, %i mins, %i s remaining.\n' %u.remaining(i + 1, steps, time0)))

    t = np.array(t)
    N = np.array(N)
    M = np.array(M)
    eigs = np.array(eigs)
    aa = np.array(aa)
    a = np.array(a)
    Q = np.array(Q)

    return t, N, M, eigs, aa, a, Q


def makeNFig(t, N):
    '''
    Make number operator figure.

    Parameters
    ---------------------------------------------------------------------------
    t: float
      Simulation time of a given timestep
    N: array-like
      Number operator 
    '''
    fig, ax = plt.subplots(figsize = (6,6))

    ax.set_xlabel(r'$t$')

    for i in range(fo.N):
        ax.plot(t, N[:,i], label = r'$E[\hat N_%i]$' %i)

    ax.set_xlim(0, np.max(t) )
    ax.set_ylim(0, np.max(N)*1.05 )

    ax.legend()

    fig.savefig("../Figs/" + fo.name + "_Num.pdf",bbox_inches = 'tight')


def makeMFig(t, lams):
    '''
    Make M operator figure.

    Parameters
    ---------------------------------------------------------------------------
    t: float
      Simulation time of a given timestep
    M: array-like
      M operator 
    '''
    fig, ax = plt.subplots(figsize = (6,6))

    ax.set_xlabel(r'$t$')

    for i in range(fo.N):
        ax.plot(t, lams[:,i], label = r'$\lambda_%i$' %i)

    ax.set_xlim(0, np.max(t) )
    ax.set_ylim(0, np.max(lams)*1.05 )

    ax.legend()

    fig.savefig("../Figs/" + fo.name + "_lams.pdf",bbox_inches = 'tight')


def makePOQFig(t, eigs, Q):
    '''
    Make classical approximation error tracker figure.

    Parameters
    ---------------------------------------------------------------------------
    t: float
      Simulation time of a given timestep
    eigs: array-like
      Eigenvalues of M operator
    Q: array-like
      Error tracking matrix
    '''
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

def constructSq(a,aa,M):
    '''
    Construct squeezing operator

    Parameters
    ---------------------------------------------------------------------------
    a: array-like
      The a operator
    aa: array-like
      The aa operator
    M: array-like
      The M operator
    '''

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

def makeSqueezeFig(t, aa, M, a):
    '''
    
    '''

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

    ax.set_xlim(0, np.max(t[sq<2]) )
    ax.set_ylim(0,2.)

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
