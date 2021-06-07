import utils as u
import QUtils as qu 
import numpy as np
import cupy as cp 
import scipy.fftpack as sp
import time
import matplotlib.pyplot as plt
import pickle 
from numpy import linalg as LA 
import scipy.stats as st 

# data directory
simName = "Gr_r4"
# only looks at every decimate'th data drop
decimate = 2
# labels output files
label = ""



class figObj(object):

    def __init__(self):
        self.meta = None

        self.DN1 = [-10,10]
        self.DN2 = [-14,11]

        self.ax = None 

        self.name = None
        self.fileNames_psi = None

        self.indToTuple = None 
        self.tupleToInd = None    

        self.reduceMap = None
        self.NMap = None

        self.tags = None
        
        self.M_data = None

        self.N = None 
        self.dt = None 
        self.frames = None 
        self.framsteps = None
        self.IC = None
        self.phi = None 
        self.omega0 = None 
        self.Lambda0 = None 
fo = figObj()



def Tag2Array(tag_):
    tags_ = tag_.replace('[','')
    tags_ = tags_.replace(']','')
    tags_ = tags_.split(' ')

    tags = []

    for i in range(len(tags_)):
        if (len(tags_[i]) > 0): 
            tags.append(int(tags_[i]))

    return np.array(tags) 


def coherentICs():
    psi = np.zeros(len(fo.indToTuple)) + 0j
    N = np.zeros(fo.N)

    states = []

    for dn1 in range(fo.DN1[1], fo.DN1[0] - 1, -1):
        for dn2 in range(fo.DN2[1], fo.DN2[0] - 1, -1):
            substate = fo.IC.copy()
            substate[1] += dn1 
            substate[2] += dn2 

            states.append(substate)

            ind_ = fo.tupleToInd[tuple(substate)]
            P = IC2Prob(substate)

            psi[ind_] += np.sqrt(P)
            N += P*substate
    
    return psi, N


def GetPsiAndN(j):
    psi = np.zeros(len(fo.indToTuple)) + 0j
    N = np.zeros(( len(fo.indToTuple), fo.N ))

    # for state in the initial super position
    for tag_ in fo.tags:

        fo.fileNames_psi = u.getNamesInds("Data/" + fo.name + "/" + "psi" + tag_)

        # get the wavefunction for this initial state at the relevent time
        psi_ = None 
        psi_ = np.load(fo.fileNames_psi[j])

        # load the correspond "special" hilbert space map
        with open("../Data/" + fo.name + "/" + "indToTuple" + tag_ + ".pkl", 'rb') as f:    
            indToTuple_ = pickle.load(f)

        # for each state in the special hilbert space 
        # add the weight in psi_ to the total wavefunction
        for i in range(len(indToTuple_)):
            subState = indToTuple_[i]
            ind_ = fo.tupleToInd[subState]

            subState_ = np.array(subState)

            psi[ind_] += psi_[i]
            N[ind_,:] = subState_ 

    N = np.einsum("ij,i->j", N, np.abs(psi)**2 )

    return psi, N


# sort eigenvalues A
# using key
def sortE(key, A):
	inds = key.argsort()
	A = A[inds]
	return A

def FindLams(decimate):

    t = []
    Lams = []
    
    for i in range(fo.N):
        Lams.append([])
    
    time0 = time.time() # for timing

    for i in range(len(fo.fileNames_psi)):
        # only look at every decimate'th timestep
        if (i%decimate == 0):
            # load in the ith wavefunction
            psi_, N_ = GetPsiAndN(i)

            t.append(fo.dt*fo.framsteps*(i+1)) # get the timestamp
            
            M = np.zeros((fo.N, fo.N)) + 0j
            M += np.diag(N_)
            M += GetOffDiag(psi_)

            fo.M_data.append(M)

            eigs, _ = LA.eig(M)
            eigs = sortE(np.abs(eigs),eigs)
            for n in range(len(eigs)):
                Lams[n].append(eigs[n])

            # output an estimate of the remaining time
            u.repeat_print(('%i hrs, %i mins, %i s remaining.' %u.remaining(i/decimate + 2, len(fo.fileNames_psi)/decimate + 1, time0)))

    
    return t, np.array(Lams)


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
        



def makeFig(decimate):
    fig, fo.ax = plt.subplots(figsize = (10,10))

    t, lams = FindLams(decimate)

    fo.ax.set_xlabel(r'$t$')

    for i in range(fo.N):
        plt.plot(t, lams[i])

    plt.show()

    return fig, t, lams



def GetDicts():
    newIndToTuple = {} # dictionary describing index to tuple mapping for the total hilbert space
    newTupleToInd = {} # dictionary describing tuple to index mapping -- --

    # start by constructing the total hilbert space maps
    # for state in the initial super position
    for tag_ in fo.tags:

        # load its "special" Hilbert space map
        with open("../Data/" + fo.name + "/" + "indToTuple" + tag_ + ".pkl", 'rb') as f:    
            indToTuple = pickle.load(f)

        # for state in the special hilbert space
        for i in range(len(indToTuple)):
            state_ = indToTuple[i] # get the state

            # add this state to the total hilbert space maps
            ind_ = len(newIndToTuple) 
            newIndToTuple[ind_] = state_
            newTupleToInd[state_] = ind_     
    
    fo.tupleToInd = newTupleToInd
    fo.indToTuple = newIndToTuple


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

    fo.omega0 = fo.meta["omega0"]
    fo.Lambda0 = fo.meta["Lambda0"]
    
    #fo.DN1 = FindDN(fo.IC[1])
    #fo.DN2 = FindDN(fo.IC[2])

    np.random.seed(1)
    fo.phi = np.random.uniform(0, 2 * np.pi, fo.N)

    fo.M_data = []
    
    # get and set all the dictionaries and the reduction map
    # the reduction map just tracks all the combination of states in
    # the partial trace that have non zero contributions to the sum
    GetDicts()


def SaveStuff(t, lams):
    # save relevent data
    np.save("../Data/" + fo.name + "/" + "M_t" + label + ".npy", t)
    np.save("../Data/" + fo.name + "/" + "M_lams" + label + ".npy", lams)
    np.save("../Data/" + fo.name + "/" + "M" + label + ".npy", np.array(fo.M_data))


def main(name, tags, label = "", decimate = 1):
    time0 = time.time() # for timing the analysis
    u.orientPlot() # format plots

    # initialize the figure object class
    setFigObj(name)

    # this is basically just to see how many time drops there were
    fo.fileNames_psi = u.getNamesInds("Data/" + name + "/" + "psi" + tags[0])

    print "Started fig: " + name + "_M" + label

    fig, t, lams = makeFig(decimate)
    SaveStuff(t, lams)

    # save the figure
    fig.savefig("../Figs/" + name + "_M" + label + ".pdf",bbox_inches = 'tight')
    
    # report on output
    print '\ncompleted in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
    print "output: ", "../Data/" + name + "_M" + label +".pdf"


if __name__ == "__main__":
    # load in the tags on the data directories
    try:
        fo.tags = np.load("../Data/" + simName + "/tags.npy")
    except IOError:
        fo.tags = [""]
    # run the main function
    main(simName, fo.tags, label, decimate)