import utils as u
import numpy as np
import scipy.fftpack as sp
import time
import matplotlib.pyplot as plt
import scipy.stats as st
import pickle 

# the data directory
simName = "Gr_r1"

class figObj(object):

    def __init__(self):
        self.meta = None 

        self.ax = None 

        self.name = None
        self.fileNames_psi = None

        self.tags = None

        self.indToTuple = None 
        self.tupleToInd = None  

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

def GetPsiAndN(j):
    psi = np.zeros(len(fo.indToTuple)) + 0j
    N = np.zeros(( len(fo.indToTuple), fo.N ))

    # for state in the initial super position
    for tag_ in fo.tags:

        fo.fileNames_psi = u.getNamesInds('Data/' + fo.name + "/" + "psi" + tag_)

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


def GetTAndNum(tags):
    t = np.zeros(len(fo.fileNames_psi))
    Num = np.zeros((fo.N, len(fo.fileNames_psi)))

    time0 = time.time() # for timing

    for i in range( len(fo.fileNames_psi) ):
        _, N = GetPsiAndN(i)

        t[i] = fo.dt*fo.framsteps*(i+1)

        Num[:,i] = N

        u.repeat_print(('%i hrs, %i mins, %i s remaining.' %u.remaining(i + 1, len(fo.fileNames_psi), time0)))

    return t, Num

def makeFig(tags):
    fig, fo.ax = plt.subplots(figsize = (10,10))

    t, Num = GetTAndNum(tags)

    fo.ax.set_xlabel(r'$t$')

    for i in range(fo.N):
        fo.ax.plot(t, Num[i], label = 'E[N%i]' %i)

    plt.legend(loc=1) 
    
    return fig, t, Num


def setFigObj(name, tags):
    meta = u.getMetaKno(name, dir = 'Data/', N = "N", dt = "dt", frames = "frames", 
        framesteps = "framesteps", IC = "IC", omega0 = "omega0", Lambda0 = "Lambda0")
    fo.meta = meta

    fo.name = name

    fo.tags = tags

    fo.N = fo.meta["N"]
    fo.dt = fo.meta["dt"]
    fo.framsteps = fo.meta["framesteps"]
    fo.IC =  fo.meta["IC"]
    fo.omega0 = fo.meta["omega0"]
    fo.Lambda0 = fo.meta["Lambda0"]

    np.random.seed(1)
    fo.phi = np.random.uniform(0, 2 * np.pi, fo.N)

    SetDicts()


def SetDicts():
    newIndToTuple = {} # dictionary describing index to tuple mapping for the total hilbert space
    newTupleToInd = {} # dictionary describing tuple to index mapping -- --

    # start by constructing the total hilbert space maps
    # for state in the initial super position
    print len(fo.tags)

    for tag_ in fo.tags:

        # load its "special" Hilbert space map
        with open("../Data/" + fo.name + "/" + "indToTuple" + tag_ + ".pkl", 'rb') as f:    
            indToTuple = pickle.load(f)

        # for state in the special hilbert space
        for i in range(len(indToTuple)):
            state_ = indToTuple[i] # get the state

            # add this state to the total hilbert space maps
            
            if state_ in newTupleToInd:
                pass # this state is already represented in the dictionaries
            else:
                # else add the state and index to the dictionaries
                ind_ = len(newIndToTuple) 
                newIndToTuple[ind_] = state_
                newTupleToInd[state_] = ind_    
    
    fo.tupleToInd = newTupleToInd
    fo.indToTuple = newIndToTuple

def SaveStuff(t, Num):
    np.save("../Data/" + fo.name + "/" + "Num_t" + ".npy", t)
    np.save("../Data/" + fo.name + "/" + "Num_Num" + ".npy", Num)

def main(name, tags):
    time0 = time.time()
    u.orientPlot()

    setFigObj(name, tags)

    fo.fileNames_psi = u.getNamesInds('Data/' + name + "/" + "psi" + tags[0])

    fig, t, Num = makeFig(tags)
    print Num.sum(axis = 0)[0], Num.sum(axis = 0)[-1]
    SaveStuff(t, Num)
    fig.savefig("../Figs/" + name + "_Num.pdf",bbox_inches = 'tight')
    print 'completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
    print "output: ", "../Data/" + name + "_Num.pdf"

if __name__ == "__main__":
    tags = np.load("../Data/" + simName + "/tags.npy")
    main(simName, tags = tags)