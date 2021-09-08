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

simName = "Gr_r3"
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


def getR(sig, psi_j, indToTuple_j,i):
    
    R = 0j 

    for b in range(fo.N):
        a1 = b 
        a2 = b

        new_p = sig[1] + b - a1 - a2 
        newsig = (sig[0] - 1, new_p)

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

                state_i_ = state_j.copy()
                state_i_[a1] = state_i_[a1] - 1
                state_i_[a2] = state_i_[a2] - 1
                state_i_[b] = state_i_[b] + 1

                if tuple(state_i_) in tupleToInd_:
                    state_i = state_j.copy()
                    W = 1. 

                    W *= np.sqrt(state_i[a2])
                    state_i[a2] -= 1

                    W *= np.sqrt(state_i[a1])
                    state_i[a1] -= 1
                    
                    W *= np.sqrt(state_i[b] + 1)
                    state_i[b] += 1

                    i_ = tupleToInd_[ tuple(state_i) ]

                    if a1 == a2:
                        R += W * c_j * np.conj(psi_[i_])
                    else:
                        R += 2*W * c_j * np.conj(psi_[i_])

    return R



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

        R_ = getR(sig, psi_, indToTuple_,i)

        sto.result = R_

    R = 0J 

    for i, key_ in enumerate(outputs.keys()):

        R_ = outputs[key_]
        R += R_

    return t, R

def analyze():
    print("Starting di_analysis...")

    time0 = time.time()

    t, R = [], []

    steps = len(range(0,len(fo.fileNames_psi)))

    for i in range(0,len(fo.fileNames_psi), fo.decimate):
        t_, R_ = analyzeTimeStep(i)

        t.append(t_)
        R.append(R_)

        if is_root:
            u.repeat_print(('%i hrs, %i mins, %i s remaining.\n' %u.remaining(i + 1, steps, time0)))
    
    t = np.array(t)
    R = np.array(R)
    
    return t, R


def SaveStuff(t, R):
    R = qu.sortE(t, R)
    t = qu.sortE(t,t)

    np.save("../Data/" + fo.name + "/" + "_tR" + ".npy", t)
    np.save("../Data/" + fo.name + "/" + "_R" + ".npy", R)


def main(name, tags = [], label = "", decimate = 1, plot = PLOT):
    time0 = time.time()

    fo.tags = tags

    setFigObj(name, decimate)

    t, R = analyze()

    SaveStuff(t, R)

    if plot:
        u.orientPlot()

    print('completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))



if __name__ == "__main__":
    # load in the tags on the data directories
    try:
        fo.tags = np.load("../Data/" + simName + "/tags.npy")
    except IOError:
        fo.tags = [""]
    main(simName, fo.tags, decimate = decimate)