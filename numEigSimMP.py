# --------------- imports --------------- #
import scipy.stats as st
from scipy.stats import multinomial
import SimObj as S 
import time 
import numpy as np
import multiprocessing as mp 
import utils as u
import os
import di_analysis
from distutils.dir_util import copy_tree
from shutil import copyfile
import datetime
import FullQuantumObjRetry as FQ 
import yt; yt.enable_parallelism();
#end = lambda id, start: print(f"Finish {id} in {time.time()-start:.4f} seconds")
import sys
# --------------------------------------- #

# --------------- Config Params --------------- #
r = 5 # scaling parameter
ofile =  "NE_r" + str(r)  # name of directory to be created
# this can be used to restart the simulation if it needs to be stopped for some reason
# basically it should copy the completed parts of the sim if you specify the old directory
# you need to make a new directory for the new sim data
checkDir = [] 

CUPY = True
quad = True # should the velocity dispersion be quadratic (as opposed to linear)
O2 = True # should a second order integrator be used

dt = 1e-4 / np.sqrt(r) # simulation timestep

frames = 300 # how many data drops should there be
framesteps = int(256 * np.sqrt(r)) # number of timesteps between data drops

IC = np.asarray([0,2,2,1,0])*r # initial occupation expectations
mn = multinomial(IC.sum(), IC*1./IC.sum())
N = len(IC) # the number of allowed momentum modes
np.random.seed(1) 
phi = np.random.uniform(0, 2 * np.pi, N) # field phases

omega0 = 1. # kinetic constant
lambda0 = 0 # 4-point interaction constant
C = -.1 / r # long range interaction constant

dIs = [di_analysis] # data interpreters
# ----------------------------------------- #

# a class used to control the global namespace
class Meta(object):

    def __init__(self):
        self.time0 = 0 # sim start time
        self.total = 0

        self.tags = []

        self.H_sp = {}

        MakeMetaFile(N = N, dt = dt, frames = frames, framesteps = framesteps, IC = IC,
            omega0 = omega0, Lamda0 = lambda0)


def MakeMetaFile(**kwargs):
    try:
        os.mkdir("../Data/" + ofile + "/")
    except OSError:
        pass
    f = open("../Data/" + ofile + "/" + ofile + "Meta.txt", 'w+')
    f.write("sim start: " + str(datetime.datetime.now()) + '\n\n')

    for key, value in kwargs.items():
        f.write(str(key) + ": " + str(value) + '\n')
    f.write('\n')

    f.close()
m = Meta()

def FindDone():
    files = ["../Data/" + ofile + "/" + file for file in os.listdir("../Data/" + ofile) if (file.lower().startswith('indtotuple'))]
    return len(files)


def CheckRedundant(signature, checkDir):

    if len(checkDir) == 0:
        return False

    redundant = False

    tag = str(signature)

    # check if this sim has already been run
    if os.path.isdir("../" + checkDir + "/psi" + tag) and os.path.isdir("../" + checkDir + "/Num" + tag):
        print( "\nspecial Hilbert space already simulated, copying data...")
        redundant = True 

        try:
            os.mkdir("../" + ofile + "/psi" + tag)
        except OSError:
            pass
        try:
            os.mkdir("../" + ofile + "/Num" + tag)
        except OSError:
            pass

        fromDirectory = "../" + checkDir + "/psi" + tag
        toDirectory = "../" + ofile + "/psi" + tag
        copy_tree(fromDirectory, toDirectory)

        fromDirectory = "../" + checkDir + "/Num" + tag
        toDirectory = "../" + ofile + "/Num" + tag
        copy_tree(fromDirectory, toDirectory)

        src = "../" + checkDir + "/" + "indToTuple" + tag + ".pkl"
        dst = "../" + ofile + "/" + "indToTuple" + tag + ".pkl"
        copyfile( src, dst )

        src = "../" + checkDir + "/" + "tupleToInd" + tag + ".pkl"
        dst = "../" + ofile + "/" + "tupleToInd" + tag + ".pkl"
        copyfile( src, dst )

    return redundant


def initSim():
    s = S.SimObj() # the simulation object

    s.N = N # number of modes
    s.dt = dt # time step
    s.omega0 = omega0 # kinetic constant, hbar/m
    s.C = C # long range interaction constant
    s.Lambda0 = lambda0 # constant interaction constant
    s.frames = frames # number of data drops 
    s.framesteps = framesteps # timesteps in between data drops
    s.ofile = ofile # name of data directory

    s.kord = np.arange(N) - N/2 # kmodes in natural order (as oppose to fft ordering)

    return s

# given sim object s
# changes to the ICs dn, i.e. we simulate the term with IC + dn occupation
# other states in special hilbert space
# signature
def initFQ(s, IC_, HS, sign):
    fQ = FQ.QuantObj() # initialize the quantum object

    fQ.is_dispersion_quadratic = quad # quadratic dispersion 
    fQ.second_Order = O2 # order 2 solver 
    fQ.E_m = s.kord # modes in natural order


    fQ.IC = IC_ # the occupations for one of the terms in the special hilbert space
    ntot = 0 # the total number of particles
    ptot = 0 # the net momentum
    for i in range(len(IC_)):
        n = fQ.IC[i]
        ntot += n 
        ptot += n*i

    
    fQ.tag = str(sign) # tag for data drops

    # ------------ step 3a,b,c -------------- #
    fQ.SetOpsFromIC(s) 
    # ------------------------------------- #

    fQ.signature = (ntot, ptot)

    fQ.track_psi = True
    fQ.track_EN = True 
    fQ.track_rho = False

    return fQ



def copyList(A):
    A_ = []
    for i in range(len(A)):
        A_.append(A[i])
    return A_



# recursive version of inspectstates
# finds the allowed state space, ie The Hilbert Space
# returns the probability of being in this special hilbert space
def InspectStatesR(E, terms):
    N_m = len(IC)
    #E = (np.arange(N_m)*self.IC).sum() # get the amount of momentum our state has
    E_max = IC.sum()*(N_m-1)

    inds = []
    return InspectStatesP(inds, E_max - E, terms)

# (private) partner function
# given the indices (inds) of the previous for loops,
# and difference between the new maximum possible energy and the requried energy
def InspectStatesP(inds, dE, terms):
    m = len(inds) # the current mode being populated
    n_used = sum(inds) # number of particles already put into modes
    N_m = len(IC)
    N_p = IC.sum()

    P = 0

    # put i particles in the mode m 
    for i in range(N_p+1-n_used):
        new_dE = dE - (N_m - m - 1)*i # new max energy excess
        
        # if have occupied enough modes to know the final state
        if (m == N_m - 2):

            # if it has the correct energy
            if (new_dE == 0):
                # it is a valid state
                tuple_ = tuple(inds + [i] + [N_p - n_used - i])
                # calc prob and return it
                P_ = mn.pmf(np.array(tuple_))

                if P_ > 0.:
                    terms.append(np.array(tuple_) )
                    P += P_


        else:
            # if I populate the particles in this way,
            # is the new max possible energy excess at least 0?				
            if (new_dE >= 0):
                # then keep populating modes
                P += InspectStatesP(inds + [i], new_dE, terms)
    
    return P


def GetTerms():
    n = IC.sum()
    p = (np.arange(N)*IC).sum()

    signature = (n,p)
    m.H_sp[signature] = [IC]
    m.tags.append(str(signature))

# given a signature of a special hilbert space, sign
def RunTerm(sign):
    s = initSim() # initialize the simulation object

    # check if the special Hilbert space has already been simulated
    redundant_ = False
    for j in range(len(checkDir)):
        checkDir_ = checkDir[j]
        if not(redundant_):
            redundant_ = CheckRedundant(sign, checkDir_)
    
    # if it has not then begin the simulation
    if not(redundant_):
        
        # ------------ step 3a-c -------------- #
        fQ = initFQ(s, m.H_sp[sign][-1], m.H_sp[sign], sign) # initialize the full quantum solver

        if CUPY:
            fQ.ToCUPY() # solve using cupy
        # ------------------------------------- #

        s.solvers = [fQ] # add the full quantum solver to the simObj's solvers list
        fQ.ReadyDir(ofile) # create directories for data drops
        s.time0 = time.time() # time stuff

        # ------------ step 3d,e -------------- #
        s.Run() # run the simulation 
        # ------------------------------------- #

        s.EndSim() # end the simulation

    done = FindDone()
    str_ = (('%.2f percent data created' % (100*float(done)/m.total) ) 
        + (' in %i hrs, %i mins, %i s' %u.hms(time.time()-m.time0)))
    u.repeat_print(str_)
    
    return 1


def main():
    m.time0 = time.time() # record the simulation 

    print( 'initialization completed in %i hrs, %i mins, %i s' %u.hms(time.time()-m.time0))
    print( "begining sim", ofile)

    # ------------ step 1-2 --------------- #
    GetTerms()
    # ----------------------------------- #

    # these are used for timing
    m.total = len(m.H_sp.keys())
    m.done = 0

    # -------------- step 3 ------------- #
    # simulate each special Hilbert space in parallel
    #pool = mp.Pool(mp.cpu_count()) #OLD
    #pool.map(RunTerm, m.H_sp.keys()) #OLD
    start = time.time()
    for key in m.H_sp:
        print("\nDoing", key)
        sys.stdout.flush()
        RunTerm(key)

    #end(4, start)
    # ----------------------------------- #

    time1 = time.time()

    tags_ = np.array(m.tags)
    np.save("../Data/" + ofile + "/" + "tags" + ".npy", tags_)

    print("\nbegining data interpretation")

    if yt.is_root():
        for i in range(len(dIs)):
            dIs[i].main(ofile, tags_, plot = False)
    print('analysis completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time1))

    print('script completed in %i hrs, %i mins, %i s' %u.hms(time.time()-m.time0))
    u.ding()
    sys.stdout.flush()


if __name__ == "__main__":
    main()