# --------------- imports --------------- #
import scipy.stats as st
import SimObj as S 
import time 
import numpy as np
import multiprocessing as mp 
import utils as u
import os
import di_CS_Num
from distutils.dir_util import copy_tree
from shutil import copyfile
import datetime
import FullQuantumObjRetry as FQ 
import yt; yt.enable_parallelism()
end = lambda id, start: print(f"Finish {id} in {time.time()-start:.4f} seconds")
import sys
# --------------------------------------- #

# --------------- Config Params --------------- #
r = 5 # scaling parameter
ofile =  "test_r" + str(r)  # name of directory to be created
# this can be used to restart the simulation if it needs to be stopped for some reason
# basically it should copy the completed parts of the sim if you specify the old directory
# you need to make a new directory for the new sim data
checkDir = [] 

quad = True # should the velocity dispersion be quadratic (as opposed to linear)
O2 = True # should a second order integrator be used

dt = 1e-4 / np.sqrt(r) # simulation timestep

frames = 300 # how many data drops should there be
framesteps = int(256 * np.sqrt(r)) # number of timesteps between data drops

IC = np.asarray([0,2,2,1,0])*r # initial occupation expectations
N = len(IC) # the number of allowed momentum modes
np.random.seed(1) 
phi = np.random.uniform(0, 2 * np.pi, N) # field phases

omega0 = 1. # kinetic constant
lambda0 = 0 # 4-point interaction constant
C = -.1 / r # long range interaction constant

dIs = [di_CS_Num] # data interpreters
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
        print("\nspecial Hilbert space already simulated, copying data...")
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

    # ------------ step 3a,b -------------- #
    fQ.SetOpsFromIC(s) 
    # ------------------------------------- #

    fQ.signature = (ntot, ptot)

    # ------------ step 3c -------------- #
    fQ.SetPsiHS(HS, IC, phi)
    # ----------------------------------- #

    fQ.track_psi = True
    fQ.track_EN = True 
    fQ.track_rho = False

    return fQ

def FindDN(mu):
    P_goal = .999
    sig = 1
    found = False

    while(not(found)):
        P = 0
        lower = np.max([0, mu - sig])
        for i in range(lower, mu + sig + 1):
            P += st.poisson.pmf(i, mu)
        if P >= P_goal:
            found = True 
        else:
            sig += 1


    lower = np.max([0, mu - sig])
    return [lower - mu, sig]
def GetDNs():
    DN = []
    for i in range(N):
        IC_ = IC[i]
        if IC_ == 0:
            DN.append([0,0])
        else:
            DN.append(FindDN(IC[i]))
    Results = []
    GetDNsR(DN, [], Results)

    return Results

def GetDNsR(DN, dn, Results):
    DN_ = copyList(DN)
    if len(DN) > 0:
        DN__ = DN_.pop(0)
        for dn_ in range(DN__[1], DN__[0] - 1, -1):
            GetDNsR(DN_, dn + [dn_], Results)
    else:
        Results.append(dn)

def copyList(A):
    A_ = []
    for i in range(len(A)):
        A_.append(A[i])
    return A_


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
        # ------------------------------------- #

        s.solvers = [fQ] # add the full quantum solver to the simObj's solvers list
        fQ.ReadyDir(ofile) # create directories for data drops
        s.time0 = time.time() # time stuff

        # ------------ step 3d,e -------------- #
        s.Run(False) # run the simulation 
        # ------------------------------------- #

        s.EndSim(text = False) # end the simulation

    done = FindDone()
    str_ = (('%.2f percent data created' % (100*float(done)/m.total) ) 
        + (' in %i hrs, %i mins, %i s' %u.hms(time.time()-m.time0)))
    u.repeat_print(str_)
    
    return 1


def main():
    m.time0 = time.time() # record the simulation 

    print("begining sim", ofile)

    # ------------ step 1 --------------- #
    start = time.time()
    DNs = GetDNs() # find the term expressed as differences from the expectation values
    end(0, start)

    terms = []

    start = time.time()
    # create a list of terms as list of number eigenstates
    for i in range(len(DNs)):
        IC_ = IC.copy()
        dn = DNs[i]
        for j in range(len(dn)):
            IC_[j] += dn[j]
        terms.append(IC_)
    end(1, start)
    # ----------------------------------- #

    # ----------- step 2 ---------------- #
    # find the special Hilbert spaces (SHS) associated with these terms
    # for each term in the coherent state sum
    start = time.time()
    for i in range(len(terms)):
        term = terms[i] 
        p = (np.arange(N)*np.array(term)).sum() # identify momentum
        n = np.array(term).sum() # identify total particle number
        signature = (n,p) # signature identifying a given SHS
        if not(signature in m.H_sp): # if this SHS is not in the dictionary 
            m.H_sp[signature] = [term] # add it to the dictionary
            m.tags.append(str(signature)) # the tags list is used in data interpretation
        else:
            m.H_sp[signature].append(term) # add this term to the SHS
    end(3, start)
    # ----------------------------------- #

    # these are used for timing
    m.total = len(m.H_sp.keys())
    m.done = 0

    # -------------- step 3 ------------- #
    print("running %i terms on %i cpus" %(len(m.H_sp), mp.cpu_count()))

    # simulate each special Hilbert space in parallel
    #pool = mp.Pool(mp.cpu_count()) #OLD
    #pool.map(RunTerm, m.H_sp.keys()) #OLD
    start = time.time()
    for key in yt.parallel_objects(m.H_sp.keys(),0):
        print("Doing", key)
        sys.stdout.flush()
        RunTerm(key)

    end(4, start)
    # ----------------------------------- #

    time1 = time.time()

    tags_ = np.array(m.tags)
    np.save("../Data/" + ofile + "/" + "tags" + ".npy", tags_)

    print("\nbegining data interpretation")

    for i in range(len(dIs)):
        dIs[i].main(ofile, tags_)
    print('analysis completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time1))

    print('script completed in %i hrs, %i mins, %i s' %u.hms(time.time()-m.time0))
    u.ding()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
