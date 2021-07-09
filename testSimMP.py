# --------------- imports --------------- #
import scipy.stats as st
import SimObj as S 
import time 
import numpy as np
import multiprocessing as mp 
import utils as u
import QUtils as qu
import os
import di_analysisBig
from distutils.dir_util import copy_tree
from shutil import copyfile
import datetime
import FullQuantumObjRetry as FQ 
import yt; yt.enable_parallelism(); is_root = yt.is_root()
#end = lambda id, start: print(f"Finish {id} in {time.time()-start:.4f} seconds")
import sys
import argparse



# --------------- Simulation Params --------------- #
r = 2 # scaling parameter
#IC = np.asarray([0,2,2,1,0])*r # initial occupation expectations
IC = np.asarray([0,2,2,1])*r # initial occupation expectations

name_ = "_("
for i in range(len(IC)):
    name_ += str(IC[i])
    if i != len(IC) - 1:
        name_ += ","

name_ += ")"
name_ = ""

# NOTE: if not overwriting all simulation and meta params need to be the same
# as the simulation being resumed
OVERWRITE = True # should I overwrite existing files or resume from where I left off
ofile =  "fourModes_r" + str(r) + name_  # name of directory to be created

quad = True # should the velocity dispersion be quadratic (as opposed to linear)
O2 = True # should a second order integrator be used

dt = 1e-4 / np.sqrt(r) # simulation timestep

frames = 300 # how many data drops should there be
framesteps = int(256 * r) # number of timesteps between data drops

N = len(IC) # the number of allowed momentum modes
np.random.seed(1) 
phi = np.random.uniform(0, 2 * np.pi, N) # field phases

omega0 = 1. # kinetic constant
lambda0 = 0 # 4-point interaction constant
C = -.1 / r # long range interaction constant

dIs = [di_analysisBig] # data interpreters
# ----------------------------------------------- #



# a class used to control the global namespace
class Meta(object):

    """
    An object that stores all of the simulation metadata, including tags,
    number of particles, timestep(s), frames, initial conditions, and
    physical parameters. 
    """

    def __init__(self):
        
        # Simulation start time
        self.time0 = 0

        # Counter for total number of special Hilbert spaces
        self.total = 0

        # List to hold tags
        self.tags = []

        # Dict to hold all special Hilbert spaces
        self.H_sp = {}

        # Generate file containing all simulation metadata
        self.MakeMetaFile(N = N, dt = dt, frames = frames, framesteps = framesteps, IC = IC,
            omega0 = omega0, Lamda0 = lambda0, C = C)


    def MakeMetaFile(self, **kwargs):

        '''
        This method of the Meta object generates the file which contains all 
        given metadata.

        Parameters (kwargs expected y default)
        ---------------------------------------------------------------------------
        N: int
            Number of particles in the simulation.
        dt: float
            The step size which the solver takes for each step.
        framesteps: int
            The number of time steps between each data dump.
        frames: int
            The number of data dumps, which with dt and framesteps determines the 
            total sim time.
        IC: list-like
            A list-like object containing the number of particles in each mode.
        omega0: float
            The kinetic constant (1.0 by default).
        Lambda0: float
            The 4-point interaction constant.
        C: float
            The long range interaction constant.
        '''
        
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

    """
    This function scans the ../Data/ directory for the number of special 
    hilbert spaces that have already been simulated.

    Returns
    ---------------------------------------------------------------------------
    len(files): integer
        The number of special hilbert spaces that have already been simulated.

    """

    files = ["../Data/" + ofile + "/" + file for file in os.listdir("../Data/" + ofile) if (file.lower().startswith('indtotuple'))]
    return len(files)


def CheckRedundant(signature):
    if OVERWRITE:
        return False
    
    # Initialize boolean flag and signature to check for 
    tag = str(signature)

    dir_ = "../Data/" + ofile + "/psi" + tag + "/"

    # check if the directory already exists
    if os.path.isdir(dir_):
        
        # check if it has all the data drops already
        files = [dir_ + file for file in os.listdir(dir_) if (file.lower().startswith('drop'))]
        if len(files) == frames + 1:
            return True 
    
    return False


def initSim():
    
    """
    This function initializes a SimObj Object, which contains all metadata
    including number of modes, timestep size, number of timesteps, output
    directory, and physical parameters like interaction constants.

    Returns
    ---------------------------------------------------------------------------
    s: SimObj instance
        An instance of the SimObj class.
    """
    
    # Initialize a blank simulation object
    s = S.SimObj() 

    # Populate with metadata
    s.N = N                         # number of modes
    s.dt = dt                       # time step
    s.omega0 = omega0               # kinetic constant, hbar/m
    s.C = C                         # long range interaction constant
    s.Lambda0 = lambda0             # constant interaction constant
    s.frames = frames               # number of data drops 
    s.framesteps = framesteps       # timesteps in between data drops
    s.ofile = ofile                 # name of data directory

    s.kord = np.arange(N) - N/2     # kmodes in natural order (as oppose to fft ordering)

    s.OVERWRITE = OVERWRITE         # should overwrite existing data drops

    return s

def initFQ(s, IC_, HS, sign):

    """
    This function initializes a FullQuantum object with the equivalent initial
    conditions as in the QIBS method.

    Parameters
    ---------------------------------------------------------------------------
    s: SimObj instance
        An Instance of the SimObj class.
    IC_: list or tuple
        A list-like containing the number of particles in each mode.
    HS: list or tuple
        A list-like object containing the special
    sign: str
        The signature for the state in the Hilbert space.
    """

    # Initialize the FullQuantum object
    fQ = FQ.QuantObj() 

    
    fQ.is_dispersion_quadratic = quad       # Boolean flag for quadratic dispersion
    fQ.second_Order = O2                    # Boolean flag for second order accurate solver
    fQ.E_m = s.kord                         # Modes in natural order (as opposed to fft order)
    fQ.IC = IC_                             # Occupations for one of the terms in the special hilbert space
   
    ntot = 0                                # Total number of particles (counter)
    ptot = 0                                # Total momentum (counter)

    # Count particles and momentum
    for i in range(len(IC_)):
        n = fQ.IC[i]
        ntot += n 
        ptot += n*i

    # Save signature as attribute of FullQuantum object
    # tag is a string
    # signature is a tuple of integers
    fQ.tag = str(sign) 
    fQ.signature = (ntot, ptot)

    # ------------ step 3a,b -------------- #
    fQ.SetOpsFromIC(s) 
    # ------------------------------------- #


    # ------------ step 3c -------------- #
    fQ.SetPsiHS(HS, IC, phi)
    # ----------------------------------- #


    # Flags for tracking certain variables
    fQ.track_psi = True         # Wavefunction
    fQ.track_EN = False         # Expectation of Number Operator

    return fQ


def FindDN(mu, P_goal=0.999):

    """
    This function finds the discrete values around the mean of a Poisson random
    variable such that at least P_goal=0.999 of the probability mass is being
    tracked.

    The values are returned as the difference from the expectation.

    Parameters
    ---------------------------------------------------------------------------
    mu: int
        The expected number of particles in a state

    Returns
    ---------------------------------------------------------------------------
    [lower - mu, sig]: [int, int]
        The lower and upper integers such that the probability mass contained 
        in and between these two values (plus mu) is greater than P_goal=0.999.
    """

    # Start counter by checking +/- 1 integer away from mean
    sig = 1

    # Initialize convergence boolean flag
    found = False
    while(not(found)):

        # Initialize Probability Mass Counter
        P = 0

        # If sig > mu, need to set lower bound to zero
        lower = np.max([0, mu - sig])

        # Add all probability mass between mu-sig and mu+sig
        for i in range(lower, mu + sig + 1):
            P += st.poisson.pmf(i, mu)

        # Check for convergence
        if P >= P_goal:
            found = True 

        # If not converged, check a wider set of values.
        else:
            sig += 1


    # Return lower and upper bound, expressed as difference from mean
    lower = np.max([0, mu - sig])
    return [lower - mu, sig]

def GetDNs():
        
    """
    This function uses FindDN to get the collection of the discrete values 
    around the mean of a Poisson random variable such that at least
    P_goal=0.999 of the probability mass is being tracked.

    The values are returned as the difference from the expectation.

    Returns
    ---------------------------------------------------------------------------
    Results: array-like
        An array containing the residuals from the expectation value.
    """

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

    """
    ANDREW TODO
    """

    DN_ = copyList(DN)
    if len(DN) > 0:
        DN__ = DN_.pop(0)
        for dn_ in range(DN__[1], DN__[0] - 1, -1):
            GetDNsR(DN_, dn + [dn_], Results)
    else:
        Results.append(dn)

def copyList(A):

    """
    This function just copies a list. It was written before the original author
    realized that the copy module had a deepcopy function.

    Parameters
    ---------------------------------------------------------------------------
    A: list
        A list of values

    Returns
    ---------------------------------------------------------------------------
    deepcopy(A): list
        A deepcopy of the original list, A.
    """
    import copy
    return copy.deepcopy(A)


def RunTerm(sign):
        
    """
    This function runs a simulation of a special Hilbert space with a
    particular signature.

    Parameters
    ---------------------------------------------------------------------------
    sign: ANDREW TODO
        The signature of the special Hilbert space.

    Returns
    ---------------------------------------------------------------------------
    1: int
        Boolean flag, success code.
    """

    # Initialize the simulation object
    s = initSim() 

    # Check if the special Hilbert space has already been simulated
    redundant_ = CheckRedundant(sign)
    
    # If it has not then begin the simulation
    if not(redundant_):
        
        # ------------ step 3a-c -------------- #

        # Initialize the full quantum solver
        fQ = initFQ(s, m.H_sp[sign][-1], m.H_sp[sign], sign) 
        # ------------------------------------- #

        # Add the full quantum solver to the simObj's solvers list
        s.solvers = [fQ] 

        # Create directories for data dumps
        fQ.ReadyDir(ofile) 

        # Start timer
        s.time0 = time.time()

        # ------------ step 3d,e -------------- #

        # Run the simulation 
        s.Run(verbose=False)
        # ------------------------------------- #

        # End the simulation
        s.EndSim(text = False) 

    done = FindDone()
    str_ = (('%.2f percent data created' % (100*float(done)/m.total) ) 
        + (' in %i hrs, %i mins, %i s' %u.hms(time.time()-m.time0)))
    u.repeat_print(str_)
    
    return 1


def GetSortedKeys(sigs):

    """
    This function sorts the keys in a particular way so that the run time
    approximately decreases monotonically (ANDREW TODO: confirm this).

    Parameters
    ---------------------------------------------------------------------------
    sigs: ANDREW TODO
        The signatures of the special Hilbert spaces.

    Returns
    ---------------------------------------------------------------------------
    rkeys: list
        The signatures of the special Hilbert spaces, now sorted.
    """

    ns = []
    keys = []

    for key_ in sigs.keys():
        ns.append(key_[0])
        keys.append(key_)

    ns = np.array(ns)
    keys = np.array(keys)

    sortedKeys = qu.sortE(ns, keys)

    rkeys = []
    for i in range(len(sortedKeys)-1,-1,-1):
        rkeys.append(tuple(sortedKeys[i]))

    return rkeys


def main():

    # Start a timer 
    m.time0 = time.time() 
    if is_root:
        print("Beginning sim", ofile)

    # ------------ step 1 --------------- #
    # Find the term expressed as differences from the expectation values
    DNs = GetDNs() 

    # Create a list of terms as list of number eigenstates
    terms = []
    for i in range(len(DNs)):
        IC_ = IC.copy()
        dn = DNs[i]
        for j in range(len(dn)):
            IC_[j] += dn[j]
        terms.append(IC_)
    # ----------------------------------- #

    # ----------- step 2 ---------------- #
    # Find the special Hilbert spaces (SHS) associated with these terms
    # for each term in the coherent state sum.
    for i in range(len(terms)):
        term = terms[i] 
        p = (np.arange(N)*np.array(term)).sum()     # Identify momentum
        n = np.array(term).sum()                    # Identify total particle number
        signature = (n,p)                           # Signature identifying a given SHS
        if not(signature in m.H_sp):                # If this SHS is not in the dictionary, 
            m.H_sp[signature] = [term]              # add it to the dictionary
            m.tags.append(str(signature))           # The tags list is used in data interpretation
        else:
            m.H_sp[signature].append(term)          # Add this term to the SHS
    # ----------------------------------- #

    # Gather total number of Hilbert spaces (used for timer)
    m.total = len(m.H_sp.keys())
    m.done = 0

    # -------------- step 3 ------------- #
    if is_root:
        print("Running simulation with %i sp Hilbert spaces." %(len(m.H_sp)))

    # Simulate each special Hilbert space in parallel
    for key in yt.parallel_objects( GetSortedKeys(m.H_sp), 0):
        print("\nWorking on key", key)
        sys.stdout.flush()
        RunTerm(key)

    # ----------------------------------- #

    # Start timer for I/O and Data Interpretation
    time1 = time.time()

    # Save tags
    tags_ = np.array(m.tags)
    np.save("../Data/" + ofile + "/" + "tags" + ".npy", tags_)

    if is_root:
        print("\nBeginning data interpretation")
    for i in range(len(dIs)):
        dIs[i].main(ofile, tags_, plot = False)
    if is_root:
        print('Analysis completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time1))

    if is_root:
        print('Script completed in %i hrs, %i mins, %i s' %u.hms(time.time()-m.time0))
    u.ding()
    sys.stdout.flush()

if __name__ == "__main__":
    main()
