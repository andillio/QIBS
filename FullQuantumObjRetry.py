import numpy as np
import scipy.stats as st
from scipy.stats import multinomial
try:
    import cupy as cp
except ImportError:
    pass
import utils as u
import time
import os
from os import path
import sys
import pickle
eps = sys.float_info.epsilon

class QuantObj(object):
    """
    This Quantum Object class stores all data, metadata, and contains several
    built-in methods necessary for the simulation as well as some utility
    functions.
    """
    def __init__(self):

        self.indToTuple = {}
        self.tupleToInd = {}

        # Full quantum state (wavefunction)
        self.psi = None 

        # Simulation object
        self.s = None 

        # Number operators
        self.Num = None 

        # Hamiltonian operators
        self.W = None
        self.inds = None 

        # Momentum for each mode
        self.E_m = None

        self.IC = None              # Initial conditions (i.e. initial mode occupation)
        self.E_tot = None           # Total energy of state
        self.N_p = None             # Total number of particles
        self.signature = None 
    
        # Boolean to record whether numpy or cupy is being used
        self.is_np = True 

        self.is_dispersion_quadratic = False # is the dispersion relation quadratic (as opposed to linear)
        self.second_Order = True # should I include second order intergration terms

        # Booleans to flag which variables are evolved/solved/tracked
        self.track_psi = False              # track the wavefunction
        self.track_EN = True                # track the expectation values of number operators
        self.track_rho = False              # track the density matrix

        self.tag = "" # output tag
        
        # Boolean to flag whether simulation is working
        self.working = True

    # given SimObj s
    def SetOpsFromIC(self, s):

        self.s = s

        # determine what the momentum for each mode is
        if self.E_m is None:
            self.E_m = np.arange(len(self.IC))
        self.E_tot = (self.IC*self.E_m).sum()
        self.N_p = np.sum(self.IC)
        self.signature = (self.N_p, (self.IC*np.arange(len(self.IC))).sum() )

        # ------------ step 3a -------------- #
        # find the allowed state space, i.e. the special Hilbert space
        self.InspectStatesR()
        # ----------------------------------- #

        # ------------ step 3b -------------- #
        # set the number and hamiltonian operators
        self.SetOps(s)
        # ----------------------------------- #

        # set the wavefunction assuming the wavefunction is a single number eigenstate
        # with occupations IC
        self.SetPsi()

    # recursive version of inspectstates
    # finds the allowed state space, ie The Hilbert Space
    def InspectStatesR(self):
        N_m = len(self.IC)
        E = (np.arange(N_m)*self.IC).sum() # get the amount of momentum our state has
        #E = (self.E_m*self.IC).sum() # get the amount of momentum our state has
        E_max = self.IC.sum()*(N_m-1)

        inds = []
        self.InspectStatesP(inds, E_max - E)

    # (private) partner function
    # given the indices (inds) of the previous for loops,
    # and difference between the new maximum possible energy and the requried energy
    def InspectStatesP(self, inds, dE):
        m = len(inds) # the current mode being populated
        n_used = sum(inds) # number of particles already put into modes
        N_m = len(self.IC)
        N_p = self.IC.sum()

        # put i particles in the mode m 
        for i in range(N_p+1-n_used):
            new_dE = dE - (N_m - m - 1)*i # new max energy excess
            
            # if have occupied enough modes to know the final state
            if (m == N_m - 2):

                # if it has the correct energy
                if (new_dE == 0):
                    # it is a valid state
                    ind = len(self.indToTuple)
                    tuple_ = tuple(inds + [i] + [N_p - n_used - i])
                    self.indToTuple[ind] = tuple_
                    self.tupleToInd[tuple_] = ind

            else:
                # if I populate the particles in this way,
                # is the new max possible energy excess at least 0?				
                if (new_dE >= 0):
                    # then keep populating modes
                    self.InspectStatesP(inds + [i], new_dE)
        
    def SetOps(self, s):
        '''
        This method finds the Hamiltonian operators 

        Parameters
        -----------------------------------------------------------------------
        s : SimObj instance
          An instance of the SimObj class.
        '''

        N_m = len(self.IC)
        N_s = len(self.indToTuple)

        self.Num = np.zeros((N_m, N_s))

        # Key: state_i
        # Value: list of states that are coupled to this state via H 
        indToInds = {}

        # Key: state_i
        # Value: list of corresponding weights for each state in indToInds
        indToW = {}

        for i in range(N_s):
            tuple_i = self.indToTuple[i]    # corresponds to occupations in tuple_i
            np_i =np.asarray(tuple_i)       # this is the number eigenstate of the ket
            E_i = (self.E_m*np_i).sum()     # I say E, but its really momentum

            for m in range(len(np_i)):
                # the number operator on the mth mode has
                # eigenvalue tuple_i[m]
                # when acting upon the ith state in The Hilber Space
                self.Num[m, i] = tuple_i[m]
            
            T = s.omega0 * E_i            
            if self.is_dispersion_quadratic:
                T = s.omega0 * ((self.E_m**2 / 2.) * np_i).sum() 


            Inds = [i]
            Weights = [T]

            for k in range(N_m**3):
                W, j = self.GetWeight(np_i, k, s)
                
                if np.abs(W) > 0:
                    if j in Inds:
                        indexer = Inds.index(j)
                        Weights[indexer] += W
                    else:
                        Inds.append(j)
                        Weights.append(W)

            indToInds[i] = Inds
            indToW[i] = Weights

        largest = 0

        for i in range(len(indToInds)):
            Inds_ = indToInds[i]
            if len(Inds_) > largest:
                largest = len(Inds_)
        
        self.W = np.zeros((largest, N_s))
        self.inds = np.zeros((largest, N_s)).astype(int)

        for i in range(len(indToInds)):
            inds_ = indToInds[i]
            W_ = indToW[i]
            
            for j in range(len(inds_)):
                self.W[j,i] = W_[j]
                self.inds[j,i] = inds_[j]


    def GetWeight(self,np_i, k, s):
        '''
        This method returns (ANDREW TODO: complete)

        Parameters
        -----------------------------------------------------------------------
        k : int
          1D index for special hilbert space (ANDREW TODO: confirm)
        s : SimObj instance
          An instance of the SimObj class. 
        '''
        N_m = len(self.IC) # number of modes

        # express k in base N_m
        ind1 = k // N_m**2 # index on first operator, b
        ind2 = (k%N_m**2) // N_m # index on second operator, b
        ind3 = k%N_m # a
        ind4 = ind1 + ind2 - ind3 # a

        if not(s.ValidIndex(ind4)):
            return  0, 0

        np_f = np_i.copy() # number eigenstate of the bra

        #W = s.Lambda0 / 2.
        W = s.GetLam(ind1, ind2, ind3, ind4) / 2.

        # right most annihilation op
        W *= np.sqrt(np_f[ind4])
        np_f[ind4] -= 1

        # second annihilation op
        if (np_f[ind3] >= 0):
            W *= np.sqrt(np_f[ind3])
            np_f[ind3] -= 1
        else:
            return 0, 0

        # first creation op
        if (np_f[ind2] >= 0):
            W *= np.sqrt(np_f[ind2] + 1)
            np_f[ind2] += 1
        else:
            return 0, 0    

        # left most creation op
        if (np_f[ind1] >= 0):
            W *= np.sqrt(np_f[ind1] + 1)
            np_f[ind1] += 1
        else:
            return 0,0
        
        if -1 in np_f:
            return 0, 0

        # bra state stuple
        np_f = tuple(np_f)

        # if it is a valid state return the weight and bra index
        if np_f in self.tupleToInd:
            return W, self.tupleToInd[np_f]
        else:
            # this shouldn't happen
            print("exited special hilbert space", np_f)
            return 0, 0
        

    # set psi given states in other the special hilbert space, HS
    # ICs, |z| for simulation coherent state defined by param z
    # phi, angle(z) for simulation coherent state defined by param z
    def SetPsiHS(self, HS, IC, phi):

        '''
        This method takes a given special hilbert space, initial conditions,
        and phases and builds up the wavefunction (self.psi) via reductions.

        Parameters
        -----------------------------------------------------------------------
        HS: array-like
          States in special Hilbert space
        IC: array-like
          Initial conditions
        phi: array-like
          Phases
        '''

        N_s = len(self.indToTuple)

        self.psi = np.zeros(N_s) + 0j

        # loop over the states in this special Hilbert space
        for j in range(len(HS)):
            state_ = tuple(HS[j])
            ind = self.tupleToInd[state_]

            P = 1.
            phase = 1.

            for i in range(len(state_)):
                n = state_[i]
                P *= st.poisson.pmf(n, IC[i])
                phase *= np.exp(1j * n * phi[i]) 

            # assign the appropriate phase and amplitude
            self.psi[ind] += np.sqrt(P)*phase
            

    def SetPsiHS_mn(self, HS, IC, phi):
        '''
        This method takes a given special hilbert space, initial conditions,
        and phases and builds up the wavefunction (self.psi) via reductions.

        (ANDREW TODO:) For multinomial (?).

        Parameters
        -----------------------------------------------------------------------
        HS: array-like
          States in special Hilbert space
        IC: array-like
          Initial conditions
        phi: array-like
          Phases
        '''

        N_s = len(self.indToTuple)

        self.psi = np.zeros(N_s) + 0j

        mn = multinomial(IC.sum(), IC*1./IC.sum())

        for j in range(len(HS)):
            state_ = tuple(HS[j])
            ind = self.tupleToInd[state_]

            P = mn.pmf(np.array(state_))
            phase = 1.

            for i in range(len(state_)):
                n = state_[i]
                phase *= np.exp(1j * n * phi[i]) 

            # assign the appropriate phase and amplitude
            self.psi[ind] += np.sqrt(P)*phase   

    
    def SetPsi(self):

        '''
        Initializes the wavefunction for number eigenstates.
        '''

        N_s = len(self.indToTuple)

        self.psi = np.zeros(N_s) + 0j

        ind = self.tupleToInd[tuple(self.IC)]

        self.psi[ind] = 1

    def ToCUPY(self):
        '''
        Converts all stored variables to CuPY arrays
        '''
        self.psi = cp.asarray(self.psi)
        self.E_m = cp.asarray(self.E_m)
        self.Num = cp.asarray(self.Num)
        self.W = cp.asarray(self.W)
        self.inds = cp.asarray(self.inds)
        self.is_np = False

    def ToNUMPY(self):
        '''
        Converts all stored variables to numpy arrays
        '''
        self.psi = cp.asnumpy(self.psi)
        self.E_m = cp.asnumpy(self.E_m)
        self.Num = cp.asnumpy(self.Num)
        self.W = cp.asnumpy(self.W)
        self.inds = cp.asnumpy(self.inds)
        self.is_np = True

        self.is_np = True


    def stateMul(self, H, psi):
        '''
        This method returns mat-vec operation H.psi

        Parameters
        -----------------------------------------------------------------------
        H: 2-D array-like
          Hamiltonian
        psi: 1-D array-like
          Wavefunction

        Returns
        -----------------------------------------------------------------------
        H.psi: 1-D array-like
          Hamiltonian operating on wavefunction.
        '''
        if self.is_np:
            return np.einsum("fi,i->f", H, psi)
        return cp.einsum("fi,i->f", H, psi)

    def opMul(self, H, H_):
        '''
        This method returns mat-mat operation H.H_

        Parameters
        -----------------------------------------------------------------------
        H: 2-D array-like
          Hamiltonian (or other 2D matrix)
        H_: 2-D array-like
          Hamiltonian (or other 2D matrix)

        Returns
        -----------------------------------------------------------------------
        H.H_: 1-D array-like
          Hamiltonian operating on itself (or another matrix).
        '''
        if self.is_np:
            return np.einsum("ij,jk->ik", H, H_) 
        return cp.einsum("ij,jk->ik", H, H_)

    def expectation(self, N, psi):
        '''
        This method finds the expectation value of the number operator (or
        other matrix), and returns the real part

        Parameters
        -----------------------------------------------------------------------
        N: 2-D array-like
          Number operator (or other 2D matrix)
        psi: 1-D array-like
          Wavefunction

        Returns
        -----------------------------------------------------------------------
        real(conjugate(psi).N.psi): 1-D array-like
          Expectation value of N: < N > = < psi | N | psi >
        '''

        if self.is_np:
            return (np.conj(psi)*N*psi).sum().real
        return (cp.conj(psi)*N*psi).sum().real
    
    def NumExpectations(self):
        '''
        This method finds the expectation value of the number operator (or
        other matrix)

        Parameters
        -----------------------------------------------------------------------
        N: 2-D array-like
          Number operator (or other 2D matrix)
        psi: 1-D array-like
          Wavefunction

        Returns
        -----------------------------------------------------------------------
        conjugate(psi).N.psi: 1-D array-like
          Expectation value of N: < N > = < psi | N | psi >
        '''
        if self.is_np:
            return (np.einsum('i,ji,i->j', np.conj(self.psi), self.Num, self.psi )).real
        return (cp.einsum('i,ji,i->j', cp.conj(self.psi), self.Num, self.psi )).real

    def innerProd(self, psi_, N, psi):
        '''
        This method finds the expectation value of the number operator (or
        other matrix)

        Parameters
        -----------------------------------------------------------------------
        N: 2-D array-like
          Number operator (or other 2D matrix)
        psi: 1-D array-like
          Wavefunction

        Returns
        -----------------------------------------------------------------------
        conjugate(psi).N.psi: 1-D array-like
          Expectation value of N: < N > = < psi | N | psi >
        '''
        if self.is_np:
            return (np.conj(psi_)*N*psi).sum()
        return (cp.conj(psi_)*N*psi).sum()


    def CheckRedundant(self, s):
        if s.OVERWRITE:
            return False

        file = "../Data/" + s.ofile + "/psi" + self.tag + "/" + "drop" + str(s.currentFrame) + ".npy" 
        if path.exists(file):
            try:
                self.psi = np.load(file)
                return True
            except:
                #os.system(f'rm {file}')
                return False
        else:
            return False

    def Update(self, dt, s):
        '''
        This method handles the evolution of the wavefunction, given a timestep
        and an instance of the SimObj class.

        Parameters
        -----------------------------------------------------------------------
        dt: float
          The timestep to take
        s: SimObj instance
          An instance of SimObj class, which includes all metadata required for
          the simulation.
        '''
        redundant_ = self.CheckRedundant(s)

        if not(redundant_):
            # integrate Schroedinger's equation
            for i in range(s.framesteps):
                
                if self.second_Order:
                    dpsi_dt = (self.W * self.psi[self.inds]).sum(axis = 0)
                    self.psi -= 1j*dt*dpsi_dt + dt*dt*(self.W * dpsi_dt[self.inds]).sum(axis = 0)/2.
                else:
                    self.psi -= 1j*dt*(self.W * self.psi[self.inds]).sum(axis = 0)
        

    def ReadyDir(self,ofile):
        '''
        Given a path, this method creates the Data directory hierarchy
        required for all quantities that are to be tracked. 

        Parameters
        -----------------------------------------------------------------------
        ofile: str
          Location where hierarchy is to be made (within ../Data/)
        '''
        if self.track_psi:
            try:
                os.mkdir("../Data/" + ofile + "/psi" + self.tag)
            except OSError:
                pass
        if self.track_EN:
            try:
                os.mkdir("../Data/" + ofile + "/Num" + self.tag)
            except OSError:
                pass
        if (self.track_psi):
            with open("../Data/" + ofile + "/" + "indToTuple" + self.tag + ".pkl", 'wb') as f:
                pickle.dump(self.indToTuple, f, pickle.HIGHEST_PROTOCOL)
            with open("../Data/" + ofile + "/" + "tupleToInd" + self.tag + ".pkl", 'wb') as f:
                pickle.dump(self.tupleToInd, f, pickle.HIGHEST_PROTOCOL)

    def DataDrop(self,i,ofile_):
        '''
        This method dumps the quantities tracked to disk.

        Parameters
        -----------------------------------------------------------------------
        i: int
          Integer which tracks which time step (or data dump) is to be dumped
        ofile: str
          Location to dump to (within ../Data)
        '''

        # output psi        
        if self.track_psi:
            if not(self.is_np):
                cp.save("../Data/" + ofile_ + "/psi" + self.tag + "/" + "drop" + str(i) + ".npy", self.psi)
            else:
                np.save("../Data/" + ofile_ + "/psi" + self.tag + "/" + "drop" + str(i) + ".npy", self.psi)

        # output expecations
        if self.track_EN:
            Nums = self.NumExpectations()
            if not(self.is_np):
                cp.save("../Data/" + ofile_ + "/Num" + self.tag + "/" + "drop" + str(i) + ".npy", Nums)
            else:
                np.save("../Data/" + ofile_ + "/Num" + self.tag + "/" + "drop" + str(i) + ".npy", Nums)
