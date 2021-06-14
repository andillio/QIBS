import numpy as np
import scipy.stats as st
try:
    import cupy as cp
except ImportError:
    pass
import utils as u
import time
import os
import sys
import pickle
eps = sys.float_info.epsilon

class QuantObj(object):

    # given number of modes N
    def __init__(self):

        self.indToTuple = {}
        self.tupleToInd = {}

        # full quantum state
        self.psi = None # wavefunction

        self.s = None # sim object

        # number operators
        self.Num = None 

        # Hamiltonian operators
        self.W = None
        self.inds = None 

        # Momentum for each mode
        self.E_m = None

        self.IC = None # initial conditions (i.e. initial mode occupation)
        self.E_tot = None # total energy of state
        self.N_p = None # total number of particles
        self.signature = None 
    
        self.is_np = True # is using numpy or cupy

        self.is_dispersion_quadratic = False # is the dispersion relation quadratic (as opposed to linear)
        self.second_Order = True # should I include second order intergration terms

        # keep track of variables
        self.track_psi = False # track the wavefunction
        self.track_EN = True # track the expectation values of number operators
        self.track_rho = False # track the density matrix

        self.tag = "" # output tag
        
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
        
    # finds the Hamiltonian and Number operators in The Hilbert Space
    # given SimObj s
    def SetOps(self, s):
        N_m = len(self.IC)
        N_s = len(self.indToTuple)

        self.Num = np.zeros((N_m, N_s))

        # key: state_i
        # value: list of states that are coupled to this state via H 
        indToInds = {}
        # key: state_i
        # value: list of corresponding weights for each state in indToInds
        indToW = {}

        for i in range(N_s):
            tuple_i = self.indToTuple[i] # corresponds to occupations in tuple_i
            np_i =np.asarray(tuple_i) # this is the number eigenstate of the ket
            E_i = (self.E_m*np_i).sum() # I say E, but its really momentum

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
            


    
    def SetPsi(self):
        N_s = len(self.indToTuple)

        self.psi = np.zeros(N_s) + 0j

        ind = self.tupleToInd[tuple(self.IC)]

        self.psi[ind] = 1

    def ToCUPY(self):
        self.psi = cp.asarray(self.psi)
        self.E_m = cp.asarray(self.E_m)
        self.Num = cp.asarray(self.Num)
        self.W = cp.asarray(self.W)
        self.inds = cp.asarray(self.inds)
        self.is_np = False

    def ToNUMPY(self):
        self.psi = cp.asnumpy(self.psi)
        self.E_m = cp.asnumpy(self.E_m)
        self.Num = cp.asnumpy(self.Num)
        self.W = cp.asnumpy(self.W)
        self.inds = cp.asnumpy(self.inds)
        self.is_np = True

        self.is_np = True


    def stateMul(self, H, psi):
        if self.is_np:
            return np.einsum("fi,i->f", H, psi)
        return cp.einsum("fi,i->f", H, psi)

    def opMul(self, H, H_):
        if self.is_np:
            return np.einsum("ij,jk->ik", H, H_) 
        return cp.einsum("ij,jk->ik", H, H_)

    def expectation(self, N, psi):
        if self.is_np:
            return (np.conj(psi)*N*psi).sum().real
        return (cp.conj(psi)*N*psi).sum().real
    
    def NumExpectations(self):
        if self.is_np:
            return (np.einsum('i,ji,i->j', np.conj(self.psi), self.Num, self.psi )).real
        return (cp.einsum('i,ji,i->j', cp.conj(self.psi), self.Num, self.psi )).real

    def innerProd(self, psi_, N, psi):
        if self.is_np:
            return (np.conj(psi_)*N*psi).sum()
        return (cp.conj(psi_)*N*psi).sum()


    def Update(self, dt, s):
        
        # integrate Schroedinger's equation
        for i in range(s.framesteps):
            
            if self.second_Order:
                dpsi_dt = (self.W * self.psi[self.inds]).sum(axis = 0)
                self.psi -= 1j*dt*dpsi_dt + dt*dt*(self.W * dpsi_dt[self.inds]).sum(axis = 0)/2.
            else:
                self.psi -= 1j*dt*(self.W * self.psi[self.inds]).sum(axis = 0)
                    
    def ReadyDir(self,ofile):
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
