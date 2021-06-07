import scipy as sp
import numpy as np 
import cupy as cp 
import utils as u
import time
import sys
import scipy.linalg as sl
import numpy.linalg as nl 
eps = sys.float_info.epsilon

def Psi2Rho(psi):
    return np.einsum('i,j->ij', psi, np.conj(psi))


def TraceOutModes(rho, indToTuple, tupleToInd, m):
    if len(m) == 0:
        return rho, indToTuple, tupleToInd
    m = np.array(m)
    m.sort()
    newrho, newIndToTuple, newTupleToInd = TraceOutMode(rho, indToTuple, tupleToInd, m[0])
    for i in range(len(m)-2,-1,-1):
        newrho, newIndToTuple, newTupleToInd = TraceOutMode(newrho, newIndToTuple, newTupleToInd, m[i])
    return newrho, newIndToTuple, newTupleToInd


# returns a quantum state which is equal to this quantum state
# with the mode m traced over
def TraceOutMode(rho, indToTuple, tupleToInd, m):
    N_m = len(indToTuple[0])-1

    r = np.arange(N_m+1)

    # create the new ind-tuple mapping
    newIndToTuple = {}
    newTupleToInd = {}

    states = 0

    for i in range(len(indToTuple)):
        oldTup = indToTuple[i]
        
        newTup = []

        for j in range(len(oldTup)):
            if j != m:
                newTup.append(oldTup[j])

        
        newTup = tuple(newTup)

        # if this state is not yet in the dictionary
        # then add it to the dictionary
        if newTupleToInd.get(newTup) == None:
            newTupleToInd[newTup] = states
            newIndToTuple[states] = newTup
            states += 1

    N_s = len(newIndToTuple)

    # now loop through the reduced hilbert space and create
    # the density matrix and other operators

    newRho = np.zeros((N_s,N_s)) + 0j

    # set state values
    for i in range(len(indToTuple)):
        old_state_i = indToTuple[i]
        old_np_i = np.asarray(old_state_i)

        new_np_i = old_np_i[r != m]
        new_state_i = tuple(new_np_i)
        new_i = newTupleToInd[new_state_i]

        for j in range(len(indToTuple)):
            old_state_f = indToTuple[j]
            old_np_f = np.asarray(old_state_f)

            if (old_state_i[m] == old_state_f[m]):

                new_np_f = old_np_f[r != m]
                new_state_f = tuple(new_np_f)
                new_j = newTupleToInd[new_state_f]

                newRho[new_i, new_j] += rho[i,j]

    return newRho, newIndToTuple, newTupleToInd

def stateMul(H, psi):
    return np.einsum("fi,i->f", H, psi)

def opMul(H, H_, CUPY = False):
    if CUPY:
        return cp.einsum("ij,jk->ik", H, H_) 
    return np.einsum("ij,jk->ik", H, H_) 

def S_VN(rho):
    ln_rho = sl.logm(rho)
    return -np.trace( opMul(rho, ln_rho) ).real

# linear entropy
def S_lin(rho, CUPY = False):
    if CUPY:
        return (1. - cp.trace( opMul(rho, rho, True) )).real
    return (1. - np.trace( opMul(rho, rho) )).real

def S_linAlt(rho, CUPY = False):
    if CUPY:
        return (1. - (cp.abs(rho)**2).sum() ).real
    return (1. - (np.abs(rho)**2).sum() ).real


def PsiToReduceRho(psi, indToTuple, tupleToInd, reduce_, CUPY = False):
    # create the new ind-tuple mapping
    newIndToTuple = {}
    newTupleToInd = {}

    time0 = time.time()

    states = 0

    for i in range(len(indToTuple)):
        oldTup = indToTuple[i]
        
        newTup = []

        for j in range(len(oldTup)):
            if not(j in reduce_):
                newTup.append(oldTup[j])
        
        newTup = tuple(newTup)

        # if this state is not yet in the dictionary
        # then add it to the dictionary
        if newTupleToInd.get(newTup) == None:
            newTupleToInd[newTup] = states
            newIndToTuple[states] = newTup
            states += 1

    N_sNew = len(newIndToTuple)

    newRho = None 
    if CUPY:
        newRho = cp.zeros((N_sNew,N_sNew)) + 0j
    else:
        newRho = np.zeros((N_sNew,N_sNew)) + 0j

    # set state values
    # NOTE: this loop takes ~ 20 hrs in its present form
    for i in range(len(indToTuple)):
        old_state_i = indToTuple[i]

        new_state_i = []
        for j in range(len(old_state_i)):
            if not(j in reduce_):
                new_state_i.append(old_state_i[j])
        new_state_i = tuple(new_state_i)
        new_i = newTupleToInd[new_state_i]

        for j in range(len(indToTuple)):
            old_state_j = indToTuple[j]

            compatible = True 
            for k in range(len(reduce_)):
                compatible = compatible and (old_state_i[k] == old_state_j[k])

            if compatible:
                new_state_f = []
                for k in range(len(old_state_j)):
                    if not(k in reduce_):
                        new_state_f.append(old_state_j[k])
                new_state_f = tuple(new_state_f)
                new_j = newTupleToInd[new_state_f]

                if CUPY:
                    newRho[new_i, new_j] += psi[i]*cp.conj(psi[j])
                else:
                    newRho[new_i, new_j] += psi[i]*np.conj(psi[j])

    return newRho, newIndToTuple, newTupleToInd


def GetReducedDicts(indToTuple, tupleToInd, reduce_):
    # create the new ind-tuple mapping
    newIndToTuple = {}
    newTupleToInd = {}

    states = 0

    for i in range(len(indToTuple)):
        oldTup = indToTuple[i]
        
        newTup = []

        for j in range(len(oldTup)):
            if not(j in reduce_):
                newTup.append(oldTup[j])
        
        newTup = tuple(newTup)

        # if this state is not yet in the dictionary
        # then add it to the dictionary
        if newTupleToInd.get(newTup) == None:
            newTupleToInd[newTup] = states
            newIndToTuple[states] = newTup
            states += 1
    
    return newIndToTuple, newTupleToInd


def PsiToReduceRhoSmart(psi, indToTuple, tupleToInd, newIndToTuple, newTupleToInd, reduceMap, reduce_, CUPY = False, timed = False):
    N_sNew = len(newIndToTuple)

    newRho = None 
    if CUPY:
        newRho = cp.zeros((N_sNew,N_sNew)) + 0j
    else:
        newRho = np.zeros((N_sNew,N_sNew)) + 0j

    time0 = time.time()
    done = 0

    for key_ in reduceMap:

        inds = reduceMap[key_]

        for i in range(len(inds)):
            ind_i = inds[i]
            old_state_i = indToTuple[ind_i]
            
            new_state_i = []
            for j in range(len(old_state_i)):
                if not(j in reduce_):
                    new_state_i.append(old_state_i[j])
            new_state_i = tuple(new_state_i)
            new_i = newTupleToInd[new_state_i]

            for j in range(i, len(inds)):
                ind_j = inds[j]
                old_state_j = indToTuple[ind_j]

                new_state_f = []
                for k in range(len(old_state_j)):
                    if not(k in reduce_):
                        new_state_f.append(old_state_j[k])
                new_state_f = tuple(new_state_f)
                new_j = newTupleToInd[new_state_f]

                if CUPY:
                    contr_ = psi[ind_i]*cp.conj(psi[ind_j])
                    newRho[new_i, new_j] += contr_
                    if new_i != new_j:
                        newRho[new_j, new_i] += cp.conj(contr_)
                else:
                    contr_ = psi[ind_i]*np.conj(psi[ind_j])
                    newRho[new_i, new_j] += contr_
                    if new_i != new_j:
                        newRho[new_j, new_i] += np.conj(contr_)
        
        done += 1

        if timed:
            u.PrintTimeUpdate(done, len(reduceMap),time0)
    
    if timed:
        u.PrintCompletedTime(time0)
    
    return newRho


def GetFieldOps(indToTuple, tupleToInd):
    N_m = len(indToTuple[0])
    N_s = len(indToTuple)

    a = np.zeros((N_m, N_s, N_s))

    for i in range(N_s):
        state_i = np.array(indToTuple[i])

        for j in range(N_m):
            state_j = state_i.copy()
            state_j[j] -= 1
            tup_j = tuple(state_j)

            if (tup_j in tupleToInd):
                ind_j = tupleToInd[tup_j]

                a[j,ind_j,i] = np.sqrt(state_i[j])
    
    return a


def GetFieldExp(psi, indToTuple, tupleToInd, m):
    N_s = len(indToTuple)

    exp = 0 + 0j

    for i in range(N_s):
        state_i = np.array(indToTuple[i])
        state_j = state_i.copy()
        state_j[m] -= 1
        tup_j = tuple(state_j)

        if (tup_j in tupleToInd):
            ind_j = tupleToInd[tup_j]

            exp += np.sqrt(state_i[m])*psi[i]*np.conj(psi[ind_j])

    return exp

def GetNumExp(psi, indToTuple, tupleToInd, m):
    N_s = len(indToTuple)

    exp = 0

    for i in range(N_s):
        state_i = np.array(indToTuple[i])

        exp += state_i[m]*np.abs(psi[i])**2

    return exp