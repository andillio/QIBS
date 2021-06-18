import matplotlib.pyplot as plt 
import numpy as np 
import utils as u

simNames = ["Gr_r1","Gr_r2","Gr_r3","Gr_r4", "Gr_r5"]
t_dyn = 1./np.sqrt(.1*5)


class figObj(object):

    def __init__(self):
        self.fig = None 
        self.ax1 = None 
fo = figObj() 

def makeFig():
    fo.fig, axs = plt.subplots(1,1,figsize = (6,6))
    fo.ax1 = axs

    fo.ax1.set_xlabel(r"$n_{tot}$")
    fo.ax1.set_ylabel(r"$t_{br} \, [ t_{d}]$")

def PlotStuff(simName, color, ax, r):

    n = r*5

    t = np.load("Data/" + simName + "/_t.npy")
    Q = np.load("Data/" + simName + "/_Q.npy")

    t_br = np.interp(.15, Q, t) / t_dyn

    ax.plot([n],[t_br],color)

    return t_br, n, t_dyn


def PlotList(names,t_br, n, color, label, ax):

    t_br_, N = [], []

    t_dyn = 0.
    for i in range(len(names)):
        t_, n_, t_dyn = PlotStuff(names[i], color, ax, i+1)
        t_br.append(t_)
        t_br_.append(t_)
        n.append(n_)
        N.append(n_)

    N = np.array(N)
    t_br_ = np.array(t_br_)

    ax.plot([],[],color,label = label)

    return t_dyn


if __name__ == "__main__":
    u.orientPlot()
    makeFig()

    t_br = []
    n = []

    t_dyn = PlotList(simNames,t_br, n, 'bo', r'CoherentState', fo.ax1)