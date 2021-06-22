import matplotlib.pyplot as plt 
import numpy as np 
import utils as u
import QUtils as qu

#simNames = ["Gr_r1","Gr_r2","Gr_r3", "Gr_r4","Gr_r5", "Gr_r6", "Gr_r7", 
#"Gr_r8", "Gr_r9", "Gr_r10"]
simNames = ["Sin_h1_r1"]
t_dyn = 1./np.sqrt(.1*5)

class figObj(object):

    def __init__(self):
        self.fig = None 
        self.ax1 = None 
        self.ax2 = None 
fo = figObj() 

def makeFig():
    fo.fig, axs = plt.subplots(2,1,figsize = (6,12))
    fo.ax1 = axs[0]
    fo.ax2 = axs[1]

    fo.ax2.set_xlabel(r"$n_{tot}$")
    fo.ax1.set_ylabel(r"$t_{br} \, [ t_{d}]$")
    fo.ax1.set_ylabel(r"$r_{max}$")

def PlotStuff(simName, color, ax, ax2):

    t = np.load("../Data/" + simName + "/_t.npy")
    aa = np.load("../Data/" + simName + "/_aa.npy")
    a = np.load("../Data/" + simName + "/_a.npy")
    N = np.load("../Data/" + simName + "/_N.npy")
    n = np.abs(np.sum(N[0]))

    er = np.zeros((len(t), len(N[0])))

    for i in range(len(t)):
        t_ = t[i]
        aa_ = np.diag(aa[i])
        N_ = N[i]
        a_ = a[i]

        dbda = N_ - a_*np.conj(a_)
        dada = aa_ - a_*a_

        er_ = 2*dbda - 2*np.abs(dada) + 1.
        er[i,:] = er_

    er = qu.sortE(t,er)
    N = qu.sortE(t,N)
    t = qu.sortE(t,t)
    sq = er[:,2]#np.sum(er*N, axis = 1)/n

    plt.plot(t,sq)
    plt.show()

    argMin = np.argmin(sq)
    t_br = t[argMin]
    sq_min = sq[argMin]
    r_max = np.log(sq_min)/-2.

    ax.plot([n],[t_br],color)
    ax2.plot([n],[r_max],color)
    ax2.plot([n], [np.log(n**(1/6.))], 'ko')

    return t_br, n, t_dyn


def PlotList(names,t_br, n, color, label, ax, ax2):

    t_br_, N = [], []

    t_dyn = 0.
    for i in range(len(names)):
        t_, n_, t_dyn = PlotStuff(names[i], color, ax, ax2)
        t_br.append(t_)
        t_br_.append(t_)
        n.append(n_)
        N.append(n_)

    N = np.array(N)
    t_br_ = np.array(t_br_)

    ax.plot([],[],color,label = label)
    ax.legend()

    return t_dyn


if __name__ == "__main__":
    u.orientPlot()
    makeFig()

    t_br = []
    n = []

    PlotList(simNames,t_br, n, 'bo', r'CoherentState', fo.ax1, fo.ax2)

    t_sq = .6/(5*.1)
    #r_max = np.log(n_max**(1./6))
    #y_min = np.exp(-2*r_max)
    fo.ax1.set_ylim(0, np.max(t_sq)*2)
    fo.ax1.plot([0.,np.max(n)*1.05],[t_sq, t_sq], 'k--', alpha = 1.)

    fo.ax1.set_xlim(0,np.max(n)*1.05)
    fo.ax2.set_xlim(0,np.max(n)*1.05)

    plt.subplots_adjust(wspace=.0, hspace=0.)
    plt.savefig("../Figs/breaktimesSq.pdf", bbox_inches = "tight")
    plt.show()