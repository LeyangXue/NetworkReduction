# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:19:56 2022

@author: Leyang Xue

"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import powerlaw

def RankInDict(Dict):
    
    #record the edge index in the Reducted network and then generate the rank according to the edge Score   
    values= [] #values in dict
    keysInx = {} #keys index in dict
    for inx, (key,value) in enumerate(Dict.items()):
        keysInx[inx] = key 
        values.append(value)
    rankinx = np.argsort(-np.array(values)) #sort from large to small
    
    return keysInx, rankinx

def CountFrequence(Wnewnodes):
    
    wnodefreq = {}
    for node in Wnewnodes.keys():
        Nw = Wnewnodes[node]
        if wnodefreq.get(Nw)==None:
            wnodefreq[Nw] = 1
        else:
            wnodefreq[Nw] = wnodefreq[Nw]+1
    
    return wnodefreq

def DegreeDistribution(netIndex,path):
    
    Scliques = [2,3,4,5,6,7,8,9]
    KwSeq={}
    for i,sclique in enumerate(Scliques):
        edgelist = cg.load(path +'/'+str(netIndex[sclique])+'_reducedEdgelist')
        RG = cg.weightNetwork(edgelist)
        degree = cg.NodeWeightDegree(RG)
        #kprob[sclique]= ProbaDistrib(degree)
        KwSeq[sclique] = degree 
    
    edgelist = cg.load(path +'/0_reducedEdgelist')
    RG = cg.weightNetwork(edgelist)
    degree = cg.NodeWeightDegree(RG)
    KwSeq[0] = degree 

    return KwSeq

def PlotAxes(ax,xlabel,ylabel,title='',fontsize=20, n_legend = 18, mode=True):
    
    font_label = {'family': "Calibri", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=fontsize)
    if mode == True:
        ax.legend(loc='best',framealpha=0, fontsize=n_legend)
        
def PlotNetStruc(figurepath,networkpath):
    
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex, netNweight]= cg.NetworkInx(sizeCliques,networkpath)
    KwSeq= DegreeDistribution(netIndex,networkpath)

    colors = plt.get_cmap('Paired')
    n_legend = 16
    ms = 10
    mew = 2
    alp = 1
    labels = list(map(lambda x:int(x), list(netNweight.keys())))
    labels.append('original')

    fig,ax= plt.subplots(1,3,figsize=(21,7),constrained_layout=True)
    
    #axes 1
    for i,(sclique,Seq) in enumerate(KwSeq.items()):
        #print(i,sclique,Seq)
        kwSeq = list(Seq.values())
        fit = powerlaw.Fit(kwSeq, discrete = True)
        [xmin,alpha]= [fit.power_law.xmin,fit.power_law.alpha]
        x = np.array(list(set(kwSeq)))
        xmin_inx = np.where(x==xmin)[0][0]
        y = np.power(x[xmin_inx:],-alpha)/np.sum(np.power(x[xmin_inx:],-alpha))
        if sclique == 0:
            ax[0].loglog(x[xmin_inx:],y,color='black',lw=2,label=r'$O$'+r'$,\gamma=-$'+str(round(alpha,1)))
            fit.plot_pdf(ax=ax[0],marker='o',color='white',markersize=ms,mec='black',mew=mew)

        else:
            ax[0].loglog(x[xmin_inx:],y,color=colors(i),lw=2,label=r'$R=$'+str(sclique+1)+r'$,\gamma=-$'+str(round(alpha,1)))
            fit.plot_pdf(ax=ax[0],marker='o',color='white',markersize=ms,mec=colors(i),mew=mew)

    cg.PlotAxes(ax[0],r'$k_w$',r'$P(k_w)$',title='a')
    ax[0].legend(loc='best',fontsize=n_legend,framealpha=0)
    
    
    #axes 2
    Scliques = [2,3,4,5,6,7,8,9]
    Nw={}
    MaxNw = {}
    subax1 = ax[1].inset_axes((0.51,0.51,0.45,0.45))

    for i,sclique in enumerate(Scliques):
        Wnewnodes = cg.load(networkpath +'/'+str(netIndex[sclique])+'_Wnewnodes')
        Nw[sclique] = CountFrequence(Wnewnodes) 
        [keysInx, rankinx] = RankInDict(Wnewnodes)
        MaxNw[sclique] = Wnewnodes[keysInx[rankinx[0]]]
              
    for i,sclique in enumerate(Scliques):
        x = Nw[sclique].keys()
        y = np.array(list(Nw[sclique].values()))
        py = y/np.sum(y)
        ax[1].loglog(x,py,'o',mec=colors(i),color='white',alpha=alp,markersize=ms,mew = mew)
        subax1.plot(sclique+1,MaxNw[sclique],marker='o',mec=colors(i),color='white',markersize=ms,mew = mew)
    
    cg.PlotAxes(ax[1],r'$N_w$',r'$P(N_w)$',title='b') 
    ax[1].legend(loc='best',fontsize=n_legend,framealpha=0)
    PlotAxes(subax1,'k-clique',r'maximum $N_w$',fontsize=16,n_legend=10)
    subax1.set_yscale('log')

    #axes 3
    Ew={}
    MaxEw = {}
    subax2 = ax[2].inset_axes((0.51,0.51,0.45,0.45))
    
    for i,sclique in enumerate(Scliques):
        Wnewedges = cg.load(networkpath +'/'+str(netIndex[sclique])+'_Wnewedges')
        NewWnewedges = {edge:Wnewedges[edge] for edge in Wnewedges.keys() if edge[0]!=edge[1]}
        Ew[sclique] = CountFrequence(NewWnewedges) 
        [keysInx, rankinx] = RankInDict(NewWnewedges)
        MaxEw[sclique] =  Wnewedges[keysInx[rankinx[0]]]   
    
    for i,sclique in enumerate(Scliques):
        x = Ew[sclique].keys()
        y = np.array(list(Ew[sclique].values()))
        py = y/np.sum(y)
        ax[2].loglog(x,py,'o',mec=colors(i),color='white',alpha=alp,markersize=ms,mew = mew)
        subax2.plot(sclique+1,MaxEw[sclique],marker='o',mec=colors(i),color='white',markersize=ms,mew = mew)

    cg.PlotAxes(ax[2],r'$E_w$',r'$P(E_w)$',title='c') 
    PlotAxes(subax2,'k-clique',r'maximum $E_w$',fontsize=16,n_legend=10)
    
    plt.savefig(figurepath+'/FigS4.png', dpi = 600)
    plt.savefig(figurepath+'/FigS4.eps')
    plt.savefig(figurepath+'/FigS4.pdf')


if __name__ == '__main__':

    figurepath  = root_path + '/NetworkReduction/sfig4_structure/figure'
    networkpath = root_path + '/NetworkReduction/fig3_reductionSize/network/GrQc'
    
    #plot reducted network structure
    PlotNetStruc(figurepath,networkpath)
    
