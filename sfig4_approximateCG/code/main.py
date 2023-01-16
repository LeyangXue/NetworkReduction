# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:54:27 2022

@author: Leyang Xue

"""
#please change the current path if run the code
root_path  = 'G:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def coarseGrainApproxiate(path,filename):
    '''
    approximately reduction by using the different k-plex structure

    Parameters
    ----------
    path : str
        path to load the network.
    filename : str
        file name.

    Returns
    -------
    None.

    '''
    edgelist = np.loadtxt(path+'/'+filename)
    G = cg.load_network(edgelist)
    edgelist = np.array([[edge[0],edge[1]] for edge in nx.to_edgelist(G)])
    p = 0.5
    m = 3
    [Nnodes, Nedges, sizeCliques,sizeKplex,K] = cg.coarseGrain_approxiate(edgelist,p,m,path) 

def loadARData(path):
    '''
    load the data of approximate reduction
    
    '''
    sizeCliques = np.loadtxt(path+'/sizeCliques.txt') 
    sizeKplex = np.loadtxt(path+'/sizeKplex.txt')
    K = np.loadtxt(path+'/K.txt')
    Nedges = np.loadtxt(path+r'/Nedges.txt')
    Nnodes = np.loadtxt(path+r'/Nnodes.txt')
    
    return sizeCliques,sizeKplex,K,Nedges,Nnodes

def loadData(path_clique):
    
    Nedges = np.loadtxt(path_clique+'/Nedges.txt')
    Nnodes = np.loadtxt(path_clique+'/Nnodes.txt')
    sizeCliques=np.loadtxt(path_clique+'/sizeCliques.txt')
    
    return Nedges,Nnodes,sizeCliques

def PlotAxes(ax,xlabel,ylabel,title='',fontsize=20, n_legend = 18, mode=True):
    
    font_label = {'family': "Calibri", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=fontsize)
    if mode == True:
        ax.legend(loc='best',framealpha=0, fontsize=n_legend)
        
def PlotApproximateReduction(approcgpath,nonapprocgpath,figurepath):
    
    #load the appproxiate coarse-grain
    [AsizeCliques,AsizeKplex,AK,ANedges,ANnodes] = loadARData(approcgpath)
    [AnetIndex, AnetNweight]= cg.NetworkInx(AsizeCliques,approcgpath)

    [Nedges,Nnodes,sizeCliques] = loadData(nonapprocgpath)
    [netIndex, netNweight]= cg.NetworkInx(sizeCliques,nonapprocgpath)

    markers=['s','o']
    colors = plt.get_cmap('Paired')
    mec_color = 'black'#978C86'#'white'
    markersize = 10
    n_legend = 18
    alp = 1    
    lw = 2

    Anetinx= np.array(list(AnetIndex.values()))
    netinx= np.array(list(netIndex.values()))
    y = list(map(lambda x:int(x),sorted(set(AK),reverse=True)))
    x = [1.2,2.7,4.7,6.7,8.7,19,99]
    fig,ax = plt.subplots(1,3,figsize=(21,7),tight_layout=True)
    ax[0].plot(np.arange(1,len(AsizeKplex)+1),AsizeKplex,color=colors(3),label='k-plex',lw=lw)
    ax[0].plot(np.arange(1,len(AsizeCliques)+1),AsizeCliques,color=colors(1),label='k-clique',lw=lw)
    ax[0].bar(np.arange(1,len(AsizeCliques)+1),AK,width=1,color=colors(6),label='K',align='edge')
    for x_index,y_index in zip(x,y):
        ax[0].text(x_index,y_index+0.5,str(y_index),size=18,color='gray')
    ax[0].set_xscale('log')
    cg.PlotAxes(ax[0],'reduction step','size','a')
    ax[0].legend(loc='best',framealpha=0, fontsize=n_legend)
    
    axinx = ax[0].inset_axes((0.60,0.38,0.35,0.35))
    axinx.plot(np.arange(1,len(sizeCliques)+1),sizeCliques,color=colors(3),label='CG',lw=lw)
    axinx.plot(np.arange(1,len(AsizeCliques)+1),AsizeCliques,color=colors(1),label='Appro. CG',lw=lw)
    PlotAxes(axinx,'reduction step',r'maximum k-clique',title='', fontsize=16, n_legend = 10)
    axinx.set_xscale('log')
    
    ax[1].plot(AnetIndex.keys(),ANnodes[Anetinx]/ANnodes[0], marker = markers[0], color=colors(1),markersize = markersize,label='Appro. CG',mec=mec_color,alpha=alp,ls='solid')
    ax[1].plot(netIndex.keys(),Nnodes[netinx]/Nnodes[0], marker = markers[1], color=colors(3),markersize = markersize,label='CG',mec=mec_color,alpha=alp,ls='solid')
    cg.PlotAxes(ax[1],'k-clique',r'$\frac{N_r}{N_o}$','b')
    ax[1].legend(loc='best',framealpha=0, fontsize=n_legend)

    ax[2].plot(AnetIndex.keys(),ANedges[Anetinx]/ANedges[0], marker = markers[0], color=colors(1),markersize = markersize,mec=mec_color,alpha=alp,ls='solid')
    ax[2].plot(netIndex.keys(),Nedges[netinx]/Nedges[0], marker = markers[1], color=colors(3),markersize = markersize,mec=mec_color,alpha=alp,ls='solid')
    cg.PlotAxes(ax[2],'k-clique',r'$\frac{E_r}{E_o}$','c')
    
    plt.savefig(figurepath+'/FigS4.png', dpi=600)
    plt.savefig(figurepath+'/FigS4.pdf')
    plt.savefig(figurepath+'/FigS4.eps')

if __name__ == '__main__':
    
    #set the path 
    approcgpath = root_path + '/NetworkReduction/sfig4_approximateCG/network'
    nonapprocgpath = root_path + '/NetworkReduction/fig3_reductionSize/network/GrQc'
    figurepath = root_path + '/NetworkReduction/sfig4_approximateCG/figure'
    filename = 'CA-GrQc.txt'
    
    #approximate reduction 
    #coarseGrainApproxiate(approcgpath,filename)
    #plot the result about the approximate reduction
    PlotApproximateReduction(approcgpath,nonapprocgpath,figurepath)
        