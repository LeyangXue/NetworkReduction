# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:21:14 2022

@author: Leyang XUe

"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np
import itertools
import networkx as nx
import pickle as pk 
from multiprocessing import Pool

def Save(data,path,filename):
    
    pk.dump(data,open(path+'/'+filename,'wb'))


def Spreadmseed(G,alphas,mu,seeds,simulation,path,filename):
    
    args = []
    for itr in np.arange(simulation):
         args.append([itr,G,alphas,mu,seeds])
         
    pool = Pool(processes = 8)
    results = pool.map(cg.run_SIR_mseed,args)
    rhos = np.array(results)
    avg_rho = np.average(rhos,axis=0)
    
    Save(rhos,path,filename)
    np.savetxt(path+'/'+filename+'.csv',avg_rho,delimiter=',')
    
    return avg_rho 
 
def LocatetSubGNSeed(G,ReducedNodes,topn,n):
    
    block = {}
    for each in topn:
        block[each]=cg.TrackBackNodes([each],ReducedNodes) 
    
    NodeBlock_n={}
    for i,newnodes in enumerate(block.keys()):
        block_nodes = block[newnodes]
        blockG = nx.subgraph(G, block_nodes)
        degree = dict(nx.degree(blockG))
        NodeBlock_n[i] = identifyTopN(degree,n)
        
    return NodeBlock_n

def identifyTopN(centrality,N):
    
    nodeweight= np.array([[node,centrality[node]] for node in centrality.keys()])
    maxindex = np.argsort(-nodeweight[:,1])[0:N]
    maxNode = nodeweight[maxindex,0]  
    
    return maxNode


def SelectTopNNode(networkpath,spreadpath):
    
    edgelist = cg.load(networkpath+'/0_reducedEdgelist')
    G = cg.load_network(edgelist[:,0:2])
    
    #identify top n node in a block node 
    clique = 4
    sizeCliques = np.loadtxt(networkpath+'/sizeCliques.txt')
    [netIndex, netNweight]= cg.NetworkInx(sizeCliques,networkpath)
    NR5 = cg.load(networkpath+'/'+str(netIndex[clique])+'_Wnewnodes')
    ReducedNodes = cg.ReducedNodeMap(networkpath)
    
    topn = identifyTopN(NR5,2)
    n = 10
    NodeBlock_Seed = LocatetSubGNSeed(G,ReducedNodes,topn,n)
    pk.dump(NodeBlock_Seed,open(spreadpath+'/Block_Mseed','wb'))

    #spread
    block_spread = {}
    simulation = 1000
    alphas = np.arange(0,1.01,0.01)
    mu = 1
    for i,block in enumerate(NodeBlock_Seed.keys()):
        block_seed = NodeBlock_Seed[block]
        seed = []
        for j,each in enumerate(block_seed):
            file = 'block'+str(i)+'_n'+ str(j)
            print(file)
            seed.append(each)
            block_spread[file] = Spreadmseed(G,alphas,mu,seed,simulation,spreadpath,file)   

    pk.dump(block_spread,open(spreadpath+'/block_spread','wb'))

def PlotAxes(ax,xlabel,ylabel,title='',fontsize=20, n_legend = 18, mode=True):
    
    font_label = {'family': "Calibri", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=fontsize)
    if mode == True:
        ax.legend(loc='best',framealpha=0, fontsize=n_legend)
        
def PlotNR5BetaC(ax,resultpath,networkpath):
    '''
    plot the performance of NR based on different k-clique CGNs under \betac, 5\betac, 10\betac infection rate

    Parameters
    ----------
    ax : axes
        axes.
    resultpath : str
        path to load the result.
    networkpath : str
        path to load the network.

    Returns
    -------
    None.

    '''
    nrhos = cg.load(resultpath+'/NR5_nrhos')

    Scliques = [2,3,4,5,6,7,8,9]
    colors = plt.get_cmap('Paired')
    mec_color =  'white'
    mew = 1.2
    n_legend = 18
    alp = 1
    
    #subax1
    edgelist = cg.load(networkpath+'/0_reducedEdgelist')
    G = cg.load_network(edgelist[:,0:2])
    betac = cg.CriticalInfectRate(G)
    betac_inx = list(map(lambda x: int(round(x,2)*100),[betac,5*betac,10*betac]))
    
    width = 0.85
    x = np.arange(1,6,2)# the label locations
    clique_rhos= nrhos[10]
    
    for i,inx in enumerate(betac_inx):
        for j,clique in enumerate(Scliques):
            d = -width+(2*width)/len(Scliques)*j
            if i == 0:
                ax.bar(x[i]+d,clique_rhos[clique][inx]/G.order(),(2*width)/len(Scliques),color=colors(j),edgecolor = mec_color, linewidth=mew, label=r'$NR_n($'+str(clique+1)+')', alpha=alp)
            else:
                ax.bar(x[i]+d,clique_rhos[clique][inx]/G.order(),(2*width)/len(Scliques),color=colors(j),edgecolor = mec_color, linewidth=mew,alpha=alp)

    cg.PlotAxes(ax, r'$\beta$',r'$\rho$','a')
    ax.set_xticks(x)
    ax.set_xticklabels([r'$\beta_c$',r'$5\beta_c$',r'$10\beta_c$'],fontdict = {'family': "Arial", 'size':24})
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)

def PlotBlockSpread(ax,spreadpath):

    block_spread = cg.load(spreadpath+'/block_spread')
    betas = np.arange(0,1.01,0.01)
    
    colors = plt.get_cmap('tab20c')
    markers = ['o','^','s','*']
    mec_color = 'black'#'#978C86'#'white'
    n_legend = 18
    markersize = 10
    alp = 1
    mew = 0.8
    x = np.arange(0,101,5)
    for i in np.arange(0,2):
        for j in np.arange(0,4):
            file = 'block'+str(i)+'_n'+ str(j)
            rhos = block_spread[file]
            if i == 0 :
                ax.loglog(betas[x][1:],rhos[x][1:]/rhos[-1],marker = markers[j], ms=markersize, color=colors(j),alpha=alp, mec=mec_color,mew=mew,label ='Top-1, $N_s=$'+str(j+1), ls='solid')
            else:
                ax.loglog(betas[x][1:],rhos[x][1:]/rhos[-1],marker = markers[j], ms=markersize, color=colors(j+4),alpha=alp, mec=mec_color,mew=mew,label ='Top-2, $N_s=$'+str(j+1), ls='solid')
    
    cg.PlotAxes(ax,r'$\beta$', r'$\rho$','b')
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)
 
def PathLength(G,seed):
    
    path_length = 0
    pairs = list(itertools.combinations(seed, 2))
    for s in pairs:
        path_length += len(nx.shortest_path(G, s[0],s[1]))
    
    average_path = path_length/len(pairs)
    
    return average_path

def PlotPath(ax,resultpath,networkpath):
    
    npath = cg.load(resultpath+'/NR5_npath')
    edgelist = cg.load(networkpath+'/0_reducedEdgelist')
    G = cg.load_network(edgelist[:,0:2])
    
    DiffMethod = cg.load(resultpath+'/N_cseeds')
    method_seed = DiffMethod[10]
    diffMethod_path = {}
    for centra in method_seed.keys():
        seed = method_seed[centra]
        avg_path = PathLength(G,seed)
        diffMethod_path[centra] = avg_path
    
    colors = plt.get_cmap('Paired')
    x = np.arange(len(diffMethod_path))# the label locations
    width = 0.6
    mec_color = 'white'
    mew = 1.2
    
    tick_labels = ['NR5', 'k', 'KS', 'Betwn.', 'Closn.', 'Eigen.', 'Katz', 'Subgh.', 'CI', 'NB']
    for i,name in enumerate(diffMethod_path.keys()):
        ax.bar(x[i],diffMethod_path[name],width,color=colors(i),edgecolor = mec_color, linewidth=mew)
        
    cg.PlotAxes(ax,'centralities', r'$\left\langle  d \right\rangle$','c')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels,fontdict = {'family': "Arial", 'size':24}, rotation=55)
    
    axinx = ax.inset_axes((0.60,0.62,0.35,0.35))
    axinx.plot(np.array(sorted(npath[10].keys()))+1, npath[10].values(),marker = 'o', ms=8, color=colors(0), mec='black',mew=0.8, ls='solid')
    PlotAxes(axinx,'Reducted Order',r'$\left\langle  d \right\rangle$','',fontsize=16, n_legend = 10)

def PlotIMDiss(resultpath,networkpath,figurepath,spreadpath):
        
    fig,ax = plt.subplots(1,3,figsize=(21,7),constrained_layout=True)
    PlotNR5BetaC(ax[0],resultpath,networkpath)
    PlotBlockSpread(ax[1],spreadpath)
    PlotPath(ax[2],resultpath,networkpath)
    
    plt.savefig(figurepath+'/FigS8.png', dpi=600)
    plt.savefig(figurepath+'/FigS8.pdf')
    plt.savefig(figurepath+'/FigS8.eps')


if __name__ == '__main__':

    networkpath = root_path + '/NetworkReduction/fig3_reductionSize/network/GrQc'
    resultpath  = root_path + '/NetworkReduction/fig5_application/result/GrQC/IM'
    figurepath = root_path + '/NetworkReduction/sfig8_influencemaxi/figure'
    spreadpath = root_path + '/NetworkReduction/sfig8_influencemaxi/result' 
    
    #select top-5 nodes in lagerest node-weight for NR(5)
    SelectTopNNode(networkpath,spreadpath)
    PlotIMDiss(resultpath,networkpath,figurepath,spreadpath)
    