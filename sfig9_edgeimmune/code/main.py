# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:05:20 2022

@author: Leyang Xue

"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk 
from multiprocessing import Pool
import os
import copy 
import random 
import seaborn as sns

def Spreadseed(G,alphas,mu,simulation,path,filename):
    
    args = []
    for itr in np.arange(simulation):
         seeds = random.choice(list(G.nodes()))
         args.append([itr,G,alphas,mu,seeds])
         
    pool = Pool(processes = 8)
    results = pool.map(cg.run_SIR_mseed,args)
    rhos = np.array(results)
    avg_rho = np.average(rhos,axis=0)
    
    pk.dump(rhos,open(path+'/'+str(filename),'wb'))
    np.savetxt(path+'/'+str(filename)+'.csv',avg_rho,delimiter=',')
    
    return avg_rho 
       
def ERspread(networkpath,resultpath,filename,betas,Nredge):
    
    mu = 1
    simulation = 1000
    
    edgelist = cg.load(networkpath+'/0_reducedEdgelist') 
    G = cg.load_network(edgelist[:,0:2])  
    rseed = cg.load(resultpath+'/'+filename)
    
    file = filename.strip('_seed')   
    spread_path = resultpath+'/'+file +'_spread'
    if not os.path.exists(spread_path):
       os.mkdir(spread_path)

    update_G = copy.deepcopy(G)
    for num,edge in enumerate(rseed[:Nredge]):
        print('num/Nredge:',num)
        Spreadseed(update_G,betas,mu,simulation,spread_path,num)
        update_G.remove_edge(edge[0],edge[1])  

def RunERDiss(networkpath,resultpath):
    
    sclique = [2,3,4,5,6,7,8,9]
    betas = np.arange(0,1.01,0.01)
    topNedge = 100
    
    #run the spread
    for clique in sclique:
        print('clique',clique)
        #EdgeRemove(path,clique,topNedge)
        filenames = 'NR'+str(clique)+'_topN_seed'
        ERspread(networkpath,resultpath,filenames,betas,topNedge)
    
def PlotERDiss(networkpath,resultpath,figurepath):
    
    #Plot the fig
    colors = plt.get_cmap('Paired')
    n_legend = 18

    edgelist = cg.load(networkpath+'/0_reducedEdgelist')
    G = cg.load_network(edgelist[:,0:2])
    sclique = [2,3,4,5,6,7,8,9]

    fig,ax = plt.subplots(1,3,figsize=(21,7),constrained_layout=True)
    rhos = pd.DataFrame()
    rhos['beta'] = np.arange(0,1.01,0.01)
    for i,clique in enumerate(sclique):
        filenames = 'NR'+str(clique)+'_topN_spread'
        filenames_lccs = 'NR'+str(clique)+'_topN_lccs'
        spread = cg.load(resultpath+'/'+filenames+'/50')
        lccs = cg.load(resultpath+'/'+filenames_lccs)
        rhos[str(clique)] = np.average(spread,axis=0)/G.order()
        sns.scatterplot(data=rhos,x='beta',y=str(clique),ax=ax[0],marker='o',color=colors(i),label=r'$NR_e('+str(clique+1)+')$',sizes=30)#,,label = file
        x = np.array(list(lccs.keys()))
        y = np.array(list(lccs.values()))
        ax[1].plot(x[:101]*G.size(),y[:101]/G.order(),color = colors(i))
        
    cg.PlotAxes(ax[0],r'$\beta$',r'$\rho$','a')
    cg.PlotAxes(ax[1],'$n$',r'lcc','b')
    ax[0].legend(loc='best',framealpha=0, fontsize=n_legend)
    
    centrality = ['kO','k','BetwO','Betw','NR5']
    label = ['$K(k_1 x k_2)$','A.k','Bet.','A.Bet.',r'$NR_e(5)$']
    for i,each in enumerate(centrality):
        filenames = each + '_lccs'
        lccs = cg.load(resultpath+'/'+filenames)
        
        x = np.array(list(lccs.keys()))[:100]*G.size()
        y = np.array(list(lccs.values()))[:100]/G.order()
        ax[2].plot(x,y,color = colors(2*i+1),label =label[i])
    cg.PlotAxes(ax[2],'$n$',r'lcc','c')
    ax[2].legend(loc='best',framealpha=0, fontsize=n_legend)
    
    plt.savefig(figurepath + '/FigS9.png', dpi=600)
    plt.savefig(figurepath + '/FigS9.eps')
    plt.savefig(figurepath + '/FigS9.pdf')
    
if __name__ == '__main__':    
    
    networkpath = root_path +'/NetworkReduction/fig3_reductionSize/network/GrQc'
    resultpath = root_path + '/NetworkReduction/sfig9_edgeimmune/result'
    figurepath = root_path + '/NetworkReduction/sfig9_edgeimmune/figure'
    
    #ER spread
    RunERDiss(networkpath,resultpath)
    #plot the result
    PlotERDiss(networkpath,resultpath,figurepath)
    