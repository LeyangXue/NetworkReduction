# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:36:14 2022

@author: Leyang Xue
"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np

def PlotCGDatasets(files,resultpath,figurepath):
    
    fig,ax = plt.subplots(1,3,figsize=(21,7),constrained_layout=True)
    
    colors = plt.get_cmap('Paired')
    #mec_color = 'black'
    #markers = ['o','v','^','<','>','8','s','p','P','*','h','+']
    #markersize = 10
    n_legend = 18
    lw = 2
    #mew = 0.8
    #alp = 1
    
    #clique_exponent = {}
    #new_clique_exponent = {}
    for i,each in enumerate(files):
        print('datasets:',each)
        
        L = np.loadtxt(resultpath+'/'+each+'_RD.csv', delimiter = ',')
        ax[0].plot(L[:,0],L[:,1],color=colors(i),linestyle='solid',lw=lw,label=each)
        ax[1].plot(L[:,0],L[:,2],color=colors(i),linestyle='solid',lw=lw)
        ax[2].plot(L[:,1],L[:,2],color=colors(i),linestyle='solid',lw=lw)
    
    cg.PlotAxes(ax[0],'k-clique',r'$\frac{N_r}{N_o}$','a')
    cg.PlotAxes(ax[1],'k-clique',r'$\frac{E_r}{E_o}$','b')
    cg.PlotAxes(ax[2],r'$\frac{N_r}{N_o}$',r'$\frac{E_r}{E_o}$','c')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].legend(loc='best',framealpha=0, fontsize=n_legend)
    
    plt.savefig(figurepath+'/sfig2.png', dpi=600)
    plt.savefig(figurepath+'/sfig2.pdf')
    plt.savefig(figurepath+'/sfig2.eps')

    
if __name__ == '__main__':
    
  networkpath = root_path + '/NetworkReduction/sfig2_multidatasets/network'  
  resultpath = root_path + '/NetworkReduction/sfig2_multidatasets/result'
  figurepath = root_path + '/NetworkReduction/sfig2_multidatasets/figure'

  files = ['AstroPh','CondMat','GrQc','HepPh','NetScience','DBLP','Deezer','Email-Enron','Musae_DE','Musae_facebook',
          'Musae_git','PhoneCalls'] 
  
  #plot the figure of network reduction for different datasets
  PlotCGDatasets(files,resultpath,figurepath)