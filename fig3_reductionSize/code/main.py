# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:06:45 2022

@author: Leyang Xue

"""
#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import powerlaw

def coarseGrain(path,filename):
    '''
    run the coarse-graining process for a given network 

    Parameters
    ----------
    path : str
        path to load the network and save the CGNs.
    filename : str
        name of datasets.

    Returns
    -------
    None.

    '''
    #load the original network
    edgelist = np.loadtxt(path+'/'+filename)
    G = cg.load_network(edgelist)
    edgelist = np.array([[edge[0],edge[1]] for edge in nx.to_edgelist(G)])
    Rorder = 3
    [Nnodes, Nedges, sizeCliques]= cg.coarseGrain(edgelist,Rorder,path)
    
def PlotCoarseGrain(root_path,figure_path, file): 
    
    plt.figure(figsize=(18,10),constrained_layout=True)
    ax1 = plt.subplot2grid((8,3), (0,0), colspan=1,rowspan=4)
    ax2 = plt.subplot2grid((8,3), (0,1), colspan=2,rowspan=4)
    ax3 = plt.subplot2grid((8,3), (4,0), colspan=1,rowspan=4)
    ax4 = plt.subplot2grid((8,3), (4,1), colspan=1,rowspan=4)
    ax5 = plt.subplot2grid((8,3), (4,2), colspan=1,rowspan=4)
    
    colors = plt.get_cmap('Paired')
    mec_color = 'black'
    markers = ['o','v','h','^','H','<','s','>']
    markersize = 8
    n_legend = 18
    lw = 1
    mew = 0.8
    alp = 1
    
    for i,each in enumerate(file):
        
        path = root_path+'/'+each  
        
        #Subfig1 data
        OG_edgelist = cg.load(path +'/'+'0_reducedEdgelist')
        OG = cg.load_network(OG_edgelist[:,0:2])
        [CliquesDtrb,CliqueSize] = cg.CliqueDistribution(OG)
    
        #subfig3 data
        sizeCliques = np.loadtxt(path+'/sizeCliques.txt')
        [netIndex, netNweight]= cg.NetworkInx(sizeCliques,path)
        [NewCliqueNum,NewClique]= cg.NumNewClique(path, netIndex)
        #cliqueLabels= cg.ReducedNodeMap(path)
        #BoxNum= CountNumBox(cliqueLabels)
        #BoxNumNorm = {box:BoxNum[box]/sum(BoxNum.values()) for box in BoxNum.keys()}
        
        #subfig4 data
        #boxweightDegree= BoxNodeWeight(OG, path, netIndex)
        Nnode = np.loadtxt(path+'/Nnodes.txt')
        Nedge = np.loadtxt(path+'/Nedges.txt')
        
        #plot the sfig1
        x_values = np.array(sorted(CliquesDtrb.keys()))
        y_values = np.array([CliquesDtrb[each] for each in x_values])  
        #if  i == 0:
            #subax1 = ax1.inset_axes((0.51,0.51,0.45,0.45))
            #subax1.spines['top'].set_visible(False)
            #subax1.spines['right'].set_visible(False)

        #ax1.loglog(x_values,y_values,marker=markers[i],color=colors(2*i+1),markersize = markersize,mec=mec_color,mew=mew,ls='')
        fit = powerlaw.Fit(CliqueSize, discrete = True)
        [xmin,alpha]= [fit.power_law.xmin,fit.power_law.alpha]
        x_inx = np.argwhere(x_values == xmin)[0][0]
        x = x_values[x_inx:]
        y = np.power(x,-alpha)/np.sum(np.power(x,-alpha))
        ax1.loglog(x,y,ls='-',lw=1.5,color=colors(2*i+1), label=r'$\alpha$=-'+str(round(alpha,1)))
        fit.plot_pdf(ax=ax1,marker=markers[i],color=colors(2*i+1),markersize=markersize,mec=mec_color,mew=mew, ls='')

        #plot the sfig2
        ax2.plot(range(1,len(sizeCliques)+1),sizeCliques,color=colors(2*i+1),label = each)

        #plot the sfig3
        #if  i == 0:
            #subax3 = ax3.inset_axes((0.51,0.51,0.45,0.45))
        
        fit = powerlaw.Fit(NewClique, discrete = True)
        [xmin,alpha]= [fit.power_law.xmin,fit.power_law.alpha]
        
        x_values = np.array(list(NewCliqueNum.keys()))
        y_values = np.array(list(NewCliqueNum.values()))  
        x_inx = np.argwhere(x_values == xmin)[0][0]
        y_inx = np.argwhere(y_values != 0)[0][0]
        x = x_values[y_inx:int(x_inx+1)]
        y = np.power(x,-alpha)/np.sum(np.power(x,-alpha))
        ax3.loglog(x,y,ls='-',lw=1.5,color=colors(2*i+1), label=r'$\alpha$=-'+str(round(alpha,1)))
        fit.plot_pdf(ax=ax3, marker=markers[i],color=colors(2*i+1), mec=mec_color,markersize = markersize,mew=mew,ls='')

        #ax3.loglog(x_values,y_values/sum(y_values),markers[i],color=colors(2*i+1), markersize = markersize,mec=mec_color,linestyle='',lw=lw,mew=mew)
        
        #sfig4
        netinx= np.array(list(netIndex.values()))
        
        ax4.plot(netIndex.keys(),Nnode[netinx]/Nnode[0],markers[i],color=colors(2*i+1),markersize = markersize,mec=mec_color,linestyle='solid',alpha=alp,lw=lw,mew=mew,label=each)
        ax5.plot(netIndex.keys(),Nedge[netinx]/Nedge[0],markers[i],color=colors(2*i+1),markersize = markersize,mec=mec_color,linestyle='solid',alpha=alp,lw=lw,mew=mew)
        
        if i==3:
          cg.PlotAxes(ax1,'k-clique',r'$P(k)$','a')
          ax1.legend(loc='upper right',framealpha=0, fontsize=n_legend)
          #PlotAxes(subax1,'k-clique','$P(k)$',fontsize=16,n_legend=10)

          cg.PlotAxes(ax2,'reduction step','maximum k-clique','b')
          ax2.legend(loc='best',framealpha=0, fontsize=n_legend)
          ax2.set_yscale('log')
          ax2.set_xscale('log')
          
          cg.PlotAxes(ax3,'k-clique',r'$P(\Delta k)$','c')
          ax3.legend(loc='best',framealpha=0, fontsize=n_legend)
          #PlotAxes(subax3,'k-clique',r'$P(\Delta k)$',fontsize=16,n_legend=10)

          cg.PlotAxes(ax4,'k-clique',r'$\frac{N_r}{N_o}$','d')
          ax4.set_xscale('log')
          ax4.legend(loc='best',framealpha=0, fontsize=n_legend)

          cg.PlotAxes(ax5,r'k-clique',r'$\frac{E_r}{E_o}$','e')
          ax5.set_xscale('log')

    plt.savefig(figure_path+'/fig3.png',dpi=600)
    plt.savefig(figure_path+'/fig3.eps')
    plt.savefig(figure_path+'/fig3.pdf')

if __name__ == '__main__':
    
  root_path  = 'F:/work/work5_reductionability'
  networkpath = root_path + '/NetworkReduction/fig3_reductionSize/network'  
  resultpath = root_path + '/NetworkReduction/fig3_reductionSize/result'
  figurepath = root_path + '/NetworkReduction/fig3_reductionSize/figure'
  
  #1 calculate the basic structure properities of networks
  cg.netStatistics(networkpath,resultpath)
  
  #2 run the coarse-graining process for fours networks
  files = ['GrQC','CondMat','HepPh','NetScience']#'Email-Enron','Musae_facebook','CondMat','HepPh','NetScience'
  filenames = ['CA-GrQc.txt','CA-CondMat.txt','CA-HepPh.txt','NetSci2019_edgelist.csv']
  for file,filename in zip(files,filenames):
      #run the coarse-graining for each datasets
      eachNetpath = networkpath + '/' + file
      coarseGrain(eachNetpath,filename)
      
  #3 Plot Reducction 
  PlotCoarseGrain(networkpath,figurepath,files)  

  