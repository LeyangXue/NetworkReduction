# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:13:47 2022

@author: Leyang Xue

"""

#please change the current path if run the code
root_path  = 'G:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np

def LoadSpreadProbBeta(networkpath,resultpath,files,inx):
    
    data = {}
    for i,each in enumerate(files):
        
        netpath = networkpath + '/' + each
        resulpath = resultpath + '/' + each
        
        Nnode = np.loadtxt(netpath+'/Nnodes.txt')
        Nedge = np.loadtxt(netpath+'/Nedges.txt')
        sizeCliques = np.loadtxt(netpath+'/sizeCliques.txt')
        [netIndex,Nweight] = cg.NetworkInx(sizeCliques,netpath)
        
        spreadPath = resulpath + '/SpreadProbBeta'
        
        beta20_node ={}
        beta20_edge ={}
        beta5_node ={}
        beta5_edge ={}
    
        point20_node = {}
        point20_edge = {}
        point5_node = {}
        point5_edge = {}
    
        for scq in netIndex.keys():
            vs = cg.load(spreadPath+'/'+str(float(scq))+'_BetaT_vectorD')[inx]
            index = int(netIndex[scq])
        
            beta20_node[vs[0]] = Nnode[index]/Nnode[0]
            beta20_edge[vs[0]] = Nedge[index]/Nedge[0]
            beta5_node[vs[1]] =  Nnode[index]/Nnode[0]
            beta5_edge[vs[1]] =  Nedge[index]/Nedge[0]
            
            if int(scq)==19:
                point20_node[vs[0]] =  Nnode[index]/Nnode[0]
                point20_edge[vs[0]] =  Nedge[index]/Nedge[0]
            if int(scq) == 4:
                point5_node[vs[1]] =  Nnode[index]/Nnode[0]
                point5_edge[vs[1]] =  Nedge[index]/Nedge[0]
        
        data[each]=[beta20_node,beta20_edge,beta5_node,beta5_edge,point20_node,point20_edge,point5_node,point5_edge]    
    
    return data

def PlotApproAccuracy(data1,data2,figurepath):
    
    markersize = 12
    mew = 0.6
    n_legend = 18
       
    colors = plt.get_cmap('Paired')
    fig,ax = plt.subplots(2,2,figsize=(12,12),constrained_layout=True,sharey=True)
    ax[0,0].sharex(ax[1,0])
    ax[0,1].sharex(ax[1,1])
    
    for i,data_name in enumerate(data1.keys()):
        
        [beta20_node,beta20_edge,beta5_node,beta5_edge,point20_node,point20_edge,point5_node,point5_edge] = data1[data_name]
        [beta20_node2,beta20_edge2,beta5_node2,beta5_edge2,point20_node2,point20_edge2,point5_node2,point5_edge2] = data2[data_name]

        ax[0,0].plot(beta20_node.keys(),beta20_node.values(),color=colors(2*i+1),linestyle='dashed')
        ax[0,0].plot(point20_node.keys(),point20_node.values(),marker='s',mec = 'black',color=colors(2*i+1), ms = markersize,mew=mew)
        ax[0,0].plot(beta20_edge.keys(),beta20_edge.values(),color=colors(2*i+1),linestyle='solid')
        ax[0,0].plot(point20_edge.keys(),point20_edge.values(),marker='o',mec ='black',color=colors(2*i+1), ms = markersize,mew=mew)
        
        ax[1,0].plot(beta20_node2.keys(),beta20_node2.values(),color=colors(2*i+1),linestyle='dashed')
        ax[1,0].plot(point20_node2.keys(),point20_node2.values(),marker='s',mec = 'black',color=colors(2*i+1), ms = markersize,mew=mew)
        ax[1,0].plot(beta20_edge2.keys(),beta20_edge2.values(),color=colors(2*i+1),linestyle='solid')
        ax[1,0].plot(point20_edge2.keys(),point20_edge2.values(),marker='o',mec ='black',color=colors(2*i+1), ms = markersize,mew=mew)
        
        ax[0,1].plot(beta5_node.keys(),beta5_node.values(),color=colors(2*i+1),linestyle='dashed')
        ax[0,1].plot(point5_node.keys(),point5_node.values(),marker='s',mec = 'black',color=colors(2*i+1), ms = markersize,mew=mew)
        ax[0,1].plot(beta5_edge.keys(),beta5_edge.values(),color=colors(2*i+1),linestyle='solid', label=data_name)
        ax[0,1].plot(point5_edge.keys(),point5_edge.values(),marker='o',mec ='black',color=colors(2*i+1), ms = markersize,mew=mew)
        
        ax[1,1].plot(beta5_node2.keys(),beta5_node2.values(),color=colors(2*i+1),linestyle='dashed')
        ax[1,1].plot(point5_node2.keys(),point5_node2.values(),marker='s',mec = 'black',color=colors(2*i+1), ms = markersize,mew=mew)
        ax[1,1].plot(beta5_edge2.keys(),beta5_edge2.values(),color=colors(2*i+1),linestyle='solid', label=data_name)
        ax[1,1].plot(point5_edge2.keys(),point5_edge2.values(),marker='o',mec ='black',color=colors(2*i+1), ms = markersize,mew=mew)


    ax[0,0].set_xscale("log")
    ax[0,1].set_xscale("log")
    ax[1,0].set_xscale("log")
    ax[1,1].set_xscale("log")

    #ax[0,0].set_xticklabels([0.02,'',0.1,'',''])
    ax[0,0].set_xticks([0.01,0.1])
    ax[1,0].set_xticks([0.01,0.1])
    ax[0,1].set_xticks([0.01,0.1])
    ax[1,1].set_xticks([0.01,0.1])
      
    cg.PlotAxes(ax[0,0],'',r'$\frac{E_r}{E_o}$,  $\frac{N_r}{N_o}$','a')    
    cg.PlotAxes(ax[0,1],'','','b')
    cg.PlotAxes(ax[1,0],r'$D_{prob}$',r'$\frac{E_r}{E_o}$,  $\frac{N_r}{N_o}$','c')
    cg.PlotAxes(ax[1,1],r'$D_{prob}$','','d')
    
    ax[0,1].legend(loc='lower right',framealpha=0, fontsize=n_legend)  
    
    ax[0,0].text(0.15,0.85,r'$\beta=$'+str(0.36),color='#949396',size=20)
    ax[1,0].text(0.15,0.85,r'$\beta=$'+str(0.36),color='#949396',size=20)
    ax[0,1].text(0.04,0.85,r'$\beta=$'+str(0.83),color='#949396',size=20)
    ax[1,1].text(0.04,0.85,r'$\beta=$'+str(0.83),color='#949396',size=20)

    #xn = list(point_node.keys())[0]
    #ax.axvline(xn,color='gray',linestyle='dashed')
    #ax.text(xn+0.002,0.2,r'$D_{prob}^{\beta^t=\beta}$',color='gray',size=15)
    #PlotAxes(axes2,'',r'$\frac{E_r}{E_o}$(edge)')
    
    plt.savefig(figurepath + '/FigS6.png', dpi = 600)
    plt.savefig(figurepath + '/FigS6.pdf')
    plt.savefig(figurepath + '/FigS6.eps')
    

if __name__ == '__main__':

    networkpath = root_path + '/NetworkReduction/fig3_reductionSize/network'
    resultpath = root_path + '/NetworkReduction/fig4_reductionAccuracy/result'
    figurepath = root_path + '/NetworkReduction/sfig6_accuvsreductsize/figure'
    files = ['GrQC','CondMat','HepPh','NetScience']
    
    #Plot the approximate accuracy
    data1 = LoadSpreadProbBeta(networkpath,resultpath,files,2) 
    data2 = LoadSpreadProbBeta(networkpath,resultpath,files,0) 
    
    PlotApproAccuracy(data1, data2, figurepath)