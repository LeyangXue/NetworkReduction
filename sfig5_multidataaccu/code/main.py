# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:48:29 2022

@author: Leyang Xue

"""

#please change the current path if run the code
root_path  = 'F:/work/work5_reductionability'

import sys
sys.path.append(root_path+'/NetworkReduction')
from utils import coarse_grain as cg
import matplotlib.pyplot as plt
import numpy as np

def PlotAxes(ax,xlabel,ylabel,title='',fontsize=20, n_legend = 18, mode=True):
    
    font_label = {'family': "Calibri", 'size':fontsize}
    ax.set_xlabel(xlabel,  fontdict = font_label)
    ax.set_ylabel(ylabel, fontdict = font_label)
    ax.set_title(title, loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=fontsize)
    if mode == True:
        ax.legend(loc='best',framealpha=0, fontsize=n_legend)
        
def PlotPcDataSet(ax,path):
    
    mec_color = 'white'#'#978C86'#'white'
    n_legend = 18
    mew = 1.2
    colors = plt.get_cmap('Paired') 
    #alphas = np.arange(0,1.01,0.01)
    RWpc = {}
    RWcv = {}
    
    cliques = [4,5,6,7,8,9]
    spread = np.zeros((0,101))
    files = ['GrQc','CondMat','HepPh','NetScience']
    x = np.arange(len(files))  # the label locations
    width = 0.35  # the width of the bars
    
    for i,file in enumerate(files):
        
        for j, clique in enumerate(cliques):
            file_path = path + '/' + file +'/simulation/'
            Ospread= cg.load(file_path + str(clique) + '_Onetmc')
            Rwspread = cg.load(file_path + str(clique) + '_RWGnetmc')
            spread = np.row_stack((spread,Ospread))
            
            [RWcv[clique],RWpc[clique]] = cg.IdentifyCriInfectRate(Rwspread.T)
            if i == 0:
                ax.bar(x[i]-width+0.1*j,RWpc[clique],0.1,color=colors(j),edgecolor = mec_color, linewidth=mew, label='R='+str(clique+1))
            else:
                ax.bar(x[i]-width+0.1*j,RWpc[clique],0.1,color=colors(j),edgecolor = mec_color,linewidth=mew)
        
        [Ocv,Opc]= cg.IdentifyCriInfectRate(spread.T)
        if i == 0:
            ax.bar(x[i]+0.25,Opc,0.1,color='#D1D1D1',edgecolor = mec_color,linewidth=mew,label='O')
        else:
            ax.bar(x[i]+0.25,Opc,0.1,color='#D1D1D1',edgecolor = mec_color,linewidth=mew)

    #fontsize=20
    #ax.set_xscale('log')
    n_legend = 18
    ax.legend(loc='upper right',framealpha=0, fontsize=n_legend)
    ax.set_xticks(x)
    ax.set_xticklabels(files)
    cg.PlotAxes(ax,'Datasets',r'$\beta_c$','c')
    
def PlotInfProbDataSet(ax,path):
    
    mec_color = 'black'#'#978C86'#'white'
    n_legend = 18
    markersize = 12
    alp = 1
    mew = 0.6
    colors = plt.get_cmap('Paired') 
    
    cliques = [4,5,6,7,8,9]
    alphas = np.arange(0,1.01,0.05)
    sclique = cliques[0]
    axinx = ax.inset_axes((0.55,0.58,0.35,0.35))

    #transform the datasets
    files = ['GrQc','CondMat','HepPh','NetScience']
    # for i,file in enumerate(files[3:]):                
    #     file_path = path + '/' + file
    #     LoadProbValue(file_path,cliques,alphas)
    
    #load the datasets
    for i,file in enumerate(files):                
        file_path = path + '/' + file
        Avg_Dprob_weight = cg.load(file_path+'/Inf_Prob/Avg_Dprob_weight')
        Std_Dprob_weight = cg.load(file_path+'/Inf_Prob/Std_Dprob_weight') 
    
        Avg_Pearson_weight = cg.load(file_path+'/Inf_Prob/Avg_Pearson_weight')
        Std_Pearson_weight = cg.load(file_path+'/Inf_Prob/Std_Pearson_weight') 
        
        ax.errorbar(alphas,Avg_Dprob_weight[sclique],yerr = Std_Dprob_weight[sclique],fmt='o-',capthick=2,capsize=5,ms= markersize,mec=mec_color,color=colors(2*i+1),alpha=alp,mew=mew)
        axinx.errorbar(alphas[0:-1],Avg_Pearson_weight[sclique][0:-1],yerr = Std_Pearson_weight[sclique][0:-1],fmt ='o-',capthick=2,capsize=5,ms=7,mec=mec_color,color=colors(2*i+1),alpha=alp,mew=mew)

    axinx.set_ylim(-0.05,1.05)
    axinx.set_xlim(-0.05,1.05)

    PlotAxes(axinx,r'$\beta$',r'$r$','',fontsize=16, n_legend = 10)
    cg.PlotAxes(ax,r'$\beta$',r'$D_{Prob}$','b')#r'|$P_R^{inf}-P_O^{inf}|$'
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)
    
def PlotSpreadDataSet(ax,path):
    '''
    plot the accuracy of reduction for diffferent k-clique CGNs 
    '''
    markers = ['o','^','s','*','P','H']
    mec_color = 'black'#'#978C86'#'white'
    n_legend = 18
    markersize = 12
    alp = 1
    mew = 0.6
    colors = plt.get_cmap('Paired') 
    
    cliques = [4,5,6,7,8,9]
    betac = np.array([83,76,71,66,61,58])
    files = ['GrQc','CondMat','HepPh','NetScience']
    betas = np.arange(0,1.01,0.01)
    x = np.arange(0,101,10)
    
    axinx = ax.inset_axes((0.60,0.1,0.35,0.35))
    for i,file in enumerate(files):
        
        file_path = path + '/' + file +'/simulation/'
        Omcs = cg.load(file_path + 'Omcs')
        Rwmcs = cg.load(file_path + 'Rwmcs')
            
        RW_avg = Rwmcs/Rwmcs[-1,-1]
        O_avg = np.average(Omcs,axis=1)/Omcs[-1,-1]
        
        for j, sclique in enumerate(cliques):
            if j == 0:
                if file == 'GrQc':
                    ax.plot(O_avg[betac[j]],RW_avg[betac[j],j], marker=markers[0],ms=markersize, color=colors(2*i+1),alpha=alp, mec=mec_color,mew=mew,label ='GrQC', ls='')
                else:
                    ax.plot(O_avg[betac[j]],RW_avg[betac[j],j], marker=markers[0],ms=markersize, color=colors(2*i+1),alpha=alp, mec=mec_color,mew=mew,label =file, ls='')
            else:
                ax.plot(O_avg[betac[j]],RW_avg[betac[j],j], marker=markers[0],ms=markersize, color=colors(2*i+1),alpha=alp, mec=mec_color,mew=mew,ls='')
        
        axinx.plot(betas[x],RW_avg[x,0],marker=markers[0],ms=8, color=colors(2*i+1),alpha=alp, mec=mec_color,mew=mew,ls='')  
        axinx.plot(betas[x],O_avg[x],color=colors(2*i+1),ls='solid',lw=1)
    
    line = np.arange(0.58,1.0,0.01)
    ax.plot(line,line,color='black',alpha=alp,ls='solid')
    cg.PlotAxes(ax,r'$\rho^O$',r'$\rho^R$','a')
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)

    PlotAxes(axinx,r'$\beta$',r'$\rho^R$','',fontsize=16, n_legend = 10)
    axinx.tick_params(direction='in', which='both',length =3, width=1, labelsize=14)

def PlotAccuracyDataSet(figurepath,resultpath):
    
    fig,ax = plt.subplots(1,3,figsize=(21,7),constrained_layout=True)
    
    #subfig1
    PlotSpreadDataSet(ax[0],resultpath)
    PlotInfProbDataSet(ax[1],resultpath)
    PlotPcDataSet(ax[2],resultpath)
    
    plt.savefig(figurepath+'/FigS5.png',dpi=600)
    plt.savefig(figurepath+'/FigS5.pdf')
    plt.savefig(figurepath+'/FigS5.eps')

if __name__ == '__main__':

    figurepath  = root_path + '/NetworkReduction/sfig5_multidataaccu/figure'
    resultpath = root_path + '/NetworkReduction/fig4_reductionAccuracy/result'
    
    PlotAccuracyDataSet(figurepath,resultpath)

