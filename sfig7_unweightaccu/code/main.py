# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:51:01 2022

@author: Leyang Xue
"""

#please change the current path if run the code
root_path  = 'G:/work/work5_reductionability'

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
        
def PlotSpread(ax,path,Scliques):
    
    colors = plt.get_cmap('Paired')    
    mec_color = 'black'#'#978C86'#'white'
    markersize = 12
    n_legend = 18
    alp = 1
    
    #load the dataset
    Omcs = cg.load(path +'/simulation/Omcs')
    Rmcs = cg.load(path + '/simulation/Rmcs')
    #Rwmcs = cg.load(path+'\\Rwmcs')
    
    omcs = Omcs/Omcs[-1,0]
    rmcs = Rmcs/Rmcs[-1,0]
    #rwmcs = Rwmcs/Rwmcs[-1,0]
    omc = np.average(omcs,axis =1)

    alphas = np.arange(0,1.01,0.01)
    x = np.arange(0,101,5)
    betac = np.array([0.83,0.76,0.71,0.66,0.61,0.58])
    betac_inx = np.array(list(map(lambda x:int(x*100), betac)))

    for i,sclique in enumerate(Scliques):
        ax.plot(alphas[x],rmcs[:,i][x],'o',ms=markersize, color=colors(i),alpha=alp, mec=mec_color,mew=0.6,label = 'R='+str(sclique+1))

    ax.plot(alphas[x],omc[x],'-',lw=2,color='black',label='O')
    cg.PlotAxes(ax,r'$\beta$',r'$\rho_R$','a')
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)

    #subfig1
    axinx = ax.inset_axes((0.60,0.10,0.35,0.35))
    for i,sclique in enumerate(Scliques):
        axinx.plot(omc[betac_inx[i]],rmcs[betac_inx[i],i],'o',ms=markersize,color=colors(i),mec=mec_color,alpha=alp,mew=0.6)

    axinx.plot(omc[betac_inx],omc[betac_inx],color='black',alpha=alp)
    PlotAxes(axinx,r'$\rho_O$',r'$\rho_R$','',fontsize=16, n_legend = 10)
    axinx.set_xlim(0.58,0.9)
    axinx.set_ylim(0.58,0.9)
    axinx.tick_params(direction='in', which='both',length =3, width=1, labelsize=14)

def PlotProbVector(ax,path,Scliques):
    
    colors = plt.get_cmap('Paired')    
    mec_color = 'black'#978C86'#'white'
    markersize = 12
    alp = 1
    mew = 0.6
    n_legend = 18
    
    #beta_l = np.array([0.83,0.76,0.71,0.66,0.61,0.58])
    #mec=mec_color
    alphas = np.arange(0,1.01,0.05)
    #kt = cg.load(path+'/VS')
    #LoadProbValue(path,Scliques,alphas)
    
    Avg_Dprob = cg.load(path+'/Inf_Prob/Avg_Dprob')
    Std_Dprob = cg.load(path+'/Inf_Prob/Std_Dprob') 
    
    Avg_Pearson = cg.load(path+'/Inf_Prob/Avg_Pearson')
    Std_Pearson = cg.load(path+'/Inf_Prob/Std_Pearson') 
    
    #subfig1
    axinx = ax.inset_axes((0.55,0.58,0.35,0.35))
    
    for i,sclique in enumerate(Scliques):
        #ax.errorbar(alphas,kt[i][2],yerr = kt[i][3],fmt ='o-',capthick=2,capsize=5,ms= markersize,mec=mec_color,color=colors(i),alpha=alp,mew=mew)
        #ax.vlines(beta_l[i],ymin=0.0, ymax=1, color = colors(i), linewidth=1.5,linestyle='--')
        ax.errorbar(alphas,Avg_Dprob[sclique],yerr = Std_Dprob[sclique],fmt ='o-',capthick=2,capsize=5,ms= markersize,mec=mec_color,color=colors(i),alpha=alp,mew=mew)
        axinx.errorbar(alphas[0:-1],Avg_Pearson[sclique][0:-1],yerr = Std_Pearson[sclique][0:-1],fmt ='o-',capthick=2,capsize=5,ms=7,mec=mec_color,color=colors(i),alpha=alp,mew=mew)
    
    axinx.set_xlim(-0.05,1.05)
    axinx.set_ylim(-0.05,1.10)
    ax.set_ylim(-0.05,1.05)
    
    PlotAxes(axinx,r'$\beta$',r'$r$','',fontsize=16, n_legend = 10)
    cg.PlotAxes(ax,r'$\beta$',r'$D_{Prob}$','b')#r'|$P_R^{inf}-P_O^{inf}|$'
    ax.legend(loc='best',framealpha=0, fontsize=n_legend)

def PlotPc(ax,path,Scliques):
    
    Rpc = {}
    Rcv = {}
    alphas = np.arange(0,1.01,0.01)
    spread = np.zeros((0,101))
    colors = plt.get_cmap('Paired')    
    
    axinx = ax.inset_axes((0.30,0.58,0.35,0.35))
    for i,Sclique in enumerate(Scliques):
        
        OSpread = cg.load(path+'/simulation/'+str(Sclique)+'_Onetmc')
        spread = np.row_stack((spread,OSpread))
        RWSpread = cg.load(path+'/simulation/'+str(Sclique)+'_Rnetmc')
        [Rcv[Sclique],Rpc[Sclique]] = cg.IdentifyCriInfectRate(RWSpread.T)
        #ax.plot(alphas[1:],RWcv[Sclique][1:],color=colors(i),linewidth=1.5)
        ax.axvspan(Rpc[Sclique],Rpc[Sclique],color=colors(i),linewidth=1.2,linestyle='solid',label =r'$\beta_c^{R'+str(Sclique+1)+'}$')
        axinx.axvspan(Rpc[Sclique],Rpc[Sclique],color=colors(i),linewidth=1.2,linestyle='solid')
 
    axinx.set_xscale('log')
    [cv,pc] = cg.IdentifyCriInfectRate(spread.T)
    ax.plot(alphas[1:],cv[1:],color='black',linewidth=1.2,linestyle='solid',label = r'$\beta_c^{O}$')
    axinx.plot(alphas[1:],cv[1:],color='black',linewidth=1.2,linestyle='solid')
    ax.axvspan(pc,pc,color='black',linewidth=1.2)
    axinx.axvspan(pc,pc,color='black',linewidth=1.2)

    #fontsize=20 
    n_legend = 18
    #font_label = {'family': "Calibri", 'size':fontsize}
    #ax.set_xlabel(r'$\beta$',  fontdict = font_label)
    #ax.set_ylabel(r'$\chi$', fontdict = font_label)
    #ax.set_title('c', loc='left',y=1.02,fontdict = {'family': "Calibri", 'size':30})
    #ax.tick_params(direction='in', which='both',length =3, width=1, labelsize=n_legend)
    ax.legend(loc='upper right',framealpha=0, fontsize=n_legend)
    PlotAxes(axinx,r'$\beta$',r'$\chi$',fontsize=16,n_legend=10)
    cg.PlotAxes(ax,r'$\beta$',r'$\chi$','c')
    
def PlotSpreadCompare(resultpath,figurepath,files):
    
   Scliques = [4,5,6,7,8,9]
   fig,ax = plt.subplots(1,3,figsize=(21,7),constrained_layout=True)
   path = resultpath+'/'+files[0] 
   
   #PlotWeightSpread(ax[0,0],path,Scliques)
   PlotSpread(ax[0],path,Scliques)
   #PlotWeightProbVector(ax[0,1],path,Scliques)
   PlotProbVector(ax[1],path,Scliques)
   #PlotWeightPc(ax[0,2],path,Scliques)
   PlotPc(ax[2],path,Scliques)
   
   plt.savefig(figurepath+'/FigS7.png',dpi=600)
   plt.savefig(figurepath+'/FigS7.pdf')
   plt.savefig(figurepath+'/FigS7.eps')

if __name__ == '__main__':

    resultpath = root_path + '/NetworkReduction/fig4_reductionAccuracy/result'
    figurepath = root_path + '/NetworkReduction/sfig7_unweightaccu/figure'
    files = ['GrQC','CondMat','HepPh','NetScience']
    
    PlotSpreadCompare(resultpath,figurepath,files)
    