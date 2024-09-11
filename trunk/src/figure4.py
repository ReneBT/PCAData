# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 06:57:54 2024

@author: rene
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.file_locations import images_folder

class fig( ):
    def __init__( self, data ):
        # Generate first figure in paper, showing the generated basis spectra 
        # and the generated dataset
### Spectra vs GC Variation
        figGCscores, axGCscores = plt.subplots(1, 3, figsize=(
            data.fig_settings.fig_Size[0],data.fig_settings.fig_Size[0]/3))
        axGCscores[0] = plt.subplot2grid((1,22), (0, 0), colspan=6 )
        axGCscores[1] = plt.subplot2grid((1,22), (0, 8), colspan=6 )
        axGCscores[2] = plt.subplot2grid((1,22), (0, 16), colspan=6 )
        figGCscores.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.18)
        grps = [[0,0,2,2,4,4,6,6],[False,True,False,True,False,True,False,True],['db','db','or','or','^c','^c','sm','sm']]
        for iS in range(8):
            if grps[1][iS]:
                cGrp = data.Grouping
                fillS = 'full'
            else:
                cGrp = ~data.Grouping
                fillS = 'none'
            ix = np.where(np.logical_and(data.feed==grps[0][iS] , cGrp))[0]
            axGCscores[0].plot(data.pcaMC.component_weight[ix, 0],data.pcaMC.component_weight[ix, 1],grps[2][iS],fillstyle=fillS,markersize=6)
            axGCscores[1].plot(-data.GCsc[ix, 0],-data.GCsc[ix, 1],grps[2][iS],fillstyle=fillS,markersize=6)
        axGCscores[2].plot(data.GCsc[:, 2]/np.ptp(data.GCsc[:, 2]),data.pcaMC.component_weight[:, 2]/np.ptp(data.pcaMC.component_weight[:, 2]),'<',color=[0.85,0.85,0.85],markersize=6)
        axGCscores[2].plot(-data.GCsc[:, 1]/np.ptp(data.GCsc[:, 1]),data.pcaMC.component_weight[:, 1]/np.ptp(data.pcaMC.component_weight[:, 1]),'+',color=[0.4,0.4,0.4],markersize=6)
        axGCscores[2].plot(-data.GCsc[:, 0]/np.ptp(data.GCsc[:, 0]),data.pcaMC.component_weight[:, 0]/np.ptp(data.pcaMC.component_weight[:, 0]),'.k',markersize=6)

        axGCscores[0].set_xlabel('t[1]Spectral',labelpad=-1)
        axGCscores[0].set_ylabel('t[2]Spectral',labelpad=-1)
        axGCscores[0].legend(['0mg E','0mg L','2mg','_','4mg','_','6mg'],
                              fontsize=data.fig_settings.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)
        axGCscores[2].legend(['#3','#2','#1'],
                              fontsize=data.fig_settings.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)
        axGCscores[1].set_xlabel('t[1]GC',labelpad=-1)
        axGCscores[1].set_ylabel('t[2]GC',labelpad=-1)
        axGCscores[2].set_xlabel('t[#]GC',labelpad=-1)
        axGCscores[2].set_ylabel('t[#]Spectral',labelpad=-1)
        subLabels = ["a)","b)", "c)", "d)"]
        for ax1 in range(3):
            axGCscores[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0.025, 1.025),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="left",
            )

        image_name = " Figure 4 Spectral vs GC score plots"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figGCscores.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figGCscores.show()
        plt.close()
        print( 'Figure 4 generated at ' + full_path )
        return