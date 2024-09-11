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
        
### mean centring
        figMCplots, axMCplots = plt.subplots(2, 2, figsize=data.fig_settings.fig_Size)
        axMCplots[0,0] = plt.subplot2grid((13,13), (0, 0), colspan=6, rowspan=6)
        axMCplots[0,1] = plt.subplot2grid((13,13), (0, 7), colspan=6, rowspan=6)
        axMCplots[1,0] = plt.subplot2grid((13,13), (7, 0), colspan=6, rowspan=6)
        axMCplots[1,1] = plt.subplot2grid((13,13), (7, 7), colspan=6, rowspan=6)
        figMCplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        axMCplots[0,0].plot( data.pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaNonMC.spectral_loading[:3,:])/2)
                            +data.PCACCpSF.spectral_loading[:3,:].T, lw=2)
        axMCplots[0,0].plot( data.pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaNonMC.spectral_loading[:3,:])/2)
                            +data.pcaNonMC.spectral_loading[:3,:].T,'--k', lw=1.5 )
        axMCplots[0,1].plot( data.pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaNonMC.spectral_loading[:3,:])/2)
                            +data.PCACCxSF.spectral_loading[:3,:].T, lw=2)
        axMCplots[0,1].plot( data.pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaNonMC.spectral_loading[:3,:])/2)
                            +data.pcaNonMC.spectral_loading[:3,:].T,'--k', lw=1.5 )
        axMCplots[1,0].plot( data.pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaMC.spectral_loading[:3,:])/2)
                            +data.PCACCpSFMC.spectral_loading[:3,:].T, lw=2)
        axMCplots[1,0].plot( data.pcaMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaMC.spectral_loading[:3,:])/2)
                            +data.pcaMC.spectral_loading[:3,:].T,'--k', lw=1.5)
        axMCplots[1,1].plot( data.pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaMC.spectral_loading[:3,:])/2)
                            +data.PCACCxSFMC.spectral_loading[:3,:].T, lw=2)
        axMCplots[1,1].plot( data.pcaMC.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaMC.spectral_loading[:3,:])/2)
                            +data.pcaMC.spectral_loading[:3,:].T,'--k', lw=1.5)
        axMCplots[0,0].legend(['PC1','PC2','PC3','Basic'],bbox_to_anchor=(1.2, 0),
                              fontsize=data.fig_settings.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)

        subLabels = [r"a) $L^\top_{Matrix}$ vs $L^\top_{Basic}$",
                     r"b) $L^\top_{Squared}$ vs $L^\top_{Basic}$",
                     r"c) $L^\top_{Matrix}$MC vs $L^\top_{Basic}$MC",
                     r"d) $L^\top_{Squared}$MC vs $L^\top_{Basic}$MC"]
        for ax1 in range(2):
            for ax2 in range(2):
                axMCplots[ax1,ax2].annotate(
                    subLabels[ax1*2+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.5, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="center",
                )
                if not data.fig_settings.fig_Show_Values:
                    axMCplots[ax1,ax2].axis("off")

                if data.fig_settings.fig_Show_Labels:
                    axMCplots[ax1,ax2].set_ylabel(data.fig_settings.fig_Y_Label)
                    axMCplots[ax1,ax2].set_xlabel(data.fig_settings.fig_X_Label)

        image_name = " Figure 7 Mean Centring plots"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figMCplots.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figMCplots.show()
        plt.close()
        print( 'Figure 7 generated at ' + full_path )
        return