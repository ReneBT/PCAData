# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 06:57:54 2024

@author: rene
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.file_locations import images_folder
from matplotlib import patches

class fig( ):
    def __init__( self, data ):
        # Generate first figure in paper, showing the generated basis spectra 
        # and the generated dataset
        
### Sample wise Normalisation
        figIntplots, axIntplots = plt.subplots(1, 4, figsize=data.fig_settings.fig_Size)
        axIntplots[0] = plt.subplot2grid((1,27), (0, 0), colspan=6)
        axIntplots[1] = plt.subplot2grid((1,27), (0, 7), colspan=6)
        axIntplots[2] = plt.subplot2grid((1,27), (0, 14), colspan=6)
        axIntplots[3] = plt.subplot2grid((1,27), (0, 21), colspan=6)
        figIntplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)


        axIntplots[0].plot( data.PCA_Int.pixel_axis ,
                            np.arange(0,-np.ptp(data.PCA_Int.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.PCA_Int.spectral_loading[:3,:])/2)
                            +data.PCA_Int.spectral_loading[:3,:].T, lw=2)
        axIntplots[0].add_artist(patches.Ellipse((data.PCA_Int.pixel_axis[250][0],
                                          np.min(data.PCA_Int.spectral_loading[0,200:300]) + np.ptp(data.PCA_Int.spectral_loading[0,200:300])/2),
                                         width=data.PCA_Int.pixel_axis[300]-data.PCA_IntCO.pixel_axis[200],
                                         height=np.ptp(data.PCA_Int.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        axIntplots[1].plot( data.PCA_IntCO.pixel_axis ,
                            np.arange(0,-np.ptp(data.PCA_IntCO.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.PCA_IntCO.spectral_loading[:3,:])/2)
                            +data.PCA_IntCO.spectral_loading[:3,:].T,lw=2 )
        axIntplots[1].add_artist(patches.Ellipse((data.PCA_IntCO.pixel_axis[250][0],
                                          np.min(data.PCA_IntCO.spectral_loading[0,200:300]) + np.ptp(data.PCA_IntCO.spectral_loading[0,200:300])/2),
                                         width=data.PCA_IntCO.pixel_axis[300]-data.PCA_IntCO.pixel_axis[200],
                                         height=np.ptp(data.PCA_IntCO.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        cylim = axIntplots[1].get_ylim()
        axIntplots[1].plot(np.tile(data.PCA_IntCO.pixel_axis[520],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[1].plot(np.tile(data.PCA_IntCO.pixel_axis[570],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[1].plot([data.PCA_IntCO.pixel_axis[520],data.PCA_IntCO.pixel_axis[570]],np.tile(cylim[0]*0.9,2),'g',linestyle='--')
        axIntplots[1].plot([data.PCA_IntCO.pixel_axis[520],data.PCA_IntCO.pixel_axis[570]],np.tile(cylim[1]*0.9,2),'g',linestyle='--')
        axIntplots[1].set_ylim(cylim)
        axIntplots[2].plot( data.PCA_IntCHx.pixel_axis ,
                            np.arange(0,-np.ptp(data.PCA_IntCHx.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.PCA_IntCHx.spectral_loading[:3,:])/2)
                            +data.PCA_IntCHx.spectral_loading[:3,:].T, lw=2)
        axIntplots[2].add_artist(patches.Ellipse((data.PCA_IntCHx.pixel_axis[250][0],
                                          np.min(data.PCA_IntCHx.spectral_loading[0,200:300]) + np.ptp(data.PCA_IntCHx.spectral_loading[0,200:300])/2),
                                         width=data.PCA_IntCHx.pixel_axis[300]-data.PCA_IntCHx.pixel_axis[200],
                                         height=np.ptp(data.PCA_IntCHx.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        cylim = axIntplots[2].get_ylim()
        axIntplots[2].plot(np.tile(data.PCA_IntCHx.pixel_axis[210],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[2].plot(np.tile(data.PCA_IntCHx.pixel_axis[280],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[2].plot([data.PCA_IntCHx.pixel_axis[210],data.PCA_IntCHx.pixel_axis[280]],np.tile(cylim[0]*0.9,2),'g',linestyle='--')
        axIntplots[2].plot([data.PCA_IntCHx.pixel_axis[210],data.PCA_IntCHx.pixel_axis[280]],np.tile(cylim[1]*0.9,2),'g',linestyle='--')
        axIntplots[2].set_ylim(cylim)
        axIntplots[3].plot( data.PCA_IntVec.pixel_axis ,
                            np.arange(0,-np.ptp(data.PCA_IntVec.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.PCA_IntVec.spectral_loading[:3,:])/2)
                            +data.PCA_IntVec.spectral_loading[:3,:].T,lw=2 )
        axIntplots[3].add_artist(patches.Ellipse((data.PCA_IntVec.pixel_axis[250][0],
                                          np.min(data.PCA_IntVec.spectral_loading[0,200:300]) + np.ptp(data.PCA_IntVec.spectral_loading[0,200:300])/2),
                                         width=data.PCA_IntVec.pixel_axis[300]-data.PCA_IntVec.pixel_axis[200],
                                         height=np.ptp(data.PCA_IntVec.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        cylim = axIntplots[3].get_ylim()
        axIntplots[3].plot(np.tile(data.PCA_IntVec.pixel_axis[0],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[3].plot(np.tile(data.PCA_IntVec.pixel_axis[600],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[3].plot([data.PCA_IntVec.pixel_axis[0],data.PCA_IntVec.pixel_axis[600]],np.tile(cylim[0]*0.9,2),'g',linestyle='--')
        axIntplots[3].plot([data.PCA_IntVec.pixel_axis[0],data.PCA_IntVec.pixel_axis[600]],np.tile(cylim[1]*0.9,2),'g',linestyle='--')
        axIntplots[3].set_ylim(cylim)

        subLabels = [r"a) $L^\top_{Int}$ MC Unnorm", r"b) $L^\top_{Int}$ MC/C=O",
                     r"c) $L^\top_{Int}$ MC/CH$_x$", r"d)$L^\top_{Int}$ MC/Norm"]
        #find out how to display a square root rather than calculate one
        for ax1 in range(axIntplots.shape[0]):
            axIntplots[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0.5, 0.98),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="center",
            )
            if not data.fig_settings.fig_Show_Values:
                axIntplots[ax1].axis("off")

            if data.fig_settings.fig_Show_Labels:
                axIntplots[ax1].set_ylabel(data.fig_settings.fig_Y_Label)
                axIntplots[ax1].set_xlabel(data.fig_settings.fig_X_Label)

        image_name = " Figure 9 Normalisation plots"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figIntplots.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figIntplots.show()
        plt.close()
        print( 'Figure 9 generated at ' + full_path )
        return