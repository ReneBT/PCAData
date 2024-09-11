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
        
### Variable wise normalisation
        figUVplots, axUVplots = plt.subplots(2, 2, figsize=data.pcaMC.fig_Size)
        axUVplots[0,0] = plt.subplot2grid((13,13), (0, 0), colspan=6, rowspan=6)
        axUVplots[0,1] = plt.subplot2grid((13,13), (0, 7), colspan=6, rowspan=6)
        axUVplots[1,0] = plt.subplot2grid((13,13), (7, 0), colspan=6, rowspan=6)
        axUVplots[1,1] = plt.subplot2grid((13,13), (7, 7), colspan=6, rowspan=6)
        figUVplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        axUVplots[0,0].plot( data.pcaMCUV.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaMCUV.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaMCUV.spectral_loading[:3,:])/2)
                            +data.PCACCxSFMCUV.spectral_loading[:3,:].T, lw=2)
        axUVplots[0,0].plot( data.pcaMCUV.pixel_axis ,
                            np.arange(0,-np.ptp(data.pcaMCUV.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pcaMCUV.spectral_loading[:3,:])/2)
                            +data.pcaMCUV.spectral_loading[:3,:].T, '--k', lw=1.5)
        MAD_MCUVvsxSFMCUV = np.absolute(data.PCACCxSFMCUV.spectral_loading[:3,:]-data.pcaMCUV.spectral_loading[:3,:]).mean() #mean absolute difference
        print('Mean Absolute Deviation for correcting scale with MCUV:' + str( MAD_MCUVvsxSFMCUV))


        axUVplots[0,1].plot( data.pca_1_noiseUV.pixel_axis ,
                            np.arange(0,-np.ptp(data.pca_1_noise.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pca_1_noise.spectral_loading[:3,:])/2)
                            +data.pca_1_noiseUV.spectral_loading[:3,:].T, lw=2)
        axUVplots[0,1].plot( data.pca_1_noise.pixel_axis ,
                            np.arange(0,-np.ptp(data.pca_1_noise.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pca_1_noise.spectral_loading[:3,:])/2)
                            +data.pca_1_noise.spectral_loading[:3,:].T,'--k', lw=1.5)

        axUVplots[1,0].plot( data.pca_1_noiseSqrt.pixel_axis ,
                            np.arange(0,-np.ptp(data.pca_1_noiseSqrt.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pca_1_noiseSqrt.spectral_loading[:3,:])/2)
                            +data.pca_1_noiseSqrt.spectral_loading[:3,:].T, lw=2)
        axUVplots[1,1].plot( data.pca_1_noiseLn.pixel_axis ,
                            np.arange(0,-np.ptp(data.pca_1_noiseLn.spectral_loading[:3,:])*1.1,
                                      -np.ptp(data.pca_1_noiseLn.spectral_loading[:3,:])/2)
                            +data.pca_1_noiseLn.spectral_loading[:3,:].T, lw=2)
        axUVplots[0,0].legend(['PC1','PC2','PC3','Basic'],bbox_to_anchor=(1.2, 0),
                              fontsize=data.pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)

        subLabels = [r"a) $L^\top_{Squared}$ MCUV vs $L^\top_{Basic}$ MCUV ",
                     r"b) $L^\top_{NoisyBasic}$ MCUV vs $L^\top_{NoisyBasic}$ MC",
                     r"c) $L^\top_{\sqrt{NoisyBasic}}$ MC",
                     r"d) $L^\top_{Ln(NoisyBasic)}$ MC "]
        #find out how to display a square root rather than calculate one
        for ax1 in range(2):
            for ax2 in range(2):
                axUVplots[ax1,ax2].annotate(
                    subLabels[ax1*2+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.5, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="center",
                )
                if not data.pcaMC.fig_Show_Values:
                    axUVplots[ax1,ax2].axis("off")

                if data.pcaMC.fig_Show_Labels:
                    axUVplots[ax1,ax2].set_ylabel(data.pcaMC.fig_Y_Label)
                    axUVplots[ax1,ax2].set_xlabel(data.pcaMC.fig_X_Label)

        image_name = " Figure 8 Scaling plots"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figUVplots.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figUVplots.show()
        plt.close()
        print( 'Figure 8 generated at ' + full_path )
        return