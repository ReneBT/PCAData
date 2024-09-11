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
### Scree Plot
        figScree, axScree = plt.subplots(2, 3, figsize=data.fig_settings.fig_Size)
        figScree.subplots_adjust(wspace = 0.35)
        axScree[0,0] = plt.subplot2grid((9,11),(0,0),colspan=3, rowspan=4)
        axScree[0,1] = plt.subplot2grid((9,11),(0,4),colspan=3, rowspan=4)
        axScree[0,2] = plt.subplot2grid((9,11),(0,8),colspan=3, rowspan=4)
        axScree[1,0] = plt.subplot2grid((9,11),(5,0),colspan=11, rowspan=4)

        axScree[0,0].plot(range(1,11), data.pcaMC.Eigenvalue[:10]**0.5, "k")
        axScree[0,0].plot(range(1,11), data.pca_q_noise.Eigenvalue[:10]**0.5, "b")
        axScree[0,0].plot(range(1,11), data.pca_1_noise.Eigenvalue[:10]**0.5, "r")
        axScree[0,0].plot(range(1,11), data.pca_4_noise.Eigenvalue[:10]**0.5, "g")
        axScree[0,0].plot(range(1,11), 0.25*data.pca_noise.Eigenvalue[:10]**0.5, 
                          "--",color=[0.5, 0.5, 0.5])
        axScree[0,0].plot(range(1,11), 0.25*data.pca_noise.Eigenvalue[:10]**0.5, 
                          "--",color=[0.5, 0.5, 1])
        axScree[0,0].plot(range(1,11), data.pca_noise.Eigenvalue[:10]**0.5, 
                          "--",color=[1, 0.5, 0.5])
        axScree[0,0].plot(range(1,11), 4*data.pca_noise.Eigenvalue[:10]**0.5,
                          "--",color=[0.5, 1, 0.5])
        axScree[0,0].set_ylabel("Eigenvalue", labelpad=0)
        axScree[0,0].set_xlabel("PC rank", labelpad=0)
        axScree[0,0].legend(("$SNR_\infty$","$SNR_{400}$","$SNR_{100}$",
                             "$SNR_{25}$","Noise"),fontsize='small')

        axScree[0,1].plot(range(1,11), np.cumsum(100*(
            data.pcaMC.Eigenvalue[:10]**0.5)/sum(data.pcaMC.Eigenvalue**0.5)), 
            "k")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(
            data.pca_q_noise.Eigenvalue[:10]**0.5)/sum(
                data.pca_q_noise.Eigenvalue**0.5)), "b")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(
            data.pca_1_noise.Eigenvalue[:10]**0.5)/sum(
                data.pca_1_noise.Eigenvalue**0.5)), "r")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(
            data.pca_4_noise.Eigenvalue[:10]**0.5)/sum(
                data.pca_4_noise.Eigenvalue**0.5)), "g")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(
            data.pca_noise.Eigenvalue[:10]**0.5)/sum(
                data.pca_noise.Eigenvalue**0.5)), "--",color=[0.75,0.75,0.75])
        axScree[0,1].set_ylabel("% Variance explained", labelpad=0)
        axScree[0,1].set_xlabel("PC rank", labelpad=0)

        axScree[0,2].plot(range(1,11), np.abs( np.diagonal(
            data.corrPCs_noise4)[:10]), color = (0, 0.75, 0))
        axScree[0,2].plot(range(1,11), data.maxCorr_noise4[:10], 
                          color = (0.25, 1, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs( np.diagonal(
            data.corrPCs_noise1)[:10]), color = (0.75, 0, 0))
        axScree[0,2].plot(range(1,11), data.maxCorr_noise1[:10], 
                          color = ( 1 , 0.25, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs( np.diagonal(
            data.corrPCs_noiseq)[:10]), color = (0, 0, 0.75))
        axScree[0,2].plot(range(1,11), data.maxCorr_noiseq[:10], 
                          color = (0.25, 0.5, 1), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs( np.diagonal(
            data.corrPCs_noise)[:10]), color = (0.45, 0.45, 0.45))
        axScree[0,2].plot(range(1,11), data.maxCorr_noise[:10], 
                          color = (0.75, 0.75, 0.75), linewidth=0.75)
        axScree[0,2].set_ylabel("Correlation vs noiseless", labelpad=0)
        axScree[0,2].set_xlabel("PC rank", labelpad=0)

        axScree[1,0].plot(data.pca_noise.pixel_axis, 
                          data.pca_noise.spectral_loading[:3].T-[0,0.3,0.6],
                          color = (0.75, 0.75, 0.75),linewidth=2)
        axScree[1,0].plot(data.pca_4_noise.pixel_axis, 
                          data.pca_4_noise.spectral_loading[:3].T-[0,0.3,0.6],
                          color = (0, 1, 0),linewidth=1.75)
        axScree[1,0].plot(data.pca_1_noise.pixel_axis, 
                          data.pca_1_noise.spectral_loading[:3].T-[0,0.3,0.6],
                          color = (1, 0.15, 0.15),linewidth=1.5)
        axScree[1,0].plot(data.pca_q_noise.pixel_axis, 
                          data.pca_q_noise.spectral_loading[:3].T-[0,0.3,0.6],
                          color = (0.3, 0.3, 1),linewidth=1.25)
        axScree[1,0].plot(data.pcaMC.pixel_axis, 
                          data.pcaMC.spectral_loading[:3].T-[0,0.3,0.6],
                          color = (0, 0, 0),linewidth = 1)
        axScree[1,0].set_ylabel("Weight")
        axScree[1,0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axScree[1,0].set_yticklabels([])
        axScree[1,0].annotate("",
                xy=(data.pca_noise.pixel_axis[100], 
                    data.pca_noise.spectral_loading[0,100]),
                xytext=(data.pca_noise.pixel_axis[180], 
                        data.pca_noise.spectral_loading[0,100]),
                textcoords="data",
                xycoords="data",
                fontsize=data.pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                arrowprops=dict( lw=0.1, color='c',headwidth=5)
            )
        axScree[1,0].annotate("",
                xy=(data.pca_noise.pixel_axis[237], 
                    data.pca_noise.spectral_loading[0,237]),
                xytext=(data.pca_noise.pixel_axis[317], 
                        data.pca_noise.spectral_loading[0,237]),
                textcoords="data",
                xycoords="data",
                fontsize=data.pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                arrowprops=dict( lw=0.1, color='c',headwidth=5)
            )
        for subsub in np.arange(3):
            axScree[1,0].annotate(
                str(subsub+1),
                xy=(data.pca_noise.pixel_axis[0]*0.98,-0.3*subsub),
                xytext=(data.pca_noise.pixel_axis[0]*0.98,-0.3*subsub),
                xycoords="data",
                textcoords="data",
                fontsize=data.pcaMC.fig_Text_Size*0.75,
                horizontalalignment="left",
                va="center",
                alpha = 0.75,
                bbox=dict(boxstyle='square,pad=0', fc='w', ec='none')
            )
        subLabels = [r"a) Scree Plot", r"b) Cumulative Variance",
                     r"c) PC Correlations", r"d) Loadings", r"skip",
                     r"e) Reconstructed Data"]
        for ax1 in range(2):
            for ax2 in range(3):
                if (ax1==1 and ax2>1)==False:
                    axScree[ax1,ax2].annotate(
                        subLabels[ax1*3+ax2],
                        xy=(0.18, 0.89),
                        xytext=(0, 1.02),
                        textcoords="axes fraction",
                        xycoords="axes fraction",
                        fontsize=data.pcaMC.fig_Text_Size*0.75,
                        horizontalalignment="left",
                    )
        image_name = " Figure 10 Comparing PCAs for different levels of noise"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figScree.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figScree.show()
        plt.close()
        print( 'Figure 10 generated at ' + full_path )
        return