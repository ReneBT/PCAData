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
        # 1 create 3 noise variant replicates of one sample and plot overlaid
        # 2 reconstruct replicates from noiseless PCA and plot overlaid, offset from 1
        # 3 subtract noiseless original spectrum then plot residuals overlaid, offset from 2. Scale up if necessary to compare, annotate with scaling used
        figRecon, axRecon = plt.subplots(1, 1, figsize=data.fig_settings.fig_Size)
        figRecon.subplots_adjust(wspace = 0.35)
        axRecon = plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
        axRecon.plot(data.pca_noise.pixel_axis, data.reps_4_noise)
        offset = np.mean(data.reps_4_noise[:])*2
        ypos = np.array([0, -offset, -1.75*offset, -2.75*offset, -4*offset])
        axRecon.plot(data.pca_noise.pixel_axis, data.reps_4_noise_recon+ypos[1])
        axRecon.plot(data.pca_noise.pixel_axis, (data.reps_4_noise.T-data.data[:,23]).T+ypos[2])
        axRecon.plot(data.pca_noise.pixel_axis, (data.reps_4_noise_recon.T-data.data[:,23]).T+ypos[3])
        axRecon.plot(data.pca_noise.pixel_axis, 10*(data.reps_4_noise_recon.T-data.data[:,23]).T+ypos[4])
        axRecon.set_ylabel("Intensity")
        axRecon.set_yticklabels([])
        axRecon.set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)

        subsubStr2 = ["Noisy Repetitions", "Reconstructions", "Noise in Repetitions", "Reconstruction Error","Reconstruction Error x10"]
        ypos = ypos+50
        for subsub in np.arange(5):
            axRecon.annotate(
                subsubStr2[subsub],
                xy=(data.pca_noise.pixel_axis[365],ypos[subsub]),
                xytext=(data.pca_noise.pixel_axis[365],ypos[subsub]),
                xycoords="data",
                textcoords="data",
                fontsize=data.fig_settings.fig_Text_Size*0.75,
                horizontalalignment="center",
                va="center",
                alpha = 0.75,
                bbox=dict(boxstyle='square,pad=0', fc='w', ec='none')
            )

        image_name = " Figure 11 reconstuction for noisy data"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figRecon.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figRecon.show()
        plt.close()
        print( 'Figure 11 generated at ' + full_path )
        return