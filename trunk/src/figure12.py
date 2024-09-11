# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 06:57:54 2024

@author: rene
"""
import matplotlib.pyplot as plt
import os
from src.file_locations import images_folder

class fig( ):
    def __init__( self, data ):
### Model Correlation Plot
        figNoiseCorr, axNoiseCorr = plt.subplots(2, 3, figsize=data.fig_settings.fig_Size)
        axNoiseCorr[0,0] = plt.subplot2grid((2,11),(0,0),colspan=3, rowspan=1)
        axNoiseCorr[0,1] = plt.subplot2grid((2,11),(0,4),colspan=3, rowspan=1)
        axNoiseCorr[0,2] = plt.subplot2grid((2,11),(0,8),colspan=3, rowspan=1)
        axNoiseCorr[1,0] = plt.subplot2grid((2,11),(1,0),colspan=3, rowspan=1)
        axNoiseCorr[1,1] = plt.subplot2grid((2,11),(1,4),colspan=3, rowspan=1)
        axNoiseCorr[1,2] = plt.subplot2grid((2,11),(1,8),colspan=3, rowspan=1)
        figNoiseCorr.subplots_adjust(left=0.07, right=0.99, top=0.95, wspace = 0.1, hspace=0.4)

        axNoiseCorr[0,0].plot(range(1,data.pcaMC.N_PC+1), data.corrPCs_noiseq[:,range(7)][range(data.pcaMC.N_PC),:]**2,)
        axNoiseCorr[0,1].plot(range(1,data.pcaMC.N_PC+1), data.corrPCs_noiseq[range(0,7),:][:,range(data.pcaMC.N_PC)].T**2,)
        axNoiseCorr[0,2].plot(range(1,data.pcaMC.N_PC+1),data.corrPCs_noiseq_R2sum_ax1[range(data.pcaMC.N_PC)],linewidth=2)
        axNoiseCorr[0,2].plot(range(1,data.pcaMC.N_PC+1),data.corrPCs_noiseq_R2sum_ax0[range(data.pcaMC.N_PC)],linewidth=1.5)
        axNoiseCorr[0,2].plot(range(1,data.pcaMC.N_PC+1),data.maxCorr_noiseq[range(data.pcaMC.N_PC)]**2,linewidth=1)
        axNoiseCorr[0,0].set_ylabel("R$^2$")
        axNoiseCorr[0,0].set_xlabel("PC in SNR$_{400}$")
        axNoiseCorr[0,0].set_xticks(range(1,data.pcaMC.N_PC,2))
        axNoiseCorr[0,1].set_ylabel("R$^2$")
        axNoiseCorr[0,1].set_xlabel("PC in SNR$_\infty$")
        axNoiseCorr[0,1].set_xticks(range(1,data.pcaMC.N_PC,2))
        axNoiseCorr[0,2].set_ylabel("Total R$^2$")
        axNoiseCorr[0,2].set_xlabel("PC rank")
        axNoiseCorr[0,2].set_xticks(range(1,data.pcaMC.N_PC,2))
        axNoiseCorr[0,0].legend(range(1,7),fontsize="small",loc=(0.7,0.25))
        axNoiseCorr[0,0].annotate(
            "PC in SNR$_\infty$",
            xy=(8.5,0.97),
            xytext=(8.5,0.99),
            xycoords="data",
            textcoords="data",
            fontsize=data.fig_settings.fig_Text_Size*0.75,
            horizontalalignment="center",
            va="center",
            alpha = 0.75,
            bbox=dict(boxstyle='square,pad=0', fc='w', ec='none')
        )
        axNoiseCorr[0,1].annotate(
            "PC in SNR$_{400}$",
            xy=(8.5,0.97),
            xytext=(8.5,0.99),
            xycoords="data",
            textcoords="data",
            fontsize=data.fig_settings.fig_Text_Size*0.75,
            horizontalalignment="center",
            va="center",
            alpha = 0.75,
            bbox=dict(boxstyle='square,pad=0', fc='w', ec='none')
        )
        axNoiseCorr[0,1].legend(range(1,7),fontsize="small",loc=(0.7,0.25))
        axNoiseCorr[0,2].legend(("$\Sigma R^2_{400 \mapsto \infty}$",
                               "$\Sigma R^2_{\infty \mapsto 400}$",
                               "$\max (R^2_{400 \mapsto \infty})$"),
                              fontsize="x-small",loc='upper right')
        axNoiseCorr[1,0].imshow( data.corrPCs_noiseq[:10,:10],vmin=0,vmax=1)
        axNoiseCorr[1,1].imshow( data.corrPCs_noise1[:10,:10],vmin=0,vmax=1)
        axNoiseCorr[1,2].imshow( data.corrPCs_noise4[:10,:10],vmin=0,vmax=1)
        for ax2 in range(3):
            axNoiseCorr[1,ax2].set_yticks(range(0,10))
            axNoiseCorr[1,ax2].set_yticklabels(range(1,11))
            axNoiseCorr[1,ax2].set_xticks(range(0,10))
            axNoiseCorr[1,ax2].set_xticklabels(range(1,11))
            axNoiseCorr[1,ax2].set_ylim([-0.5,9.5])
            axNoiseCorr[1,ax2].set_xlim([-0.5,9.5])
            axNoiseCorr[1,ax2].set_xlabel('PC$_{\infty}$')
            axNoiseCorr[1,ax2].set_ylabel(r'PC$_{400}$')

        subLabels = [r"a) PC$_{SNR\infty} \mapsto PC_{SNR400}$",
                     r"b) PC$_{SNR400} \mapsto PC_{SNR\infty}$",
                     r"c) $\Sigma  \, & \, \max \, R^2$",
                     r"d) r PC$_{SNR\infty}$ vs PC$_{SNR400}$",
                     r"e) r PC$_{SNR\infty}$ vs PC$_{SNR100}$",
                     r"f) r PC$_{SNR\infty}$ vs PC$_{SNR25}$",
                     ]
        for ax1 in range(3):
            for ax2 in range(2):
                axNoiseCorr[ax2,ax1].annotate(
                    subLabels[ax2*3+ax1],
                    xy=(0.18, 0.89),
                    xytext=(0, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )

        image_name = " Figure 12 correlations noisy vs noiseless"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figNoiseCorr.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figNoiseCorr.show()
        plt.close()
        print( 'Figure 12 generated at ' + full_path )
        return