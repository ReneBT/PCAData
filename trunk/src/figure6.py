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
        
        # variation due to reference data
        
        figFA, axFA = plt.subplots(1, 2, figsize=data.fig_settings.fig_Size)
        axFA[0] = plt.subplot2grid((1,20), (0, 14), colspan=6, )
        axFA[1] = plt.subplot2grid((1,20), (0, 0), colspan=12, )
        figFA.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        lOff = np.round(-(np.max(data.pcaFA.spectral_loading[1,:])-np.min(data.pcaFA.spectral_loading[0,:]))*12.5)/10 #offset by a simple number
        axFA[0].plot(data.wavelength_axis,data.pcaFA.spectral_loading[0,:],'k')
        axFA[0].plot(data.wavelength_axis,data.pcaFA.spectral_loading[1,:] + lOff,color=[0.4,0.4,0.4])
        axFA[0].plot([data.wavelength_axis[0],data.wavelength_axis[-1]],[0,0],'--',color=[0.7,0.7,0.7],lw=0.5)
        axFA[0].plot([data.wavelength_axis[0],data.wavelength_axis[-1]],[lOff,lOff],'--',color=[0.7,0.7,0.7],lw=0.5)

        axFA[1].plot(data.pcaFA.component_weight[:,0],data.pcaFA.component_weight[:,1],'.')
        offset = 1
        for iFA in range(data.pcaFA.N_Obs):
            if data.FAnames[iFA][-1]=='0':
                col = [1,0,0]
            elif data.FAnames[iFA][-1]=='t':
                col = [0,1,0]
            elif data.FAnames[iFA][-1]=='c':
                col = [0,0,1]
            else:
                col = [0,0.5,1]
                offset = 0.75
            p1 = data.pcaFA.component_weight[iFA,0]
            p2 = data.pcaFA.component_weight[iFA,1]
            axFA[1].annotate(  data.FAnames[iFA],
                                xy=(p1, p2),
                                xytext=(p1+offset*np.sign(p1), p2+offset*np.sign(p2)),
                                textcoords="data",xycoords="data",
                                fontsize=data.fig_settings.fig_Text_Size*0.75,
                                horizontalalignment="center",
                                color=col,
                                )


        axFA[0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axFA[0].set_ylabel('Spectral Coefficient',labelpad=-1)

        axFA[1].set_xlabel('t[1]Spectral',labelpad=-1)
        axFA[1].set_ylabel('t[2]Spectral',labelpad=-1)

        subLabels = ["b)","a)"]
        for ax1 in range(2):
            axFA[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0.1, 1.01),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="left",
            )
        axFA[0].annotate(
            'PC1',
            xy=(0.18, 0.89),
            xytext=(1550, data.pcaFA.spectral_loading[0,349]+0.01),
            textcoords="data",
            xycoords="axes fraction",
            horizontalalignment="center",
        )
        axFA[0].annotate(
            'PC2',
            xy=(0.18, 0.89),
            xytext=(1550, data.pcaFA.spectral_loading[1,349]+lOff+0.01),
            textcoords="data",
            xycoords="axes fraction",
            horizontalalignment="center",
        )

        image_name = " Figure 6 reference FA PCA"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figFA.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figFA.show()
        plt.close()
        print( 'Figure 6 generated at ' + full_path )
        return