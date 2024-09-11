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
### Data perturbation
        figData4Preprocessing, axData4Preprocessing = plt.subplots(2, 2, figsize=data.fig_settings.fig_Size)
        axData4Preprocessing[0,0] = plt.subplot2grid((2, 12), (0, 0), colspan=5)
        axData4Preprocessing[0,1] = plt.subplot2grid((2, 12), (0, 7), colspan=5)
        axData4Preprocessing[1,0] = plt.subplot2grid((2, 12), (1, 0), colspan=5)
        axData4Preprocessing[1,1] = plt.subplot2grid((2, 12), (1, 7), colspan=5)
        figData4Preprocessing.subplots_adjust(left=0.11,right=0.99,top=0.97,bottom=0.1)

        axData4Preprocessing[0,0].plot( data.wavelength_axis ,  data.data)
        axData4Preprocessing[0,0].plot(data.wavelength_axis[[104,204]],np.tile(np.max(data.data[101,:]),2),'r')
        axData4Preprocessing[0,0].plot(data.wavelength_axis[[104,204]],np.tile(np.min(data.data[101,:]),2),'k--')
        axData4Preprocessing[0,1].annotate(
            "a)",
            xy=(0.13, 0.96),
            xytext=(0.01, 0.94),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size,
            horizontalalignment="left",
        )
        axData4Preprocessing[0,0].annotate(
            "range: %.0f" %np.ptp(data.data[101,:]),
            xy=(0.13, 0.96),
            xytext=(data.wavelength_axis[110],np.mean(data.data[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size*0.75,
            color = 'k',
            horizontalalignment="left",
            va = 'center'
        )

        axData4Preprocessing[0,1].plot( data.wavelength_axis ,  data.reps_4_noise)
        axData4Preprocessing[0,1].plot(data.wavelength_axis[[104,204]],np.tile(np.max(data.reps_4_noise[101,:]),2),'r')
        axData4Preprocessing[0,1].plot(data.wavelength_axis[[104,204]],np.tile(np.min(data.reps_4_noise[101,:]),2),'k--')
        axData4Preprocessing[0,1].annotate(
            "b)",
            xy=(0.13, 0.96),
            xytext=(0.01, 0.94),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size,
            horizontalalignment="left",
        )
        axData4Preprocessing[0,1].annotate(
            "range: %.0f" %np.ptp(data.reps_4_noise[101,:]),
            xy=(0.13, 0.96),
            xytext=(data.wavelength_axis[110],np.mean(data.reps_4_noise[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size*0.75,
            color = 'k',
            horizontalalignment="left",
            va = 'center'
        )

        axData4Preprocessing[1,0].plot( data.wavelength_axis , data.spectraCCpSF)
        axData4Preprocessing[1,0].plot(data.wavelength_axis[[104,204]],np.tile(np.max(data.spectraCCpSF[101,:]),2),'r')
        axData4Preprocessing[1,0].plot(data.wavelength_axis[[104,204]],np.tile(np.min(data.spectraCCpSF[101,:]),2),'k--')
        axData4Preprocessing[1,0].annotate(
            "range: %.0f" %np.ptp(data.spectraCCpSF[101,:]),
            xy=(0.13, 0.96),
            xytext=(data.wavelength_axis[110],np.mean(data.spectraCCpSF[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size*0.75,
            color = 'k',
            horizontalalignment="left",
            va = 'center'
        )
        axData4Preprocessing[1,0].set_ylabel("Intensity / Counts")
        axData4Preprocessing[1,0].set_xlabel("Raman Shift cm$^{-1}$")
        axData4Preprocessing[1,0].annotate(
            "c)",
            xy=(0.13, 0.96),
            xytext=(0.01, 0.94),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size,
            horizontalalignment="left",
        )

        axData4Preprocessing[1,1].plot( data.wavelength_axis , data.spectraCCxSF  )
        axData4Preprocessing[1,1].set_ylabel("Intensity / Counts")
        axData4Preprocessing[1,1].set_xlabel("Raman Shift cm$^{-1}$")
        axData4Preprocessing[1,1].plot(data.wavelength_axis[[104,204]],np.tile(np.max(data.spectraCCxSF[101,:]),2),'r')
        axData4Preprocessing[1,1].plot(data.wavelength_axis[[104,204]],np.tile(np.min(data.spectraCCxSF[101,:]),2),'k--')
        axData4Preprocessing[1,1].annotate(
            "d)",
            xy=(0.13, 0.96),
            xytext=(0.01, 0.94),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size,
            horizontalalignment="left",
        )
        axData4Preprocessing[1,1].annotate(
            "range: %.0f" %np.ptp(data.spectraCCxSF[101,:]),
            xy=(0.13, 0.96),
            xytext=(data.wavelength_axis[110],np.mean(data.spectraCCxSF[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.fig_settings.fig_Text_Size*0.75,
            color = 'k',
            horizontalalignment="left",
            va = 'center'
        )


        image_name = " Figure 2 Spectra offset and scale"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)

        figData4Preprocessing.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figData4Preprocessing.show()
        plt.close()
        print( 'Figure 2 generated at ' + full_path )
        return