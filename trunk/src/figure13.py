# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:44:41 2023

@author: rene
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from src.file_locations import images_folder

class fig( ):
    def __init__( self, data ):
        
        figBias, axBias = plt.subplots(1, 2, figsize=data.fig_settings.fig_Size)
        axBias[0] = plt.subplot2grid((1, 11), (0, 0), colspan=5)
        axBias[1] = plt.subplot2grid((1, 11), (0, 6), colspan=5)
        axBias[0].plot( data.wavelength_axis ,
                       data.week_clean_data_noise  , linewidth = 0.75, color=[0,0,0.7])
        axBias[0].plot( data.wavelength_axis ,
                       data.week_bias_data_airpls  , linewidth = 0.75, color=[0.7,0,0])
        axBias[0].legend(('Clean Data','Dirty Data'))
        axBias[1].plot( data.wavelength_axis ,
                       np.vstack( (
                           np.mean(np.abs(data.pca_week_Bias_airPLS_test["residual"]),axis=0), 
                           np.mean(np.abs(data.pca_feed_bias_test["residual"]),axis=0), 
                           np.mean(np.abs(data.pca_week_Rand_airPLS_test["residual"]),axis=0), 
                           np.mean(np.abs(data.pca_week_Bias_clean_test["residual"]),axis=0), 
                           np.mean(np.abs(data.pca_feed_bias_clean_test["residual"]),axis=0), 
                           np.mean(np.abs(data.pca_clean_rand_test["residual"]),axis=0), 
                           )).T , linewidth = 0.75)
        axBias[1].legend(('Dirty Bias:Week','Dirty Bias:Feed','Dirty Random','Clean Bias:Week','Clean Bias:Feed','Clean Unbiased'))
        axBias[0].set_ylabel("Intensity / Counts")
        axBias[0].set_xlabel("Raman Shift cm$^{-1}$")
        axBias[1].set_ylabel("Residual Intensity / Counts")
        axBias[1].set_xlabel("Raman Shift cm$^{-1}$")
        image_name = " Figure 13 Selection Residuals"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)

        figBias.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figBias.show()
        plt.close()
        print( 'Figure 13 generated at ' + full_path )        
        return