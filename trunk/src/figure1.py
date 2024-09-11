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
        isomer = np.empty(data.simulated_FA.shape[1])
        figData, axData = plt.subplots(1, 2, figsize=data.fig_settings.fig_Size)
        axData[0] = plt.subplot2grid((1, 11), (0, 0), colspan=5)
        axData[1] = plt.subplot2grid((1, 11), (0, 6), colspan=5)
        for i in range(data.simulated_FA.shape[1]):
            isomer[i] =  data.FA_properties["Isomer"][0][i]
            if   data.FA_properties["Isomer"][0][i]==3:
                axData[0].plot( data.wavelength_axis ,
                               data.simulated_FA[:,i]+8  , linewidth = 0.75, 
                               color=[0,0,0.7])
            elif   data.FA_properties["Isomer"][0][i]==2:
                axData[0].plot( data.wavelength_axis ,
                               data.simulated_FA[:,i]+4  , linewidth = 0.75, 
                               color=[0,(( data.FA_properties["Olefins"][0][i]+6)/9)*
                                      ((26- data.FA_properties["Carbons"][0][i])/12),0])
            elif   data.FA_properties["Isomer"][0][i]==1:
                axData[0].plot( data.wavelength_axis ,
                               data.simulated_FA[:,i]+12  , linewidth = 0.75, 
                               color=[(( data.FA_properties["Olefins"][0][i]+8)/14)*
                                      ((26- data.FA_properties["Carbons"][0][i])/12),0,0])
            else:
                axData[0].plot( data.wavelength_axis ,
                               data.simulated_FA[:,i]  , linewidth = 0.75, 
                               color=np.tile(1-data.FA_properties["Carbons"][0][i]/20,3))

        axData[0].set_ylabel("Intensity / Counts")
        axData[0].set_xlabel("Raman Shift cm$^{-1}$")
        axData[0].plot(1260,np.max(data.simulated_FA[59,isomer==1])+12.5,'*',
                       color=[0.8, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\delta$ H-C=$_c$',
            xy=(0.2, 0.04),
            xytext=(1260, np.max(data.simulated_FA[59,isomer==1])+13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0.8, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1299,np.max(data.simulated_FA[98,:])+12.5,'v',
                       color=[0, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\delta$ C-H$_2$',
            xy=(1305, 1),
            xytext=(1305, np.max(data.simulated_FA[98,:])+13),
            textcoords="data",
            xycoords="data",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1436,np.max(data.simulated_FA[235,isomer==1],axis=0)+12.5,
                       'v',color=[0, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\delta$C-H$_x$',
            xy=(0.2, 0.04),
            xytext=(1440, np.max(data.simulated_FA[235,isomer==1],axis=0)+13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1653,np.max(data.simulated_FA[452,isomer==1],axis=0)+12.5,
                       '*',color=[0.8, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\nu$C=C$_c$',
            xy=(0.2, 0.04),
            xytext=(1653, np.max(data.simulated_FA[452,isomer==1],axis=0)+13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0.8, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1673,np.max(data.simulated_FA[472,isomer==2],axis=0)+4.5,
                       'o',color=[0, 0.7, 0],markersize=5)
        axData[0].annotate(
            r'$\nu$C=C$_t$',
            xy=(0.2, 0.04),
            xytext=(1673, np.max(data.simulated_FA[472,isomer==2],axis=0)+5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0, 0.7, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1738,13.5,'+',color=[0, 0, 0.6],markersize=5)
        axData[0].annotate(
            r'$\nu$C=O',
            xy=(0.2, 0.04),
            xytext=(1738, 14),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0.6],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )

        axData[0].annotate(
            'saturated',
            xy=(0.2, 1),
            xytext=(1560,1 ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0],
            horizontalalignment="center",
        )
        axData[0].annotate(
            'trans-olefin',
            xy=(0.2, 0.04),
            xytext=(1560, 5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0, 0.8, 0],
            horizontalalignment="center",
        )

        axData[0].annotate(
            'conjugated',
            xy=(0.2, 0.04),
            xytext=(1560, 9),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0.7],
            horizontalalignment="center",
        )

        axData[0].annotate(
            'cis-olefin',
            xy=(0.2, 0.04),
            xytext=(1560, 13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size*0.75,
            color=[1, 0, 0],
            horizontalalignment="center",
        )
        FAnms = ['4 C','12 C','20 C']
        FAix = [0,4,21]
        for iN in range(len(FAnms)):
            c_col = axData[0].get_lines()[FAix[iN]].get_color()
            c_int = data.simulated_FA[240,FAix[iN]]
            axData[0].plot([1392,1435],np.tile(c_int,2),color=c_col)
            axData[0].annotate(
                FAnms[iN],
                xy=(0.2, 0.04),
                xytext=(1390, c_int),
                textcoords="data",
                xycoords="axes fraction",
                fontsize=data.pcaMC.fig_Text_Size*0.75,
                color = c_col,
                horizontalalignment="right",
                va="bottom",
            )


        speciescolor = [[0.3,0.9,0.9],[0.45,0.9,0.45],[0.4,0.4,0.85],
                        [0.8,0.4,0.4]]
        for i in range(data.adipose_data.shape[1]):
            axData[1].plot( data.wavelength_axis , data.adipose_data[:,i] ,
                           linewidth = 0.5,
                           color = speciescolor[data.adipose_species[i]])
        for i in range(data.data.shape[1]):
#            coSZSl = np.array([ 1 , 1 ,  1])*(data.week[i])/30
            axData[1].plot( data.wavelength_axis , data.data[:,i] , 
                           linewidth = 0.2,
                           color=np.tile(0.7/(data.feed[i]+1),(3)))
        axData[1].set_ylabel("Intensity / Counts")
        axData[1].set_xlabel("Raman Shift cm$^{-1}$")
        axData[0].annotate(
            "a)",
            xy=(0.2, 0.95),
            xytext=(0.2, 0.92),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axData[1].annotate(
            "b)",
            xy=(0.2, 0.95),
            xytext=(0.57, 0.92),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axData[1].annotate(
            "Butter 0mg",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.72),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=[0.7,0.7,0.7]
        )
        axData[1].annotate(
            "Butter 6mg",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.76),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=[0.1,0.1,0.1]
        )
        axData[1].annotate(
            "Lamb",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.80),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[0]
        )
        axData[1].annotate(
            "Chicken",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.88),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[3]
        )
        axData[1].annotate(
            "Pork",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.92),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[2]
        )
        axData[1].annotate(
            "Beef",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.84),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=data.pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[1]
        )
        image_name = " Figure 1 Simulated Spectra"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figData.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figData.show()
        plt.close()

        if data.fig_settings.plot_show:
            plt.show()
        plt.close()
        print( 'Figure 1 generated at ' + full_path )
        return