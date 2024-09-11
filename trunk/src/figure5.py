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
        # crossover models
        figGCdata, axGCdata = plt.subplots(2, 2, figsize=( 
            data.fig_settings.fig_Size[0],data.fig_settings.fig_Size[0]))
        axGCdata[0,0] = plt.subplot2grid((15,15), (0, 0), colspan=6, rowspan=6)
        axGCdata[0,1] = plt.subplot2grid((15,15), (0, 8), colspan=6, rowspan=6)
        axGCdata[1,0] = plt.subplot2grid((15,15), (8, 0), colspan=6, rowspan=6)
        axGCdata[1,1] = plt.subplot2grid((15,15), (8, 8), colspan=6, rowspan=6)
        figGCdata.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        GCSpecLoad = np.inner(data.data,data.GCsc.T)
        GCSpecLoad = (GCSpecLoad/np.sum(GCSpecLoad**2,axis=0)**0.5)

        lOff = np.hstack((0,np.cumsum(np.floor(np.min(
            data.pcaMC.spectral_loading[0:2,:]-data.pcaMC.spectral_loading[1:3,:],
            axis=1)*20)/20)))
        axGCdata[0,0].plot(data.wavelength_axis,
                           data.pcaMC.spectral_loading[0:3,:].T+lOff,'k')
        axGCdata[0,0].plot(data.wavelength_axis,GCSpecLoad[:,0:3]*[-1,-1,1]+lOff,
                           '--',color=[0.5,0.5,0.5])
        axGCdata[0,0].plot([data.wavelength_axis[0],data.wavelength_axis[-1]],
                           np.tile(lOff,(2,1)),'.-',color=[0.8,0.8,0.8],lw=0.5)


        FAnames = ['']*data.FA_properties["Carbons"].shape[1]
        for i in range(len(FAnames)):
            if data.FA_properties[ "Isomer" ][ 0, i ]==3:
                FAnames[i] = 'CLA'
            else:
                if data.FA_properties[ "Isomer" ][ 0, i ]==0:
                    isoNm = ''
                elif data.FA_properties[ "Isomer" ][ 0, i ]==1:
                    isoNm = 'c'
                else:
                    isoNm = 't'
                FAnames[i] = ( str( int(data.FA_properties[ "Carbons" ][ 0, i ] )) 
                + ':' + str( int(data.FA_properties[ "Olefins" ][ 0, i ]) ) + isoNm )
            
        SpecGCLoad = np.inner(data.butter_profile,
                              data.pcaMC.component_weight[:,:11].T)
        SpecGCLoad = (SpecGCLoad/np.sum(SpecGCLoad**2,axis=0)**0.5)
        lSOff = np.hstack((0,np.cumsum(np.floor(np.min(SpecGCLoad[:,0:2]-
                                           SpecGCLoad[:,1:3],axis=0)*20)/20)))
        xval = np.empty(27)
        for i in range(data.N_FA):
            xval[i] =  ( data.FA_properties["Carbons"][0][i] +  
            data.FA_properties["Isomer"][0][i]/4 +  
            data.FA_properties["Olefins"][0][i]/40 )
        xIx = np.argsort(xval)
        axGCdata[0,1].plot(np.arange(len(xIx)),
                           [-1,-1,1]*data.pcaGC.components_[0:3, xIx].T+lSOff,'k')
        axGCdata[0,1].plot(np.arange(len(xIx)),SpecGCLoad[xIx,0:3]+lSOff,'--',
                           color=[0.6,0.6,0.6])
        axGCdata[0,1].plot(axGCdata[0,1].get_xlim(),np.tile(lSOff,(2,1)),'--',
                           color=[0.8,0.8,0.8])
        axGCdata[0,1].set_xticks(range(len(xIx)))
        axGCdata[0,1].set_xticklabels(labels=FAnames,rotation=90, fontsize=8)
        axGCdata[0,1].set_xlim([0,len(xIx)-1])



        axGCdata[1,0].plot(GCSpecLoad[:,2],data.pcaMC.spectral_loading[2,:],'<',
                           color=[0.85,0.85,0.85],markersize=6)
        axGCdata[1,0].plot(-GCSpecLoad[:,1],data.pcaMC.spectral_loading[1,:],'+',
                           color=[0.4,0.4,0.4],markersize=6)
        axGCdata[1,0].plot(-GCSpecLoad[:,0],data.pcaMC.spectral_loading[0,:],'.k',
                           markersize=6)
        xran = (axGCdata[1,0].get_xlim()[0]*0.95,axGCdata[1,0].get_xlim()[1]*0.95)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(GCSpecLoad[:,2].T,
                         data.pcaMC.spectral_loading[2,:],1))(xran),'--',
                           color=[0.85,0.85,0.85],lw=0.5)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(-GCSpecLoad[:,1].T,
                         data.pcaMC.spectral_loading[1,:],1))(xran),'--',
                           color=[0.4,0.4,0.4],lw=0.5)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(-GCSpecLoad[:,0].T,
                         data.pcaMC.spectral_loading[0,:],1))(xran),'--k',lw=0.5)

        axGCdata[1,1].plot(data.pcaGC.components_[2, :],SpecGCLoad[:,2],'<',
                           color=[0.85,0.85,0.85],markersize=6)
        axGCdata[1,1].plot(-data.pcaGC.components_[1, :],SpecGCLoad[:,1],'+',
                           color=[0.4,0.4,0.4],markersize=6)
        axGCdata[1,1].plot(-data.pcaGC.components_[0, :],SpecGCLoad[:,0],'.k',
                           markersize=6)
        xran = (axGCdata[1,1].get_xlim()[0]*0.95,axGCdata[1,1].get_xlim()[1]*0.95)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(
            data.pcaGC.components_[2,:],SpecGCLoad[:,2],1))(xran),'--',
            color=[0.85,0.85,0.85],lw=0.5)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(
            -data.pcaGC.components_[1,:],SpecGCLoad[:,1],1))(xran),'--',
            color=[0.4,0.4,0.4],lw=0.5)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(
            -data.pcaGC.components_[0,:],SpecGCLoad[:,0],1))(xran),'--k',lw=0.5)


        axGCdata[0,0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axGCdata[0,0].set_ylabel('W',labelpad=-1)
        axGCdata[0,0].set_yticks([])
        axGCdata[0,0].legend(['RS','_','_','GCX'],
                             framealpha=0.5,
                              borderpad=0.2)

        axGCdata[0,1].set_ylabel('W',labelpad=-1)
        axGCdata[0,1].set_yticks([])
        axGCdata[0,1].legend(['GC','_','_','RSX'],
                              fontsize=data.fig_settings.fig_Text_Size*0.65,
                              framealpha=0.5,
                              borderpad=0.2)

        axGCdata[1,0].set_xlabel('W spectral',labelpad=-1)
        axGCdata[1,0].set_ylabel('W GCX',labelpad=-1)
        axGCdata[1,0].legend(['#3','#2','#1'],
                              fontsize=data.fig_settings.fig_Text_Size*0.65,
                              framealpha=0.5,
                              borderpad=0.2)

        axGCdata[1,1].set_xlabel('W GC',labelpad=-1)
        axGCdata[1,1].set_ylabel('W SpectralX',labelpad=-1)
        axGCdata[1,1].legend(['#3','#2','#1'],
                              fontsize=data.fig_settings.fig_Text_Size*0.65,
                              framealpha=0.5,
                              borderpad=0.2)
        subLabels = [r"a)",
                     r"b)",
                     r"c)",
                     r"d)",
                     ]
        for ax1 in range(2):
            for ax2 in range(2):
                axGCdata[ax1,ax2].annotate(
                    subLabels[ax1*2+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.025, 1.025),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )


        image_name = " Figure 5 GC crossover plots"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figGCdata.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figGCdata.show()
        plt.close()
        print( 'Figure 5 generated at ' + full_path )
        return