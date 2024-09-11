# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 06:57:54 2024

@author: rene
"""
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import os
from src.file_locations import images_folder

class fig( ):
    def __init__( self, data ):
# apply butter model to each alternative

        # Validation Figure
        figVal, axVal = plt.subplots(2, 2, figsize=data.fig_settings.fig_Size)
        axVal[0,0] = plt.subplot2grid((23, 11), (0, 0), colspan=5, rowspan=10)
        axVal[0,1] = plt.subplot2grid((23, 11), (0, 6), colspan=5, rowspan=10)
        axVal[1,0] = plt.subplot2grid((23, 11), (13, 0), colspan=5, rowspan=10)
        axVal[1,1] = plt.subplot2grid((23, 11), (13, 6), colspan=5, rowspan=10)
        figVal.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.09)

        # Generated GC profiles
        axVal[0,0].plot(np.arange(0,27)-0.2 , data.adipose_profile,'dg', markersize=2)
        axVal[0,0].plot(np.arange(0,27)-0.1 , data.butter_profileAdipose,'^c',alpha=0.5, markersize=2)
        axVal[0,0].plot(np.arange(0,27) , data.butter_profile_NoCov,'*b',alpha=0.5, markersize=2)
        axVal[0,0].plot(np.arange(0,27)+0.1 , data.butter_profile_Val, '.r',alpha=0.5, markersize=2)
        axVal[0,0].plot(np.arange(0,27)+0.2 , data.butter_profile,'+k',alpha=0.5, markersize=2)
        axVal[0,0].set_xlim(0,26)
        axVal[0,0].set_ylim(0,axVal[0,0].get_ylim()[1])
        axVal[0,0].set_xticks(np.arange(len(data.FAnames)))
        axVal[0,0].set_xticklabels(data.FAnames,fontsize=6, rotation=45)
        axVal[0,0].set_xlabel('Fatty Acid')
        axVal[0,0].set_ylabel('Molar %')

        # generated spectra
        trunc = np.concatenate([np.arange(29,130),np.arange(209,300),np.arange(429,490)])#,np.arange(519,570)])
        truncTicks = np.concatenate([np.arange(0,96,25),[95],np.arange(106,180,25),[186],np.arange(197,247,25),[246]])#,np.arange(257,304,25),[303]])
        tempIx = np.concatenate([np.arange(96,106),np.arange(187,197)])#,np.arange(247,257)])

        tempDat = np.mean(data.spectra_butter_Val[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'r')
        tempDat = np.mean(data.spectraNoCov[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'b',linestyle=(0,(1.1,1.3)))
        tempDat = np.mean(data.spectra_butter_adiposeCov[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'c',linestyle=(1.5,(1.3,1.7)))
        tempDat = np.mean(data.adipose_data[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'g')
        axVal[0,1].plot(tempDat*0,'k',linestyle=(0.5,(0.3,0.7)))


        tempDat = np.std(data.spectra_butter_Val[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'r')
        tempDat = np.std(data.spectraNoCov[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'b')
        tempDat = np.std(data.spectra_butter_adiposeCov[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'c')
        tempDat = np.std(data.adipose_data[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'g')
        axVal[0,1].plot(tempDat*0 - 100,'k',linestyle=(0.5,(0.3,0.7)))

        axVal[0,1].set_xticks(truncTicks)
        axVal[0,1].set_xticklabels(data.wavelength_axis[trunc[truncTicks]], rotation=45,fontsize='x-small')
        axVal[0,1].legend(['Val','NoCov','Adipose','AdiCov'],fontsize='x-small')
        axVal[0,1].set_ylim(axVal[0,1].get_ylim())
        axVal[0,1].plot([95,95],axVal[0,1].get_ylim(),'--k')
        axVal[0,1].plot([106,106],axVal[0,1].get_ylim(),'--k')
        axVal[0,1].plot([186,186],axVal[0,1].get_ylim(),'--k')
        axVal[0,1].plot([197,197],axVal[0,1].get_ylim(),'--k')
        axVal[0,1].set_xlabel('Raman Shift cm$^{-1}$')
        axVal[0,1].set_ylabel('Intensity (Counts)')
        axVal[0,1].annotate(
                    "$\mu$",
                    xy=(0.1, 0.9),
                    xytext=(0.025, 0.9),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
        axVal[0,1].annotate(
                    "$\sigma$",
                    xy=(0.1, 0.35),
                    xytext=(0.025, 0.25),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )

        axVal[1,0].plot(np.arange(1,data.pcaMC.N_PC+1),data.TestSet_Train["Q2"] , 'k')
        axVal[1,0].plot(np.arange(1,data.pcaMC.N_PC+1),data.TestSet_Val["Q2"],'r')
        axVal[1,0].plot(np.arange(1,data.pcaMC.N_PC+1),data.TestSet_NoCov["Q2"],'b')
        axVal[1,0].plot(np.arange(1,data.pcaMC.N_PC+1),data.TestSet_adiposeCov["Q2"],'c')
        axVal[1,0].plot(np.arange(1,data.pcaMC.N_PC+1),data.TestSet_adipose["Q2"],'g')
        axVal[1,0].legend(['Train','Val','NoCov','AdiCov','Adipose'])
        axVal[1,0].set_xlabel('PC rank')
        axVal[1,0].set_ylabel('Cumulative $Q^2$')

        BVsc=data.pcaGC.transform(np.transpose( data.butter_profile_Val))
        BNsc=data.pcaGC.transform(np.transpose( data.butter_profile_NoCov))
        ADsc = data.pcaGC.transform(np.transpose(data.adipose_profile))
        BAsc = data.pcaGC.transform(np.transpose(data.butter_profileAdipose))
        species_marker = ['*', '+', 'x', '1']
        axVal[1,1].plot(-data.GCsc[:, 0],-data.GCsc[:, 1],'.k',fillstyle='full')
        axVal[1,1].plot(-BVsc[:, 0],-BVsc[:, 1],'+r',fillstyle='none')
        axVal[1,1].plot(-BNsc[:, 0],-BNsc[:, 1],'<b',fillstyle='none')
        axVal[1,1].plot(-BAsc[:, 0],-BAsc[:, 1],'*c',fillstyle='none')
        for iS in range(4):
            ix = np.where(data.adipose_species==iS )[0]
            axVal[1,1].plot(-ADsc[ix, 0],-ADsc[ix, 1],species_marker[iS],color = 'g', fillstyle='none')
        #Calculate ellipse bounds and plot with scores
        theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
        circle = np.array((np.cos(theta), np.sin(theta)))
        sigma = np.cov(np.array((data.GCsc[:, 0], data.GCsc[:, 1])))
        ed = np.sqrt(stat.chi2.ppf(0.95, 2))
        ell = np.transpose(circle).dot(np.linalg.cholesky(sigma) * ed)
        a, b = np.max(ell[: ,0]), np.max(ell[: ,1]) #95% ellipse bounds
        t = np.linspace(0, 2 * np.pi, 100)
        axVal[1,1].plot(a * np.cos(t), b * np.sin(t), '--k', linewidth=0.2)
        xlims = axVal[1,1].get_xlim()
        axVal[1,1].plot([0,0], xlims, '-k', linewidth=0.2)
        axVal[1,1].plot( xlims,[0,0], '-k', linewidth=0.2)
        axVal[1,1].set_xlim(xlims)
        bbox = axVal[1,1].get_window_extent().transformed(
            figVal.dpi_scale_trans.inverted())
        axVal[1,1].set_ylim(np.array(xlims)*bbox.height/bbox.width + 2)

        axVal[1,1].set_xlabel('[t1]')
        axVal[1,1].set_ylabel('[t2]')

        subLabels = [["a)","b)"] , ["c)", "d)"]]
        for ax1 in range(np.shape(subLabels)[0]):
            for ax2 in range(np.shape(subLabels)[1]):
                axVal[ax1,ax2].annotate(
                    subLabels[ax1][ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.025, 1.025),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
        image_name = " Figure 14 Validation Sets"
        full_path = os.path.join( str(data.fig_settings.WD) , 
                                 str(images_folder), 
                                 data.fig_settings.fig_Project +
                                image_name + '.' + 
                                data.fig_settings.fig_Format)
        figVal.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        if data.fig_settings.plot_show:
            figVal.show()
        plt.close()
        print( 'Figure 14 generated at ' + full_path )
        
        figFApro, axFApro = plt.subplots(1, 1, figsize=data.fig_settings.fig_Size)
        axFApro.bar(np.arange(27),data.FA_profiles_sanity[0,:],facecolor='k')
        axFApro.bar(np.arange(27)-0.3,data.FA_profiles_sanity[1,:],facecolor='r',width=0.12)
        axFApro.bar(np.arange(27)-0.1,data.FA_profiles_sanity[2,:],facecolor='b',width=0.12)
        axFApro.bar(np.arange(27)+0.1,data.FA_profiles_sanity[3,:],facecolor='c',width=0.12)
        axFApro.bar(np.arange(27)+0.3,data.FA_profiles_sanity[4,:],facecolor='g',width=0.12)
        axFApro.set_xticks(np.arange(len(data.FAnames)))
        axFApro.set_xticklabels(data.FAnames,fontsize=6, rotation=45)
        axFApro.set_xlabel('Fatty Acid')
        axFApro.set_ylabel('Molar %')
        axFApro.legend(['Train','Val','NoCov','AdiCov','Adipose'],fontsize='x-small')

        image_name = " Supplementary Figure 2 FA profiles sanity checks."
        full_path = os.path.join(images_folder, data.fig_settings.fig_Project +
                                image_name + data.fig_settings.fig_Format)
        figFApro.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        plt.close()

        figFAproMC, axFAproMC = plt.subplots(1, 1, figsize=data.fig_settings.fig_Size)
        axFAproMC.bar(np.arange(27),data.FA_profiles_sanity[0,:]-data.FA_profiles_sanity[0,:],facecolor='k')
        axFAproMC.bar(np.arange(27)-0.3,data.FA_profiles_sanity[1,:]-data.FA_profiles_sanity[0,:],facecolor='r',width=0.12)
        axFAproMC.bar(np.arange(27)-0.1,data.FA_profiles_sanity[2,:]-data.FA_profiles_sanity[0,:],facecolor='b',width=0.12)
        axFAproMC.bar(np.arange(27)+0.1,data.FA_profiles_sanity[3,:]-data.FA_profiles_sanity[0,:],facecolor='c',width=0.12)
        axFAproMC.bar(np.arange(27)+0.3,data.FA_profiles_sanity[4,:]-data.FA_profiles_sanity[0,:],facecolor='g',width=0.12)
        axFAproMC.set_xticks(np.arange(len(data.FAnames)))
        axFAproMC.set_xticklabels(data.FAnames,fontsize=6, rotation=45)
        axFAproMC.set_xlabel('Fatty Acid')
        axFAproMC.set_ylabel('Molar %')
        axFAproMC.legend(['Train','Val','NoCov','Adipose','AdiCov'],fontsize='x-small')


        image_name = " Supplementary Figure 3 FA profile Sanity Check Difference"
        full_path = os.path.join(images_folder, data.fig_settings.fig_Project +
                                image_name + data.fig_settings.fig_Format)
        figFAproMC.savefig(full_path,
                         dpi=data.fig_settings.fig_Resolution)
        plt.close()

        return