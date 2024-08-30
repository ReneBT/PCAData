import numpy as np
#import scipy.io as sio
import h5py
import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from copulas.multivariate import GaussianMultivariate
from matplotlib import patches
#from matplotlib import transforms
from sklearn.decomposition import PCA
import src.airPLS as air # https://github.com/zmzhang/airPLS
#from matplotlib.colors import ListedColormap
#from matplotlib.patches import ConnectionPatch
import src.local_nipals as npls
from src.file_locations import data_folder,images_folder
# This expects to be called inside the jupyter project folder structure.


# Using pathlib we can handle different filesystems (mac, linux, windows) using a common syntax.
# file_path = data_folder / "raw_data.txt"
# More info on using pathlib:
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f


class graphicalPCA:
### ******      START CLASS      ******
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned
    # along the columns and each row corresponds to different variables

    # comments include references to relevant lines in the pseudocode listed in the paper

    def __init__(pcaMC):
### ***   START  Data Calculations   ***
### Read in data
# simulated fatty acid spectra and associated experimental concentration data
        mpl.rcParams['lines.markersize'] = 3

#        GC_data = sio.loadmat(
#           data_folder / "AllGC.mat", struct_as_record=False
#       )
        GC_data = h5py.File(data_folder / "AllGC.mat")
# Gas Chromatograph (GC) data is modelled based on Beattie et al. Lipids 2004 Vol 39 (9):897-906
# it is reconstructed with 4 underlying factors
#        simplified_fatty_acid_spectra = sio.loadmat(data_folder / "FA spectra.mat", struct_as_record=False)
        simplified_fatty_acid_spectra = h5py.File(data_folder / "FA spectra.mat")
# simplified_fatty_acid_spectrkma are simplified spectra of fatty acid methyl esters built from the properties described in
# Beattie et al. Lipids  2004 Vol 39 (5): 407-419
        wavelength_axis = simplified_fatty_acid_spectra["FAXcal"][:,]
        simulated_FA = simplified_fatty_acid_spectra["simFA"][:,:].T
        simulated_Collagen = simulated_FA[:,27]*1000
        simulated_Heme = simulated_FA[:,28]
        simulated_FA = simulated_FA[:,range(27)]
        FA_properties = simplified_fatty_acid_spectra["FAproperties"]
        
        min_spectral_values = np.tile(
            np.min(simulated_FA, axis=1), (np.shape(simulated_FA)[1], 1)
        )
# convert the mass base profile provided into a molar profile
        butter_profile = GC_data["ButterGC"][:,:] / FA_properties["MolarMass"][:,:].T
        butter_profile = 100.0 * butter_profile / sum(butter_profile)
        sam_codes = GC_data["sample_ID"][:,]
        Grouping = np.empty(sam_codes.shape[1],dtype=bool)
        feed = np.empty(np.size(Grouping))
        week = np.empty(np.size(Grouping))
        for iSam in range(sam_codes.shape[1]):
            Grouping[ iSam ] = sam_codes[ 0, iSam ]=='B'
            feed[ iSam ] = int( chr( sam_codes[ 1, iSam ] ) )*2
            week[ iSam ] = int( chr( sam_codes[ 2, iSam] ) + chr( sam_codes[ 3, iSam ] ) )

        adipose_profile = GC_data["AdiposeGC"] [:,:] / FA_properties["MolarMass"][:,:].T
        adipose_profile = 100.0 * adipose_profile / sum(adipose_profile)
        adipose_species = GC_data["AdiposeSpecies"][0].astype(int)
### generate simulated observational
# spectra for each sample by multiplying the simulated FA reference spectra by
# the Fatty Acid profiles. Note that the simplified_fatty_acid_spectra spectra
# have a standard intensity in the carbonyl mode peak (the peak with the
# highest pixel position)
        data = np.dot(simulated_FA, butter_profile)
        min_data = np.dot(
            np.transpose(min_spectral_values), butter_profile
        )  # will allow scaling of min_spectral_values to individual sample

        adipose_data = np.dot(simulated_FA, adipose_profile)

        N_FA = FA_properties["Carbons"].shape[1]

### calculate PCA
# with the full fatty acid covariance, comparing custom NIPALS and built in PCA function.

        pcaMC = npls.nipals(
            X_data=data,
            maximum_number_PCs=10,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pcaMC.calc_PCA()

### Plot Pure Simulated Data & Plot Simulated Observational Data
        isomer = np.empty(simulated_FA.shape[1])
        figData, axData = plt.subplots(1, 2, figsize=pcaMC.fig_Size)
        axData[0] = plt.subplot2grid((1, 11), (0, 0), colspan=5)
        axData[1] = plt.subplot2grid((1, 11), (0, 6), colspan=5)
        for i in range(simulated_FA.shape[1]):
            isomer[i] =  FA_properties["Isomer"][0][i]
            if   FA_properties["Isomer"][0][i]==3:
                axData[0].plot( wavelength_axis ,
                               simulated_FA[:,i]+8  , linewidth = 0.75, color=[0,0,0.7])
            elif   FA_properties["Isomer"][0][i]==2:
                axData[0].plot( wavelength_axis ,
                               simulated_FA[:,i]+4  , linewidth = 0.75, color=[0,(( FA_properties["Olefins"][0][i]+6)/9)*((26- FA_properties["Carbons"][0][i])/12),0])
            elif   FA_properties["Isomer"][0][i]==1:
                axData[0].plot( wavelength_axis ,
                               simulated_FA[:,i]+12  , linewidth = 0.75, color=[(( FA_properties["Olefins"][0][i]+8)/14)*((26- FA_properties["Carbons"][0][i])/12),0,0])
            else:
                axData[0].plot( wavelength_axis ,
                               simulated_FA[:,i]  , linewidth = 0.75, color=np.tile(1-FA_properties["Carbons"][0][i]/20,3))

        axData[0].set_ylabel("Intensity / Counts")
        axData[0].set_xlabel("Raman Shift cm$^{-1}$")
        axData[0].plot(1260,np.max(simulated_FA[59,isomer==1])+12.5,'*',color=[0.8, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\delta$ H-C=$_c$',
            xy=(0.2, 0.04),
            xytext=(1260, np.max(simulated_FA[59,isomer==1])+13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0.8, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1299,np.max(simulated_FA[98,:])+12.5,'v',color=[0, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\delta$ C-H$_2$',
            xy=(1305, 1),
            xytext=(1305, np.max(simulated_FA[98,:])+13),
            textcoords="data",
            xycoords="data",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1436,np.max(simulated_FA[235,isomer==1],axis=0)+12.5,'v',color=[0, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\delta$C-H$_x$',
            xy=(0.2, 0.04),
            xytext=(1440, np.max(simulated_FA[235,isomer==1],axis=0)+13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1653,np.max(simulated_FA[452,isomer==1],axis=0)+12.5,'*',color=[0.8, 0, 0],markersize=5)
        axData[0].annotate(
            r'$\nu$C=C$_c$',
            xy=(0.2, 0.04),
            xytext=(1653, np.max(simulated_FA[452,isomer==1],axis=0)+13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0.8, 0, 0],
            horizontalalignment="center",
            rotation=90,
            va="bottom",
        )
        axData[0].plot(1673,np.max(simulated_FA[472,isomer==2],axis=0)+4.5,'o',color=[0, 0.7, 0],markersize=5)
        axData[0].annotate(
            r'$\nu$C=C$_t$',
            xy=(0.2, 0.04),
            xytext=(1673, np.max(simulated_FA[472,isomer==2],axis=0)+5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
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
            fontsize=pcaMC.fig_Text_Size*0.75,
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
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0],
            horizontalalignment="center",
        )
        axData[0].annotate(
            'trans-olefin',
            xy=(0.2, 0.04),
            xytext=(1560, 5),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0, 0.8, 0],
            horizontalalignment="center",
        )

        axData[0].annotate(
            'conjugated',
            xy=(0.2, 0.04),
            xytext=(1560, 9),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[0, 0, 0.7],
            horizontalalignment="center",
        )

        axData[0].annotate(
            'cis-olefin',
            xy=(0.2, 0.04),
            xytext=(1560, 13),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color=[1, 0, 0],
            horizontalalignment="center",
        )
        FAnms = ['4 C','12 C','20 C']
        FAix = [0,4,21]
        for iN in range(len(FAnms)):
            c_col = axData[0].get_lines()[FAix[iN]].get_color()
            c_int = simulated_FA[240,FAix[iN]]
            axData[0].plot([1392,1435],np.tile(c_int,2),color=c_col)
            axData[0].annotate(
                FAnms[iN],
                xy=(0.2, 0.04),
                xytext=(1390, c_int),
                textcoords="data",
                xycoords="axes fraction",
                fontsize=pcaMC.fig_Text_Size*0.75,
                color = c_col,
                horizontalalignment="right",
                va="bottom",
            )


        speciescolor = [[0.3,0.9,0.9],[0.45,0.9,0.45],[0.4,0.4,0.85],
                        [0.8,0.4,0.4]]
        for i in range(adipose_data.shape[1]):
            axData[1].plot( wavelength_axis , adipose_data[:,i] ,
                           linewidth = 0.5,
                           color = speciescolor[adipose_species[i]])
        for i in range(data.shape[1]):
            col = np.array([ 1 , 1 ,  1])*(week[i])/30
            axData[1].plot( wavelength_axis , data[:,i] , linewidth = 0.2,
                           color=np.tile(0.7/(feed[i]+1),(3)))
        axData[1].set_ylabel("Intensity / Counts")
        axData[1].set_xlabel("Raman Shift cm$^{-1}$")
        axData[0].annotate(
            "a)",
            xy=(0.2, 0.95),
            xytext=(0.2, 0.92),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axData[1].annotate(
            "b)",
            xy=(0.2, 0.95),
            xytext=(0.57, 0.92),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="center",
        )
        axData[1].annotate(
            "Butter 0mg",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.72),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=[0.7,0.7,0.7]
        )
        axData[1].annotate(
            "Butter 6mg",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.76),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=[0.1,0.1,0.1]
        )
        axData[1].annotate(
            "Lamb",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.80),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[0]
        )
        axData[1].annotate(
            "Chicken",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.88),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[3]
        )
        axData[1].annotate(
            "Pork",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.92),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[2]
        )
        axData[1].annotate(
            "Beef",
            xy=(0.2, 0.95),
            xytext=(0.65, 0.84),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
            color=speciescolor[1]
        )
        image_name = " Simulated Spectra."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)        # plt.show()
        plt.close()


### ***   END  Data Calculations   ***

### ***   Start  Interpretation   ***
### Generate Noisy Data
        shot_noise = np.random.randn(np.shape(data)[0],np.shape(data)[1])/10
        majpk = np.where(np.mean(data,axis=1)>np.mean(data))
        signal = np.mean(data[majpk,:],axis=1)
        data_q_noise = data + ((data**0.5 + 10) * shot_noise / 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in quarter scaled noise: ' +
              str(np.mean(signal/np.std((data[majpk,:]**0.5 + 10)
                                        * shot_noise[majpk,:] / 4,axis=1))))
        data_1_noise = data + ((data**0.5 + 10) * shot_noise) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in unscaled noise: ' +
              str(np.mean(signal/np.std((data[majpk,:]**0.5 + 10)
                                        * shot_noise[majpk,:],axis=1))))
        data_4_noise = data + ((data**0.5 + 10) * shot_noise * 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in 4 times scaled noise: ' +
              str(np.mean(signal/np.std((data[majpk,:]**0.5 + 10)
                                        * shot_noise[majpk,:]*4,axis=1))))
        pca_q_noise = npls.nipals(
            X_data=data_q_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pca_q_noise.calc_PCA()
        pca_1_noise = npls.nipals(
            X_data=data_1_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pca_1_noise.calc_PCA()
        pca_4_noise = npls.nipals(
            X_data=data_4_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pca_4_noise.calc_PCA()

        pca_noise = npls.nipals(
            X_data=((data**0.5 + 10) * shot_noise),
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pca_noise.calc_PCA()# noise from SNR 100

        corrPCs_noiseq = np.inner(pca_q_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise1 = np.inner(pca_1_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise4 = np.inner(pca_4_noise.spectral_loading,pcaMC.spectral_loading)
        corrPCs_noise = np.inner(pca_noise.spectral_loading,pcaMC.spectral_loading)
        #loadings already standardised to unit norm
        corrPCs_noiseq_R2sum_ax0 = np.sum(corrPCs_noiseq**2,axis=0)#total variance shared between each noiseq loading and the noisless loadings
        corrPCs_noiseq_R2sum_ax1 = np.sum(corrPCs_noiseq**2,axis=1)#total variance shared between each noiseless PC and the noiseq loadings

        maxCorr_noiseq = np.max(np.abs(corrPCs_noiseq),axis=0)
        maxCorr_noise1 = np.max(np.abs(corrPCs_noise1),axis=0)
        maxCorr_noise4 = np.max(np.abs(corrPCs_noise4),axis=0)
        maxCorr_noise = np.max(np.abs(corrPCs_noise),axis=0)
        maxCorrMean = ((np.sum(maxCorr_noise**2)**0.5)/
                maxCorr_noise.shape[0]**0.5) #correlation measures noise so propagate as variance
        print('Mean optimal Correlation : ' + str(maxCorrMean))
        print('SE Correlation : ' + str(maxCorrMean + [-np.std(maxCorr_noise),np.std(maxCorr_noise)]))

        max_Ixq = np.empty(*maxCorr_noiseq.shape)
        max_Ix1 = np.copy(max_Ixq)
        max_Ix4 = np.copy(max_Ixq)
        max_IxN = np.copy(max_Ixq)
        max_Snq = np.copy(max_Ixq)
        max_Sn1 = np.copy(max_Ixq)
        max_Sn4 = np.copy(max_Ixq)
        max_SnN = np.copy(max_Ixq)

        for iCol in range(np.shape(maxCorr_noiseq)[0]):
            max_Ixq[iCol] = np.where(np.abs(corrPCs_noiseq[:,iCol])==maxCorr_noiseq[iCol])[0]
            max_Snq[iCol] = np.sign(corrPCs_noiseq[max_Ixq[iCol].astype(int),iCol])
            max_Ix1[iCol] = np.where(np.abs(corrPCs_noise1[:,iCol])==maxCorr_noise1[iCol])[0]
            max_Sn1[iCol] = np.sign(corrPCs_noise1[max_Ix1[iCol].astype(int),iCol])
            max_Ix4[iCol] = np.where(np.abs(corrPCs_noise4[:,iCol])==maxCorr_noise4[iCol])[0]
            max_Sn4[iCol] = np.sign(corrPCs_noise4[max_Ix4[iCol].astype(int),iCol])
            max_IxN[iCol] = np.where(np.abs(corrPCs_noise[:,iCol])==maxCorr_noise[iCol])[0]
            max_SnN[iCol] = np.sign(corrPCs_noise[max_IxN[iCol].astype(int),iCol])

### Scree Plot
        figScree, axScree = plt.subplots(2, 3, figsize=pcaMC.fig_Size)
        figScree.subplots_adjust(wspace = 0.35)
        axScree[0,0] = plt.subplot2grid((9,11),(0,0),colspan=3, rowspan=4)
        axScree[0,1] = plt.subplot2grid((9,11),(0,4),colspan=3, rowspan=4)
        axScree[0,2] = plt.subplot2grid((9,11),(0,8),colspan=3, rowspan=4)
        axScree[1,0] = plt.subplot2grid((9,11),(5,0),colspan=11, rowspan=4)

        axScree[0,0].plot(range(1,11), pcaMC.Eigenvalue[:10]**0.5, "k")
        axScree[0,0].plot(range(1,11), pca_q_noise.Eigenvalue[:10]**0.5, "b")
        axScree[0,0].plot(range(1,11), pca_1_noise.Eigenvalue[:10]**0.5, "r")
        axScree[0,0].plot(range(1,11), pca_4_noise.Eigenvalue[:10]**0.5, "g")
        axScree[0,0].plot(range(1,11), 0.25*pca_noise.Eigenvalue[:10]**0.5, "--",color=[0.5, 0.5, 0.5])
        axScree[0,0].plot(range(1,11), 0.25*pca_noise.Eigenvalue[:10]**0.5, "--",color=[0.5, 0.5, 1])
        axScree[0,0].plot(range(1,11), pca_noise.Eigenvalue[:10]**0.5, "--",color=[1, 0.5, 0.5])
        axScree[0,0].plot(range(1,11), 4*pca_noise.Eigenvalue[:10]**0.5, "--",color=[0.5, 1, 0.5])
        axScree[0,0].set_ylabel("Eigenvalue", labelpad=0)
        axScree[0,0].set_xlabel("PC rank", labelpad=0)
        axScree[0,0].legend(("$SNR_\infty$","$SNR_{400}$","$SNR_{100}$","$SNR_{25}$","Noise"),fontsize='small')

        axScree[0,1].plot(range(1,11), np.cumsum(100*(pcaMC.Eigenvalue[:10]**0.5)/sum(pcaMC.Eigenvalue**0.5)), "k")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_q_noise.Eigenvalue[:10]**0.5)/sum(pca_q_noise.Eigenvalue**0.5)), "b")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_1_noise.Eigenvalue[:10]**0.5)/sum(pca_1_noise.Eigenvalue**0.5)), "r")
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_4_noise.Eigenvalue[:10]**0.5)/sum(pca_4_noise.Eigenvalue**0.5)), "g")
#        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_noise.Eigenvalue[:10]**0.5)/sum(pca_4_noise.Eigenvalue**0.5)), "--",color=[0.45,0.45,0.45])
        axScree[0,1].plot(range(1,11), np.cumsum(100*(pca_noise.Eigenvalue[:10]**0.5)/sum(pca_noise.Eigenvalue**0.5)), "--",color=[0.75,0.75,0.75])
        axScree[0,1].set_ylabel("% Variance explained", labelpad=0)
        axScree[0,1].set_xlabel("PC rank", labelpad=0)

        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noise4)[:10]), color = (0, 0.75, 0))
        axScree[0,2].plot(range(1,11), maxCorr_noise4[:10], color = (0.25, 1, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noise1)[:10]), color = (0.75, 0, 0))
        axScree[0,2].plot(range(1,11), maxCorr_noise1[:10], color = ( 1 , 0.25, 0.5), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noiseq)[:10]), color = (0, 0, 0.75))
        axScree[0,2].plot(range(1,11), maxCorr_noiseq[:10], color = (0.25, 0.5, 1), linewidth=0.75)
        axScree[0,2].plot(range(1,11), np.abs(np.diagonal(corrPCs_noise)[:10]), color = (0.45, 0.45, 0.45))
        axScree[0,2].plot(range(1,11), maxCorr_noise[:10], color = (0.75, 0.75, 0.75), linewidth=0.75)
        axScree[0,2].set_ylabel("Correlation vs noiseless", labelpad=0)
        axScree[0,2].set_xlabel("PC rank", labelpad=0)

        axScree[1,0].plot(pca_noise.pixel_axis, pca_noise.spectral_loading[:3].T-[0,0.3,0.6],color = (0.75, 0.75, 0.75),linewidth=2)
        axScree[1,0].plot(pca_4_noise.pixel_axis, pca_4_noise.spectral_loading[:3].T-[0,0.3,0.6],color = (0, 1, 0),linewidth=1.75)
        axScree[1,0].plot(pca_1_noise.pixel_axis, pca_1_noise.spectral_loading[:3].T-[0,0.3,0.6],color = (1, 0.15, 0.15),linewidth=1.5)
        axScree[1,0].plot(pca_q_noise.pixel_axis, pca_q_noise.spectral_loading[:3].T-[0,0.3,0.6],color = (0.3, 0.3, 1),linewidth=1.25)
        axScree[1,0].plot(pcaMC.pixel_axis, pcaMC.spectral_loading[:3].T-[0,0.3,0.6],color = (0, 0, 0),linewidth = 1)
        axScree[1,0].set_ylabel("Weight")
        axScree[1,0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axScree[1,0].set_yticklabels([])
        axScree[1,0].annotate("",
                xy=(pca_noise.pixel_axis[100], pca_noise.spectral_loading[0,100]),
                xytext=(pca_noise.pixel_axis[180], pca_noise.spectral_loading[0,100]),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                arrowprops=dict( lw=0.1, color='c',headwidth=5)
            )
        axScree[1,0].annotate("",
                xy=(pca_noise.pixel_axis[237], pca_noise.spectral_loading[0,237]),
                xytext=(pca_noise.pixel_axis[317], pca_noise.spectral_loading[0,237]),
                textcoords="data",
                xycoords="data",
                fontsize=pcaMC.fig_Text_Size,
                horizontalalignment="center", va="center",
                arrowprops=dict( lw=0.1, color='c',headwidth=5)
            )
        for subsub in np.arange(3):
            axScree[1,0].annotate(
                str(subsub+1),
                xy=(pca_noise.pixel_axis[0]*0.98,-0.3*subsub),
                xytext=(pca_noise.pixel_axis[0]*0.98,-0.3*subsub),
                xycoords="data",
                textcoords="data",
                fontsize=pcaMC.fig_Text_Size*0.75,
                horizontalalignment="left",
                va="center",
                alpha = 0.75,
                bbox=dict(boxstyle='square,pad=0', fc='w', ec='none')
            )
        subLabels = [r"a) Scree Plot", r"b) Cumulative Variance",
                     r"c) PC Correlations", r"d) Loadings", r"skip", r"e) Reconstructed Data"]
        for ax1 in range(2):
            for ax2 in range(3):
                if (ax1==1 and ax2>1)==False:
                    axScree[ax1,ax2].annotate(
                        subLabels[ax1*3+ax2],
                        xy=(0.18, 0.89),
                        xytext=(0, 1.02),
                        textcoords="axes fraction",
                        xycoords="axes fraction",
                        fontsize=pcaMC.fig_Text_Size*0.75,
                        horizontalalignment="left",
                    )
        image_name = " Comparing PCAs for different levels of noise"
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + '.' + pcaMC.fig_Format)
        plt.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()

        # 1 create 3 noise variant replicates of one sample and plot overlaid
        # 2 reconstruct replicates from noiseless PCA and plot overlaid, offset from 1
        # 3 subtract noiseless original spectrum then plot residuals overlaid, offset from 2. Scale up if necessary to compare, annotate with scaling used
        reps_4_noise = np.tile(data[:,23],(data.shape[1],1)).T
        reps_4_noise = reps_4_noise + ((reps_4_noise**0.5 + 10) * shot_noise * 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        reps_4_noise_recon = pcaMC.reduced_Rank_Reconstruction( reps_4_noise , 10 )

        figRecon, axRecon = plt.subplots(1, 1, figsize=pcaMC.fig_Size)
        figRecon.subplots_adjust(wspace = 0.35)
        axRecon = plt.subplot2grid((1,1),(0,0),colspan=1, rowspan=1)
        axRecon.plot(pca_noise.pixel_axis, reps_4_noise)
        offset = np.mean(reps_4_noise[:])*2
        ypos = np.array([0, -offset, -1.75*offset, -2.75*offset, -4*offset])
        axRecon.plot(pca_noise.pixel_axis, reps_4_noise_recon+ypos[1])
        axRecon.plot(pca_noise.pixel_axis, (reps_4_noise.T-data[:,23]).T+ypos[2])
        axRecon.plot(pca_noise.pixel_axis, (reps_4_noise_recon.T-data[:,23]).T+ypos[3])
        axRecon.plot(pca_noise.pixel_axis, 10*(reps_4_noise_recon.T-data[:,23]).T+ypos[4])
        axRecon.set_ylabel("Intensity")
        axRecon.set_yticklabels([])
        axRecon.set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)

        subsubStr2 = ["Noisy Repetitions", "Reconstructions", "Noise in Repetitions", "Reconstruction Error","Reconstruction Error x10"]
        ypos = ypos+50
        for subsub in np.arange(5):
            axRecon.annotate(
                subsubStr2[subsub],
                xy=(pca_noise.pixel_axis[365],ypos[subsub]),
                xytext=(pca_noise.pixel_axis[365],ypos[subsub]),
                xycoords="data",
                textcoords="data",
                fontsize=pcaMC.fig_Text_Size*0.75,
                horizontalalignment="center",
                va="center",
                alpha = 0.75,
                bbox=dict(boxstyle='square,pad=0', fc='w', ec='none')
            )

        image_name = " PCA reconstuction for noisy data"
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + '.' + pcaMC.fig_Format)
        plt.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)

        plt.close()

### Model Correlation Plot
        figNoiseCorr, axNoiseCorr = plt.subplots(2, 3, figsize=pcaMC.fig_Size)
        axNoiseCorr[0,0] = plt.subplot2grid((2,11),(0,0),colspan=3, rowspan=1)
        axNoiseCorr[0,1] = plt.subplot2grid((2,11),(0,4),colspan=3, rowspan=1)
        axNoiseCorr[0,2] = plt.subplot2grid((2,11),(0,8),colspan=3, rowspan=1)
        axNoiseCorr[1,0] = plt.subplot2grid((2,11),(1,0),colspan=3, rowspan=1)
        axNoiseCorr[1,1] = plt.subplot2grid((2,11),(1,4),colspan=3, rowspan=1)
        axNoiseCorr[1,2] = plt.subplot2grid((2,11),(1,8),colspan=3, rowspan=1)
        figNoiseCorr.subplots_adjust(left=0.07, right=0.99, top=0.95, wspace = 0.1, hspace=0.4)

        axNoiseCorr[0,0].plot(range(1,pcaMC.N_PC+1), corrPCs_noiseq[:,range(7)][range(pcaMC.N_PC),:]**2,)
        axNoiseCorr[0,1].plot(range(1,pcaMC.N_PC+1), corrPCs_noiseq[range(0,7),:][:,range(pcaMC.N_PC)].T**2,)
        axNoiseCorr[0,2].plot(range(1,pcaMC.N_PC+1),corrPCs_noiseq_R2sum_ax1[range(pcaMC.N_PC)],linewidth=2)
        axNoiseCorr[0,2].plot(range(1,pcaMC.N_PC+1),corrPCs_noiseq_R2sum_ax0[range(pcaMC.N_PC)],linewidth=1.5)
        axNoiseCorr[0,2].plot(range(1,pcaMC.N_PC+1),maxCorr_noiseq[range(pcaMC.N_PC)]**2,linewidth=1)
        axNoiseCorr[0,0].set_ylabel("R$^2$")
        axNoiseCorr[0,0].set_xlabel("PC in SNR$_{400}$")
        axNoiseCorr[0,0].set_xticks(range(1,pcaMC.N_PC,2))
        axNoiseCorr[0,1].set_ylabel("R$^2$")
        axNoiseCorr[0,1].set_xlabel("PC in SNR$_\infty$")
        axNoiseCorr[0,1].set_xticks(range(1,pcaMC.N_PC,2))
        axNoiseCorr[0,2].set_ylabel("Total R$^2$")
        axNoiseCorr[0,2].set_xlabel("PC rank")
        axNoiseCorr[0,2].set_xticks(range(1,pcaMC.N_PC,2))
        axNoiseCorr[0,0].legend(range(1,7),fontsize="small",loc=(0.7,0.25))
        axNoiseCorr[0,0].annotate(
            "PC in SNR$_\infty$",
            xy=(8.5,0.97),
            xytext=(8.5,0.99),
            xycoords="data",
            textcoords="data",
            fontsize=pcaMC.fig_Text_Size*0.75,
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
            fontsize=pcaMC.fig_Text_Size*0.75,
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
        axNoiseCorr[1,0].imshow( corrPCs_noiseq[:10,:10],vmin=0,vmax=1)
        axNoiseCorr[1,1].imshow( corrPCs_noise1[:10,:10],vmin=0,vmax=1)
        axNoiseCorr[1,2].imshow( corrPCs_noise4[:10,:10],vmin=0,vmax=1)
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
        image_name = " PC correlations noisy vs noiseless "
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + '.' + pcaMC.fig_Format)
        plt.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()

### Spectra vs GC Variation
        pcaGC = PCA( n_components=11 )  # It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract.
        pcaGC.fit(np.transpose(butter_profile))
        GCsc=pcaGC.transform(np.transpose(butter_profile))

        figGCscores, axGCscores = plt.subplots(1, 3, figsize=(pcaMC.fig_Size[0],pcaMC.fig_Size[0]/3))
        axGCscores[0] = plt.subplot2grid((1,22), (0, 0), colspan=6 )
        axGCscores[1] = plt.subplot2grid((1,22), (0, 8), colspan=6 )
        axGCscores[2] = plt.subplot2grid((1,22), (0, 16), colspan=6 )
        figGCscores.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.18)
        grps = [[0,0,2,2,4,4,6,6],[False,True,False,True,False,True,False,True],['db','db','or','or','^c','^c','sm','sm']]
        for iS in range(8):
            if grps[1][iS]:
                cGrp = Grouping
                fillS = 'full'
            else:
                cGrp = ~Grouping
                fillS = 'none'
            ix = np.where(np.logical_and(feed==grps[0][iS] , cGrp))[0]
            axGCscores[0].plot(pcaMC.component_weight[ix, 0],pcaMC.component_weight[ix, 1],grps[2][iS],fillstyle=fillS,markersize=6)
            axGCscores[1].plot(-GCsc[ix, 0],-GCsc[ix, 1],grps[2][iS],fillstyle=fillS,markersize=6)
        axGCscores[2].plot(GCsc[:, 2]/np.ptp(GCsc[:, 2]),pcaMC.component_weight[:, 2]/np.ptp(pcaMC.component_weight[:, 2]),'<',color=[0.85,0.85,0.85],markersize=6)
        axGCscores[2].plot(-GCsc[:, 1]/np.ptp(GCsc[:, 1]),pcaMC.component_weight[:, 1]/np.ptp(pcaMC.component_weight[:, 1]),'+',color=[0.4,0.4,0.4],markersize=6)
        axGCscores[2].plot(-GCsc[:, 0]/np.ptp(GCsc[:, 0]),pcaMC.component_weight[:, 0]/np.ptp(pcaMC.component_weight[:, 0]),'.k',markersize=6)

        axGCscores[0].set_xlabel('t[1]Spectral',labelpad=-1)
        axGCscores[0].set_ylabel('t[2]Spectral',labelpad=-1)
        axGCscores[0].legend(['0mg E','0mg L','2mg','_','4mg','_','6mg'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)
        axGCscores[2].legend(['#3','#2','#1'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)
        axGCscores[1].set_xlabel('t[1]GC',labelpad=-1)
        axGCscores[1].set_ylabel('t[2]GC',labelpad=-1)
        axGCscores[2].set_xlabel('t[#]GC',labelpad=-1)
        axGCscores[2].set_ylabel('t[#]Spectral',labelpad=-1)
        subLabels = ["a)","b)", "c)", "d)"]
        for ax1 in range(3):
            axGCscores[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0.025, 1.025),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="left",
            )

        image_name = " Spectral vs GC score plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figGCscores.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()


        figGCdata, axGCdata = plt.subplots(2, 2, figsize=(pcaMC.fig_Size[0],pcaMC.fig_Size[0]))
        axGCdata[0,0] = plt.subplot2grid((15,15), (0, 0), colspan=6, rowspan=6)
        axGCdata[0,1] = plt.subplot2grid((15,15), (0, 8), colspan=6, rowspan=6)
        axGCdata[1,0] = plt.subplot2grid((15,15), (8, 0), colspan=6, rowspan=6)
        axGCdata[1,1] = plt.subplot2grid((15,15), (8, 8), colspan=6, rowspan=6)
        figGCdata.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        GCSpecLoad = np.inner(data,GCsc.T)
        GCSpecLoad = (GCSpecLoad/np.sum(GCSpecLoad**2,axis=0)**0.5)

        lOff = np.hstack((0,np.cumsum(np.floor(np.min(pcaMC.spectral_loading[0:2,:]-pcaMC.spectral_loading[1:3,:],axis=1)*20)/20)))
        axGCdata[0,0].plot(wavelength_axis,pcaMC.spectral_loading[0:3,:].T+lOff,'k')
        axGCdata[0,0].plot(wavelength_axis,GCSpecLoad[:,0:3]*[-1,-1,1]+lOff,'--',color=[0.5,0.5,0.5])
        axGCdata[0,0].plot([wavelength_axis[0],wavelength_axis[-1]],np.tile(lOff,(2,1)),'.-',color=[0.8,0.8,0.8],lw=0.5)

        # crossover models
        lGOff = np.hstack((0,np.cumsum(np.floor(np.min(pcaMC.spectral_loading[0:2,:]-pcaMC.spectral_loading[1:3,:],axis=1)*20)/20)))
        FAnames = ['']*FA_properties["Carbons"].shape[1]
        for i in range(len(FAnames)):
            if FA_properties[ "Isomer" ][ 0, i ]==3:
                FAnames[i] = 'CLA'
            else:
                if FA_properties[ "Isomer" ][ 0, i ]==0:
                    isoNm = ''
                elif FA_properties[ "Isomer" ][ 0, i ]==1:
                    isoNm = 'c'
                else:
                    isoNm = 't'
                FAnames[i] = str( int(FA_properties[ "Carbons" ][ 0, i ] )) + ':' + str( int(FA_properties[ "Olefins" ][ 0, i ]) ) + isoNm
            
        SpecGCLoad = np.inner(butter_profile,pcaMC.component_weight[:,:11].T)
        SpecGCLoad = (SpecGCLoad/np.sum(SpecGCLoad**2,axis=0)**0.5)
        lSOff = np.hstack((0,np.cumsum(np.floor(np.min(SpecGCLoad[:,0:2]-SpecGCLoad[:,1:3],axis=0)*20)/20)))
        xval = np.empty(27)
        for i in range(N_FA):
            xval[i] =  FA_properties["Carbons"][0][i] +  FA_properties["Isomer"][0][i]/4 +  FA_properties["Olefins"][0][i]/40
        xIx = np.argsort(xval)
        axGCdata[0,1].plot(np.arange(len(xIx)),[-1,-1,1]*pcaGC.components_[0:3, xIx].T+lSOff,'k')
        axGCdata[0,1].plot(np.arange(len(xIx)),SpecGCLoad[xIx,0:3]+lSOff,'--',color=[0.6,0.6,0.6])
        axGCdata[0,1].plot(axGCdata[0,1].get_xlim(),np.tile(lSOff,(2,1)),'--',color=[0.8,0.8,0.8])
        axGCdata[0,1].set_xticks(range(len(xIx)))
        axGCdata[0,1].set_xticklabels(labels=FAnames,rotation=90, fontsize=8)
        axGCdata[0,1].set_xlim([0,len(xIx)-1])



        axGCdata[1,0].plot(GCSpecLoad[:,2],pcaMC.spectral_loading[2,:],'<',color=[0.85,0.85,0.85],markersize=6)
        axGCdata[1,0].plot(-GCSpecLoad[:,1],pcaMC.spectral_loading[1,:],'+',color=[0.4,0.4,0.4],markersize=6)
        axGCdata[1,0].plot(-GCSpecLoad[:,0],pcaMC.spectral_loading[0,:],'.k',markersize=6)
        xran = (axGCdata[1,0].get_xlim()[0]*0.95,axGCdata[1,0].get_xlim()[1]*0.95)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(GCSpecLoad[:,2].T,pcaMC.spectral_loading[2,:],1))(xran),'--',color=[0.85,0.85,0.85],lw=0.5)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(-GCSpecLoad[:,1].T,pcaMC.spectral_loading[1,:],1))(xran),'--',color=[0.4,0.4,0.4],lw=0.5)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(-GCSpecLoad[:,0].T,pcaMC.spectral_loading[0,:],1))(xran),'--k',lw=0.5)

        axGCdata[1,1].plot(pcaGC.components_[2, :],SpecGCLoad[:,2],'<',color=[0.85,0.85,0.85],markersize=6)
        axGCdata[1,1].plot(-pcaGC.components_[1, :],SpecGCLoad[:,1],'+',color=[0.4,0.4,0.4],markersize=6)
        axGCdata[1,1].plot(-pcaGC.components_[0, :],SpecGCLoad[:,0],'.k',markersize=6)
        xran = (axGCdata[1,1].get_xlim()[0]*0.95,axGCdata[1,1].get_xlim()[1]*0.95)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(pcaGC.components_[2,:],SpecGCLoad[:,2],1))(xran),'--',color=[0.85,0.85,0.85],lw=0.5)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(-pcaGC.components_[1,:],SpecGCLoad[:,1],1))(xran),'--',color=[0.4,0.4,0.4],lw=0.5)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(-pcaGC.components_[0,:],SpecGCLoad[:,0],1))(xran),'--k',lw=0.5)


        axGCdata[0,0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axGCdata[0,0].set_ylabel('W',labelpad=-1)
        axGCdata[0,0].set_yticks([])
        axGCdata[0,0].legend(['RS','_','_','GCX'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)

        axGCdata[0,1].set_ylabel('W',labelpad=-1)
        axGCdata[0,1].set_yticks([])
        axGCdata[0,1].legend(['GC','_','_','RSX'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)

        axGCdata[1,0].set_xlabel('W spectral',labelpad=-1)
        axGCdata[1,0].set_ylabel('W GCX',labelpad=-1)
        axGCdata[1,0].legend(['#3','#2','#1'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)

        axGCdata[1,1].set_xlabel('W GC',labelpad=-1)
        axGCdata[1,1].set_ylabel('W SpectralX',labelpad=-1)
        axGCdata[1,1].legend(['#3','#2','#1'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)

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

        image_name = " GC crossover plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figGCdata.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()

        pcaFA = npls.nipals(
            X_data=simulated_FA,
            maximum_number_PCs=10,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pcaFA.calc_PCA()

        figFA, axFA = plt.subplots(1, 2, figsize=pcaMC.fig_Size)
        axFA[0] = plt.subplot2grid((1,20), (0, 14), colspan=6, )
        axFA[1] = plt.subplot2grid((1,20), (0, 0), colspan=12, )
        figFA.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        lOff = np.round(-(np.max(pcaFA.spectral_loading[1,:])-np.min(pcaFA.spectral_loading[0,:]))*12.5)/10 #offset by a simple number
        axFA[0].plot(wavelength_axis,pcaFA.spectral_loading[0,:],'k')
        axFA[0].plot(wavelength_axis,pcaFA.spectral_loading[1,:] + lOff,color=[0.4,0.4,0.4])
        axFA[0].plot([wavelength_axis[0],wavelength_axis[-1]],[0,0],'--',color=[0.7,0.7,0.7],lw=0.5)
        axFA[0].plot([wavelength_axis[0],wavelength_axis[-1]],[lOff,lOff],'--',color=[0.7,0.7,0.7],lw=0.5)

        axFA[1].plot(pcaFA.component_weight[:,0],pcaFA.component_weight[:,1],'.')
        offset = 1
        for iFA in range(pcaFA.N_Obs):
            if FAnames[iFA][-1]=='0':
                col = [1,0,0]
            elif FAnames[iFA][-1]=='t':
                col = [0,1,0]
            elif FAnames[iFA][-1]=='c':
                col = [0,0,1]
            else:
                col = [0,0.5,1]
                offset = 0.75
            p1 = pcaFA.component_weight[iFA,0]
            p2 = pcaFA.component_weight[iFA,1]
            axFA[1].annotate(  FAnames[iFA],
                                xy=(p1, p2),
                                xytext=(p1+offset*np.sign(p1), p2+offset*np.sign(p2)),
                                textcoords="data",xycoords="data",
                                fontsize=pcaMC.fig_Text_Size*0.75,
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
            xytext=(1550, pcaFA.spectral_loading[0,349]+0.01),
            textcoords="data",
            xycoords="axes fraction",
            horizontalalignment="center",
        )
        axFA[0].annotate(
            'PC2',
            xy=(0.18, 0.89),
            xytext=(1550, pcaFA.spectral_loading[1,349]+lOff+0.01),
            textcoords="data",
            xycoords="axes fraction",
            horizontalalignment="center",
        )

        image_name = " reference FA PCA."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figFA.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()

### **** END Interpretation ****

### **** Start Preprocessing
        nPC = 4
        SF = np.ones(data.shape[0])
        SF[range(73,84)] = np.arange(1,2.1,0.1)
        SF[range(84,350)] = SF[range(84,350)]*2
        SF[range(350,360)] = np.arange(2,1,-0.1)
        #offset the alyl modes but not their variation,to power of SF
        spectraCCpSF = ( data.copy().T+ simulated_Collagen ).T


        #scale the alyl modes across its dynamic range,to power of SF
        spectraCCxSF = data.copy()**2
        spectraCCxSF = spectraCCxSF*np.sum(data[520:570,:],axis=0)/np.sum(spectraCCxSF[520:570,:],axis=0)
### Data perturbation
        figData4Preprocessing, axData4Preprocessing = plt.subplots(2, 2, figsize=pcaMC.fig_Size)
        axData4Preprocessing[0,0] = plt.subplot2grid((2, 12), (0, 0), colspan=5)
        axData4Preprocessing[0,1] = plt.subplot2grid((2, 12), (0, 7), colspan=5)
        axData4Preprocessing[1,0] = plt.subplot2grid((2, 12), (1, 0), colspan=5)
        axData4Preprocessing[1,1] = plt.subplot2grid((2, 12), (1, 7), colspan=5)
        figData4Preprocessing.subplots_adjust(left=0.11,right=0.99,top=0.97,bottom=0.1)

        axData4Preprocessing[0,0].plot( wavelength_axis ,  data)
        axData4Preprocessing[0,0].plot(wavelength_axis[[104,204]],np.tile(np.max(data[101,:]),2),'r')
        axData4Preprocessing[0,0].plot(wavelength_axis[[104,204]],np.tile(np.min(data[101,:]),2),'k--')
        axData4Preprocessing[0,1].annotate(
            "a)",
            xy=(0.13, 0.96),
            xytext=(0.01, 0.94),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
        )
        axData4Preprocessing[0,0].annotate(
            "range: %.0f" %np.ptp(data[101,:]),
            xy=(0.13, 0.96),
            xytext=(wavelength_axis[110],np.mean(data[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color = 'k',
            horizontalalignment="left",
            va = 'center'
        )

        axData4Preprocessing[0,1].plot( wavelength_axis ,  reps_4_noise)
        axData4Preprocessing[0,1].plot(wavelength_axis[[104,204]],np.tile(np.max(reps_4_noise[101,:]),2),'r')
        axData4Preprocessing[0,1].plot(wavelength_axis[[104,204]],np.tile(np.min(reps_4_noise[101,:]),2),'k--')
        axData4Preprocessing[0,1].annotate(
            "b)",
            xy=(0.13, 0.96),
            xytext=(0.01, 0.94),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
        )
        axData4Preprocessing[0,1].annotate(
            "range: %.0f" %np.ptp(reps_4_noise[101,:]),
            xy=(0.13, 0.96),
            xytext=(wavelength_axis[110],np.mean(reps_4_noise[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color = 'k',
            horizontalalignment="left",
            va = 'center'
        )

        axData4Preprocessing[1,0].plot( wavelength_axis , spectraCCpSF)
        axData4Preprocessing[1,0].plot(wavelength_axis[[104,204]],np.tile(np.max(spectraCCpSF[101,:]),2),'r')
        axData4Preprocessing[1,0].plot(wavelength_axis[[104,204]],np.tile(np.min(spectraCCpSF[101,:]),2),'k--')
        axData4Preprocessing[1,0].annotate(
            "range: %.0f" %np.ptp(spectraCCpSF[101,:]),
            xy=(0.13, 0.96),
            xytext=(wavelength_axis[110],np.mean(spectraCCpSF[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
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
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
        )

        axData4Preprocessing[1,1].plot( wavelength_axis , spectraCCxSF  )
        axData4Preprocessing[1,1].set_ylabel("Intensity / Counts")
        axData4Preprocessing[1,1].set_xlabel("Raman Shift cm$^{-1}$")
        axData4Preprocessing[1,1].plot(wavelength_axis[[104,204]],np.tile(np.max(spectraCCxSF[101,:]),2),'r')
        axData4Preprocessing[1,1].plot(wavelength_axis[[104,204]],np.tile(np.min(spectraCCxSF[101,:]),2),'k--')
        axData4Preprocessing[1,1].annotate(
            "d)",
            xy=(0.13, 0.96),
            xytext=(0.01, 0.94),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size,
            horizontalalignment="left",
        )
        axData4Preprocessing[1,1].annotate(
            "range: %.0f" %np.ptp(spectraCCxSF[101,:]),
            xy=(0.13, 0.96),
            xytext=(wavelength_axis[110],np.mean(spectraCCxSF[101,:]) ),
            textcoords="data",
            xycoords="axes fraction",
            fontsize=pcaMC.fig_Text_Size*0.75,
            color = 'k',
            horizontalalignment="left",
            va = 'center'
        )


        image_name = " Spectra offset and scale."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        plt.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)        # plt.show()
        plt.close()


### PCA models

        # no preprocessing.
        PCACCpSF = npls.nipals(
            X_data=spectraCCpSF,
            maximum_number_PCs=nPC,
            preproc="none",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCACCpSF.calc_PCA()


        # peak offset mean centred.
        PCACCpSFMC = npls.nipals(
            X_data=spectraCCpSF,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCACCpSFMC.calc_PCA()


        # peak range.
        PCACCxSF = npls.nipals(
            X_data=spectraCCxSF,
            maximum_number_PCs=nPC,
            preproc="none",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCACCxSF.calc_PCA()

        PCACCxSFMC = npls.nipals(
            X_data=spectraCCxSF,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCACCxSFMC.calc_PCA()

        #  unit scaled version,
        pcaMCUV = npls.nipals(
            X_data = data,
            maximum_number_PCs = nPC,
            preproc = "MCUV",
            pixel_axis = wavelength_axis,
            spectral_weights = butter_profile,
            min_spectral_values = min_data,
        )
        pcaMCUV.calc_PCA()

        pca_1_noiseUV = npls.nipals(
            X_data=data_1_noise,
            maximum_number_PCs=nPC,
            preproc="MCUV",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pca_1_noiseUV.calc_PCA()

        pca_1_noiseSqrt = npls.nipals(
            X_data=(data_1_noise+100)**.5,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pca_1_noiseSqrt.calc_PCA()

        pca_1_noiseLn = npls.nipals(
            X_data=np.log(data_1_noise+100),
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        pca_1_noiseLn.calc_PCA()

        # unit scaled version of the dataset manipulated in one peak region
        PCACCxSFMCUV = npls.nipals(
            X_data = spectraCCxSF,
            maximum_number_PCs = nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "MCUV",
            pixel_axis = wavelength_axis,
            spectral_weights = butter_profile,
            min_spectral_values = min_data,
        )
        PCACCxSFMCUV.calc_PCA()

        # log transformed version of the manipulated data
        PCACCxSFLogMC = npls.nipals(
            X_data = np.log10(spectraCCxSF),
            maximum_number_PCs = nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "MC",
            pixel_axis = wavelength_axis,
            spectral_weights = butter_profile,
            min_spectral_values = min_data,
        )
        PCACCxSFLogMC.calc_PCA()

        pcaNonMC = npls.nipals(
            X_data = data,
            maximum_number_PCs = nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "NA",
            pixel_axis = wavelength_axis,
            spectral_weights = butter_profile,
            min_spectral_values = min_data,
        )
        pcaNonMC.calc_PCA()

### mean centring
        figMCplots, axMCplots = plt.subplots(2, 2, figsize=pcaMC.fig_Size)
        axMCplots[0,0] = plt.subplot2grid((13,13), (0, 0), colspan=6, rowspan=6)
        axMCplots[0,1] = plt.subplot2grid((13,13), (0, 7), colspan=6, rowspan=6)
        axMCplots[1,0] = plt.subplot2grid((13,13), (7, 0), colspan=6, rowspan=6)
        axMCplots[1,1] = plt.subplot2grid((13,13), (7, 7), colspan=6, rowspan=6)
        figMCplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        axMCplots[0,0].plot( pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaNonMC.spectral_loading[:3,:])/2)
                            +PCACCpSF.spectral_loading[:3,:].T, lw=2)
        axMCplots[0,0].plot( pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaNonMC.spectral_loading[:3,:])/2)
                            +pcaNonMC.spectral_loading[:3,:].T,'--k', lw=1.5 )
        MAD_NonvspSF = np.absolute(PCACCpSF.spectral_loading[:3,:]-pcaNonMC.spectral_loading[:3,:]).mean() #mean absolute difference
        axMCplots[0,1].plot( pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaNonMC.spectral_loading[:3,:])/2)
                            +PCACCxSF.spectral_loading[:3,:].T, lw=2)
        axMCplots[0,1].plot( pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaNonMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaNonMC.spectral_loading[:3,:])/2)
                            +pcaNonMC.spectral_loading[:3,:].T,'--k', lw=1.5 )
        MAD_NonvsxSF = np.absolute(PCACCxSF.spectral_loading[:3,:]-pcaNonMC.spectral_loading[:3,:]).mean() #mean absolute difference
        axMCplots[1,0].plot( pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMC.spectral_loading[:3,:])/2)
                            +PCACCpSFMC.spectral_loading[:3,:].T, lw=2)
        axMCplots[1,0].plot( pcaMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMC.spectral_loading[:3,:])/2)
                            +pcaMC.spectral_loading[:3,:].T,'--k', lw=1.5)
        MAD_MCvspSF = np.absolute(PCACCpSFMC.spectral_loading[:3,:]-pcaMC.spectral_loading[:3,:]).mean() #mean absolute difference
        axMCplots[1,1].plot( pcaNonMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMC.spectral_loading[:3,:])/2)
                            +PCACCxSFMC.spectral_loading[:3,:].T, lw=2)
        axMCplots[1,1].plot( pcaMC.pixel_axis ,
                            np.arange(0,-np.ptp(pcaMC.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMC.spectral_loading[:3,:])/2)
                            +pcaMC.spectral_loading[:3,:].T,'--k', lw=1.5)
        MAD_MCvsxSFMC = np.absolute(PCACCxSFMC.spectral_loading[:3,:]-pcaMC.spectral_loading[:3,:]).mean() #mean absolute difference
        axMCplots[0,0].legend(['PC1','PC2','PC3','Basic'],bbox_to_anchor=(1.2, 0),
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)

        subLabels = [r"a) $L^\top_{Matrix}$ vs $L^\top_{Basic}$",
                     r"b) $L^\top_{Squared}$ vs $L^\top_{Basic}$",
                     r"c) $L^\top_{Matrix}$MC vs $L^\top_{Basic}$MC",
                     r"d) $L^\top_{Squared}$MC vs $L^\top_{Basic}$MC"]
        for ax1 in range(2):
            for ax2 in range(2):
                axMCplots[ax1,ax2].annotate(
                    subLabels[ax1*2+ax2],
                    xy=(0.18, 0.89),
                    xytext=(0.5, 1.05),
                    textcoords="axes fraction",
                    xycoords="axes fraction",
                    horizontalalignment="center",
                )
                if not pcaMC.fig_Show_Values:
                    axMCplots[ax1,ax2].axis("off")

                if pcaMC.fig_Show_Labels:
                    axMCplots[ax1,ax2].set_ylabel(pcaMC.fig_Y_Label)
                    axMCplots[ax1,ax2].set_xlabel(pcaMC.fig_X_Label)
        MAD_MCvspSFMC = np.absolute(PCACCpSFMC.spectral_loading[:3,:]-pcaMC.spectral_loading[:3,:]).mean() #mean absolute difference
        print('Mean Absolute Deviation w/o correcting offset with MC:' + str(MAD_NonvspSF))
        print('Mean Absolute Deviation w/o correcting scale with MC:' + str(MAD_NonvsxSF))
        print('Mean Absolute Deviation for correcting offset with MC:' + str(MAD_MCvspSFMC))
        print('Mean Absolute Deviation for correcting scale with MC:' + str(MAD_MCvsxSFMC))
        image_name = " Mean Centring plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figMCplots.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()
### Variable wise normalisation
        figUVplots, axUVplots = plt.subplots(2, 2, figsize=pcaMC.fig_Size)
        axUVplots[0,0] = plt.subplot2grid((13,13), (0, 0), colspan=6, rowspan=6)
        axUVplots[0,1] = plt.subplot2grid((13,13), (0, 7), colspan=6, rowspan=6)
        axUVplots[1,0] = plt.subplot2grid((13,13), (7, 0), colspan=6, rowspan=6)
        axUVplots[1,1] = plt.subplot2grid((13,13), (7, 7), colspan=6, rowspan=6)
        figUVplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)

        axUVplots[0,0].plot( pcaMCUV.pixel_axis ,
                            np.arange(0,-np.ptp(pcaMCUV.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMCUV.spectral_loading[:3,:])/2)
                            +PCACCxSFMCUV.spectral_loading[:3,:].T, lw=2)
        axUVplots[0,0].plot( pcaMCUV.pixel_axis ,
                            np.arange(0,-np.ptp(pcaMCUV.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pcaMCUV.spectral_loading[:3,:])/2)
                            +pcaMCUV.spectral_loading[:3,:].T, '--k', lw=1.5)
        MAD_MCUVvsxSFMCUV = np.absolute(PCACCxSFMCUV.spectral_loading[:3,:]-pcaMCUV.spectral_loading[:3,:]).mean() #mean absolute difference
        print('Mean Absolute Deviation for correcting scale with MCUV:' + str( MAD_MCUVvsxSFMCUV))


        axUVplots[0,1].plot( pca_1_noiseUV.pixel_axis ,
                            np.arange(0,-np.ptp(pca_1_noise.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noise.spectral_loading[:3,:])/2)
                            +pca_1_noiseUV.spectral_loading[:3,:].T, lw=2)
        axUVplots[0,1].plot( pca_1_noise.pixel_axis ,
                            np.arange(0,-np.ptp(pca_1_noise.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noise.spectral_loading[:3,:])/2)
                            +pca_1_noise.spectral_loading[:3,:].T,'--k', lw=1.5)

        axUVplots[1,0].plot( pca_1_noiseSqrt.pixel_axis ,
                            np.arange(0,-np.ptp(pca_1_noiseSqrt.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noiseSqrt.spectral_loading[:3,:])/2)
                            +pca_1_noiseSqrt.spectral_loading[:3,:].T, lw=2)
        axUVplots[1,1].plot( pca_1_noiseLn.pixel_axis ,
                            np.arange(0,-np.ptp(pca_1_noiseLn.spectral_loading[:3,:])*1.1,
                                      -np.ptp(pca_1_noiseLn.spectral_loading[:3,:])/2)
                            +pca_1_noiseLn.spectral_loading[:3,:].T, lw=2)
        axUVplots[0,0].legend(['PC1','PC2','PC3','Basic'],bbox_to_anchor=(1.2, 0),
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
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
                if not pcaMC.fig_Show_Values:
                    axUVplots[ax1,ax2].axis("off")

                if pcaMC.fig_Show_Labels:
                    axUVplots[ax1,ax2].set_ylabel(pcaMC.fig_Y_Label)
                    axUVplots[ax1,ax2].set_xlabel(pcaMC.fig_X_Label)

        image_name = " Scaling plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figUVplots.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()

        # create variation in intensity, then normalise to carbonyl, CHx and vector norm
        ints = stat.skewnorm.rvs(5,loc=1,scale=0.2,size=data.shape[1],random_state=6734325)
        data_Int = data*ints
        CO_Int = np.sum(data_Int[520:570,:],axis=0)
        CHx_Int = np.sum(data_Int[210:280,:],axis=0)
        Vec_Int = np.sum(data_Int**2,axis=0)**0.5
        PCA_Int = npls.nipals(
            X_data=data_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCA_Int.calc_PCA()
        PCA_IntCO = npls.nipals(
            X_data=data_Int/CO_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCA_IntCO.calc_PCA()
        PCA_IntCHx = npls.nipals(
            X_data=data_Int/CHx_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCA_IntCHx.calc_PCA()
        PCA_IntVec = npls.nipals(
            X_data=data_Int/Vec_Int,
            maximum_number_PCs=nPC,
            preproc="MC",
            pixel_axis=wavelength_axis,
            spectral_weights=butter_profile,
            min_spectral_values=min_data,
        )
        PCA_IntVec.calc_PCA()
### Sample wise Normalisation
        figIntplots, axIntplots = plt.subplots(1, 4, figsize=pcaMC.fig_Size)
        axIntplots[0] = plt.subplot2grid((1,27), (0, 0), colspan=6)
        axIntplots[1] = plt.subplot2grid((1,27), (0, 7), colspan=6)
        axIntplots[2] = plt.subplot2grid((1,27), (0, 14), colspan=6)
        axIntplots[3] = plt.subplot2grid((1,27), (0, 21), colspan=6)
        figIntplots.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.08)


        axIntplots[0].plot( PCA_Int.pixel_axis ,
                            np.arange(0,-np.ptp(PCA_Int.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_Int.spectral_loading[:3,:])/2)
                            +PCA_Int.spectral_loading[:3,:].T, lw=2)
        axIntplots[0].add_artist(patches.Ellipse((PCA_Int.pixel_axis[250][0],
                                          np.min(PCA_Int.spectral_loading[0,200:300]) + np.ptp(PCA_Int.spectral_loading[0,200:300])/2),
                                         width=PCA_Int.pixel_axis[300]-PCA_IntCO.pixel_axis[200],
                                         height=np.ptp(PCA_Int.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        axIntplots[1].plot( PCA_IntCO.pixel_axis ,
                            np.arange(0,-np.ptp(PCA_IntCO.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_IntCO.spectral_loading[:3,:])/2)
                            +PCA_IntCO.spectral_loading[:3,:].T,lw=2 )
        axIntplots[1].add_artist(patches.Ellipse((PCA_IntCO.pixel_axis[250][0],
                                          np.min(PCA_IntCO.spectral_loading[0,200:300]) + np.ptp(PCA_IntCO.spectral_loading[0,200:300])/2),
                                         width=PCA_IntCO.pixel_axis[300]-PCA_IntCO.pixel_axis[200],
                                         height=np.ptp(PCA_IntCO.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        cylim = axIntplots[1].get_ylim()
        axIntplots[1].plot(np.tile(PCA_IntCO.pixel_axis[520],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[1].plot(np.tile(PCA_IntCO.pixel_axis[570],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[1].plot([PCA_IntCO.pixel_axis[520],PCA_IntCO.pixel_axis[570]],np.tile(cylim[0]*0.9,2),'g',linestyle='--')
        axIntplots[1].plot([PCA_IntCO.pixel_axis[520],PCA_IntCO.pixel_axis[570]],np.tile(cylim[1]*0.9,2),'g',linestyle='--')
        axIntplots[1].set_ylim(cylim)
        axIntplots[2].plot( PCA_IntCHx.pixel_axis ,
                            np.arange(0,-np.ptp(PCA_IntCHx.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_IntCHx.spectral_loading[:3,:])/2)
                            +PCA_IntCHx.spectral_loading[:3,:].T, lw=2)
        axIntplots[2].add_artist(patches.Ellipse((PCA_IntCHx.pixel_axis[250][0],
                                          np.min(PCA_IntCHx.spectral_loading[0,200:300]) + np.ptp(PCA_IntCHx.spectral_loading[0,200:300])/2),
                                         width=PCA_IntCHx.pixel_axis[300]-PCA_IntCHx.pixel_axis[200],
                                         height=np.ptp(PCA_IntCHx.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        cylim = axIntplots[2].get_ylim()
        axIntplots[2].plot(np.tile(PCA_IntCHx.pixel_axis[210],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[2].plot(np.tile(PCA_IntCHx.pixel_axis[280],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[2].plot([PCA_IntCHx.pixel_axis[210],PCA_IntCHx.pixel_axis[280]],np.tile(cylim[0]*0.9,2),'g',linestyle='--')
        axIntplots[2].plot([PCA_IntCHx.pixel_axis[210],PCA_IntCHx.pixel_axis[280]],np.tile(cylim[1]*0.9,2),'g',linestyle='--')
        axIntplots[2].set_ylim(cylim)
        axIntplots[3].plot( PCA_IntVec.pixel_axis ,
                            np.arange(0,-np.ptp(PCA_IntVec.spectral_loading[:3,:])*1.1,
                                      -np.ptp(PCA_IntVec.spectral_loading[:3,:])/2)
                            +PCA_IntVec.spectral_loading[:3,:].T,lw=2 )
        axIntplots[3].add_artist(patches.Ellipse((PCA_IntVec.pixel_axis[250][0],
                                          np.min(PCA_IntVec.spectral_loading[0,200:300]) + np.ptp(PCA_IntVec.spectral_loading[0,200:300])/2),
                                         width=PCA_IntVec.pixel_axis[300]-PCA_IntVec.pixel_axis[200],
                                         height=np.ptp(PCA_IntVec.spectral_loading[0,200:300])*1.3,
                                         facecolor='none', edgecolor='r'))
        cylim = axIntplots[3].get_ylim()
        axIntplots[3].plot(np.tile(PCA_IntVec.pixel_axis[0],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[3].plot(np.tile(PCA_IntVec.pixel_axis[600],(2)),[cylim[0]*0.9,cylim[1]*0.9],'g',linestyle='--')
        axIntplots[3].plot([PCA_IntVec.pixel_axis[0],PCA_IntVec.pixel_axis[600]],np.tile(cylim[0]*0.9,2),'g',linestyle='--')
        axIntplots[3].plot([PCA_IntVec.pixel_axis[0],PCA_IntVec.pixel_axis[600]],np.tile(cylim[1]*0.9,2),'g',linestyle='--')
        axIntplots[3].set_ylim(cylim)

        subLabels = [r"a) $L^\top_{Int}$ MC Unnorm", r"b) $L^\top_{Int}$ MC/C=O",
                     r"c) $L^\top_{Int}$ MC/CH$_x$", r"d)$L^\top_{Int}$ MC/Norm"]
        #find out how to display a square root rather than calculate one
        for ax1 in range(axIntplots.shape[0]):
            axIntplots[ax1].annotate(
                subLabels[ax1],
                xy=(0.18, 0.89),
                xytext=(0.5, 0.98),
                textcoords="axes fraction",
                xycoords="axes fraction",
                horizontalalignment="center",
            )
            if not pcaMC.fig_Show_Values:
                axIntplots[ax1].axis("off")

            if pcaMC.fig_Show_Labels:
                axIntplots[ax1].set_ylabel(pcaMC.fig_Y_Label)
                axIntplots[ax1].set_xlabel(pcaMC.fig_X_Label)

        image_name = " Normalisation plots."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figIntplots.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()
### Cage of Covariance
#exploring the impact of perturbing the cage of covariance on model applicability
# need to have a test set with same cage then variations on the cage

# Butter validation set - same cage of covariance
# determine copula of original butter GC and use to generate a new test set
        copula = GaussianMultivariate()
        copula.fit(butter_profile.T)
        butter_profile_Val = np.array(copula.sample(100).T)
        spectra_butter_Val = np.dot(
            simulated_FA,butter_profile_Val)

# Butter with no covariance at all
        #create a butter_profile with no correlation
        FAmeans = np.mean(butter_profile,1)
        FAsd = np.std(butter_profile,1)
        # generate random numbers in array same size as butter_profile, scale by 1FAsd then add on
        # mean value
        butter_profile_NoCov = np.random.randn(*butter_profile.shape)
        butter_profile_NoCov = (butter_profile_NoCov.T*FAsd)+FAmeans
        butter_profile_NoCov = 100*butter_profile_NoCov.T/np.sum(butter_profile_NoCov,axis=1)

        spectraNoCov = np.dot(simulated_FA,butter_profile_NoCov)

# Butter with imputed Adipose covariance
        PCA_GC_Adipose = PCA(n_components=10,) #use of copula method retains central as well as marginal distribution covariances, so rebuild using covariance in PCA. This will mean FA means not exactly replicated as covariance constrains possible
        meanFA_adipose = np.mean(adipose_profile, axis=1) # mean centre for model
        PCA_GC_Adipose.fit(adipose_profile.T-meanFA_adipose)
#        butter_profileAdipose = (np.inner( np.inner(
#            butter_profile.T-meanFA_adipose, PCA_GC_Adipose.components_ ),
#            PCA_GC_Adipose.components_.T ) +meanFA_adipose).T
        butter_profileAdipose = (np.inner( np.inner(
            butter_profile.T-FAmeans, PCA_GC_Adipose.components_ ),
            PCA_GC_Adipose.components_.T ) +FAmeans).T
        spectra_butter_adiposeCov = np.dot(simulated_FA,butter_profileAdipose)

# also compare with adipose - this is an independent experiment, using a
# different lipid product (adipose vs milk derived butter), although one group
# is the same species (beef and butter both from cows)

        FA_profiles_sanity = np.vstack([ FAmeans,
                                        np.mean(butter_profile_Val,axis=1),
                                        np.mean(butter_profile_NoCov,axis=1),
                                        np.mean(butter_profileAdipose,axis=1),
                                        np.mean(adipose_profile,axis=1),
                                        ])
        figFApro, axFApro = plt.subplots(1, 1, figsize=pcaMC.fig_Size)
        axFApro.bar(np.arange(27),FA_profiles_sanity[0,:],facecolor='k')
        axFApro.bar(np.arange(27)-0.3,FA_profiles_sanity[1,:],facecolor='r',width=0.12)
        axFApro.bar(np.arange(27)-0.1,FA_profiles_sanity[2,:],facecolor='b',width=0.12)
        axFApro.bar(np.arange(27)+0.1,FA_profiles_sanity[3,:],facecolor='c',width=0.12)
        axFApro.bar(np.arange(27)+0.3,FA_profiles_sanity[4,:],facecolor='g',width=0.12)
        axFApro.set_xticks(np.arange(len(FAnames)))
        axFApro.set_xticklabels(FAnames,fontsize=6, rotation=45)
        axFApro.set_xlabel('Fatty Acid')
        axFApro.set_ylabel('Molar %')
        axFApro.legend(['Train','Val','NoCov','AdiCov','Adipose'],fontsize='x-small')

        image_name = " FA profiles sanity checks."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figFApro.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()

        figFAproMC, axFAproMC = plt.subplots(1, 1, figsize=pcaMC.fig_Size)
        axFAproMC.bar(np.arange(27),FA_profiles_sanity[0,:]-FA_profiles_sanity[0,:],facecolor='k')
        axFAproMC.bar(np.arange(27)-0.3,FA_profiles_sanity[1,:]-FA_profiles_sanity[0,:],facecolor='r',width=0.12)
        axFAproMC.bar(np.arange(27)-0.1,FA_profiles_sanity[2,:]-FA_profiles_sanity[0,:],facecolor='b',width=0.12)
        axFAproMC.bar(np.arange(27)+0.1,FA_profiles_sanity[3,:]-FA_profiles_sanity[0,:],facecolor='c',width=0.12)
        axFAproMC.bar(np.arange(27)+0.3,FA_profiles_sanity[4,:]-FA_profiles_sanity[0,:],facecolor='g',width=0.12)
        axFAproMC.set_xticks(np.arange(len(FAnames)))
        axFAproMC.set_xticklabels(FAnames,fontsize=6, rotation=45)
        axFAproMC.set_xlabel('Fatty Acid')
        axFAproMC.set_ylabel('Molar %')
        axFAproMC.legend(['Train','Val','NoCov','Adipose','AdiCov'],fontsize='x-small')


        image_name = " FA profile Sanity Check Difference"
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figFAproMC.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
        plt.close()
# apply butter model to each alternative
        TestSet_Train = pcaMC.project_model(data)
        TestSet_Val = pcaMC.project_model(spectra_butter_Val)
        TestSet_NoCov = pcaMC.project_model(spectraNoCov)
        TestSet_adiposeCov = pcaMC.project_model(spectra_butter_adiposeCov)
        TestSet_adipose = pcaMC.project_model(adipose_data)

        # Validation Figure
        figVal, axVal = plt.subplots(2, 2, figsize=pcaMC.fig_Size)
        axVal[0,0] = plt.subplot2grid((23, 11), (0, 0), colspan=5, rowspan=10)
        axVal[0,1] = plt.subplot2grid((23, 11), (0, 6), colspan=5, rowspan=10)
        axVal[1,0] = plt.subplot2grid((23, 11), (13, 0), colspan=5, rowspan=10)
        axVal[1,1] = plt.subplot2grid((23, 11), (13, 6), colspan=5, rowspan=10)
        figVal.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.09)

        # Generated GC profiles
        axVal[0,0].plot(np.arange(0,27)-0.2 , adipose_profile,'dg', markersize=2)
        axVal[0,0].plot(np.arange(0,27)-0.1 , butter_profileAdipose,'^c',alpha=0.5, markersize=2)
        axVal[0,0].plot(np.arange(0,27) , butter_profile_NoCov,'*b',alpha=0.5, markersize=2)
        axVal[0,0].plot(np.arange(0,27)+0.1 , butter_profile_Val, '.r',alpha=0.5, markersize=2)
        axVal[0,0].plot(np.arange(0,27)+0.2 , butter_profile,'+k',alpha=0.5, markersize=2)
        axVal[0,0].set_xlim(0,26)
        axVal[0,0].set_ylim(0,axVal[0,0].get_ylim()[1])
        axVal[0,0].set_xticks(np.arange(len(FAnames)))
        axVal[0,0].set_xticklabels(FAnames,fontsize=6, rotation=45)
        axVal[0,0].set_xlabel('Fatty Acid')
        axVal[0,0].set_ylabel('Molar %')

        # generated spectra
        trunc = np.concatenate([np.arange(29,130),np.arange(209,300),np.arange(429,490)])#,np.arange(519,570)])
        truncTicks = np.concatenate([np.arange(0,96,25),[95],np.arange(106,180,25),[186],np.arange(197,247,25),[246]])#,np.arange(257,304,25),[303]])
        tempIx = np.concatenate([np.arange(96,106),np.arange(187,197)])#,np.arange(247,257)])

        tempDat = np.mean(spectra_butter_Val[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'r')
        tempDat = np.mean(spectraNoCov[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'b',linestyle=(0,(1.1,1.3)))
        tempDat = np.mean(spectra_butter_adiposeCov[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'c',linestyle=(1.5,(1.3,1.7)))
        tempDat = np.mean(adipose_data[trunc,:],axis=1)/4
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'g')
        axVal[0,1].plot(tempDat*0,'k',linestyle=(0.5,(0.3,0.7)))


        tempDat = np.std(spectra_butter_Val[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'r')
        tempDat = np.std(spectraNoCov[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'b')
        tempDat = np.std(spectra_butter_adiposeCov[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'c')
        tempDat = np.std(adipose_data[trunc,:],axis=1)-100
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'g')
        axVal[0,1].plot(tempDat*0 - 100,'k',linestyle=(0.5,(0.3,0.7)))

        axVal[0,1].set_xticks(truncTicks)
        axVal[0,1].set_xticklabels(wavelength_axis[trunc[truncTicks]], rotation=45,fontsize='x-small')
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

        axVal[1,0].plot(np.arange(1,pcaMC.N_PC+1),TestSet_Train["Q2"] , 'k')
        axVal[1,0].plot(np.arange(1,pcaMC.N_PC+1),TestSet_Val["Q2"],'r')
        axVal[1,0].plot(np.arange(1,pcaMC.N_PC+1),TestSet_NoCov["Q2"],'b')
        axVal[1,0].plot(np.arange(1,pcaMC.N_PC+1),TestSet_adiposeCov["Q2"],'c')
        axVal[1,0].plot(np.arange(1,pcaMC.N_PC+1),TestSet_adipose["Q2"],'g')
        axVal[1,0].legend(['Train','Val','NoCov','AdiCov','Adipose'])
        axVal[1,0].set_xlabel('PC rank')
        axVal[1,0].set_ylabel('Cumulative $Q^2$')

        BVsc=pcaGC.transform(np.transpose( butter_profile_Val))
        BNsc=pcaGC.transform(np.transpose( butter_profile_NoCov))
        ADsc = pcaGC.transform(np.transpose(adipose_profile))
        BAsc = pcaGC.transform(np.transpose(butter_profileAdipose))
        species_marker = ['*', '+', 'x', '1']
        axVal[1,1].plot(-GCsc[:, 0],-GCsc[:, 1],'.k',fillstyle='full')
        axVal[1,1].plot(-BVsc[:, 0],-BVsc[:, 1],'+r',fillstyle='none')
        axVal[1,1].plot(-BNsc[:, 0],-BNsc[:, 1],'<b',fillstyle='none')
        axVal[1,1].plot(-BAsc[:, 0],-BAsc[:, 1],'*c',fillstyle='none')
        for iS in range(4):
            ix = np.where(adipose_species==iS )[0]
            axVal[1,1].plot(-ADsc[ix, 0],-ADsc[ix, 1],species_marker[iS],color = 'g', fillstyle='none')
        #Calculate ellipse bounds and plot with scores
        theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
        circle = np.array((np.cos(theta), np.sin(theta)))
        sigma = np.cov(np.array((GCsc[:, 0], GCsc[:, 1])))
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
        width, height = bbox.width, bbox.height
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


        image_name = " Validation Sets."
        full_path = os.path.join(images_folder, pcaMC.fig_Project +
                                image_name + pcaMC.fig_Format)
        figVal.savefig(full_path,
                         dpi=pcaMC.fig_Resolution)
### ******      END CLASS      ******
        return
