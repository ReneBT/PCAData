# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:44:41 2023

@author: rene
"""
import numpy as np
import src.local_nipals as npls

### shot noise    
def  gen_data(
    data,
    ):
        ### Generate Noisy Data
        data.shot_noise = np.random.randn(np.shape(data.data)[0],np.shape(data.data)[1])/10
        data.majpk = np.where(np.mean(data.data,axis=1)>np.mean(data.data))
        data.signal = np.mean(data.data[data.majpk,:],axis=1)
        data.data_q_noise = data.data + ((data.data**0.5 + 10) * data.shot_noise / 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in quarter scaled noise: ' +
              str(np.mean(data.signal/np.std((data.data[data.majpk,:]**0.5 + 10)
                                        * data.shot_noise[data.majpk,:] / 4,axis=1))))
        data.data_1_noise = data.data + ((data.data**0.5 + 10) * data.shot_noise) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in unscaled noise: ' +
              str(np.mean(data.signal/np.std((data.data[data.majpk,:]**0.5 + 10)
                                        * data.shot_noise[data.majpk,:],axis=1))))
        data.data_4_noise = data.data + ((data.data**0.5 + 10) * data.shot_noise * 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        print('SNR achieved in 4 times scaled noise: ' +
              str(np.mean(data.signal/np.std((data.data[data.majpk,:]**0.5 + 10)
                                        * data.shot_noise[data.majpk,:]*4,axis=1))))
        print( 'Noise Data Generated, PCA: 1/4 Noise')
        data.pca_q_noise = npls.nipals(
            X_data=data.data_q_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pca_q_noise.calc_PCA()
        print( 'Noise Data Generated, PCA: x1 Noise')
        data.pca_1_noise = npls.nipals(
            X_data=data.data_1_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pca_1_noise.calc_PCA()
        print( 'Noise Data Generated, PCA: x4')
        data.pca_4_noise = npls.nipals(
            X_data=data.data_4_noise,
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pca_4_noise.calc_PCA()

        print( 'Noise Data Generated, PCA: Noise Only')
        data.pca_noise = npls.nipals(
            X_data=((data.data**0.5 + 10) * data.shot_noise),
            maximum_number_PCs=80,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pca_noise.calc_PCA()# noise from SNR 100

        print( 'Noise PCA complete. PC inter model correlations...')
        data.corrPCs_noiseq = np.inner(data.pca_q_noise.spectral_loading,data.pcaMC.spectral_loading)
        data.corrPCs_noise1 = np.inner(data.pca_1_noise.spectral_loading,data.pcaMC.spectral_loading)
        data.corrPCs_noise4 = np.inner(data.pca_4_noise.spectral_loading,data.pcaMC.spectral_loading)
        data.corrPCs_noise = np.inner(data.pca_noise.spectral_loading,data.pcaMC.spectral_loading)
        #loadings already standardised to unit norm
        data.corrPCs_noiseq_R2sum_ax0 = np.sum(data.corrPCs_noiseq**2,axis=0)#total variance shared between each noiseq loading and the noisless loadings
        data.corrPCs_noiseq_R2sum_ax1 = np.sum(data.corrPCs_noiseq**2,axis=1)#total variance shared between each noiseless PC and the noiseq loadings

        data.maxCorr_noiseq = np.max(np.abs(data.corrPCs_noiseq),axis=0)
        data.maxCorr_noise1 = np.max(np.abs(data.corrPCs_noise1),axis=0)
        data.maxCorr_noise4 = np.max(np.abs(data.corrPCs_noise4),axis=0)
        data.maxCorr_noise = np.max(np.abs(data.corrPCs_noise),axis=0)
        data.maxCorrMean = ((np.sum(data.maxCorr_noise**2)**0.5)/
                data.maxCorr_noise.shape[0]**0.5) #correlation measures noise so propagate as variance
        print('Mean optimal Correlation : ' + str(data.maxCorrMean))
        print('SE Correlation : ' + str(data.maxCorrMean + [-np.std(data.maxCorr_noise),np.std(data.maxCorr_noise)]))

        data.max_Ixq = np.empty(*data.maxCorr_noiseq.shape)
        data.max_Ix1 = np.copy(data.max_Ixq)
        data.max_Ix4 = np.copy(data.max_Ixq)
        data.max_IxN = np.copy(data.max_Ixq)
        data.max_Snq = np.copy(data.max_Ixq)
        data.max_Sn1 = np.copy(data.max_Ixq)
        data.max_Sn4 = np.copy(data.max_Ixq)
        data.max_SnN = np.copy(data.max_Ixq)

        for iCol in range(np.shape(data.maxCorr_noiseq)[0]):
            data.max_Ixq[iCol] = np.where(np.abs(data.corrPCs_noiseq[:,iCol])==data.maxCorr_noiseq[iCol])[0]
            data.max_Snq[iCol] = np.sign(data.corrPCs_noiseq[data.max_Ixq[iCol].astype(int),iCol])
            data.max_Ix1[iCol] = np.where(np.abs(data.corrPCs_noise1[:,iCol])==data.maxCorr_noise1[iCol])[0]
            data.max_Sn1[iCol] = np.sign(data.corrPCs_noise1[data.max_Ix1[iCol].astype(int),iCol])
            data.max_Ix4[iCol] = np.where(np.abs(data.corrPCs_noise4[:,iCol])==data.maxCorr_noise4[iCol])[0]
            data.max_Sn4[iCol] = np.sign(data.corrPCs_noise4[data.max_Ix4[iCol].astype(int),iCol])
            data.max_IxN[iCol] = np.where(np.abs(data.corrPCs_noise[:,iCol])==data.maxCorr_noise[iCol])[0]
            data.max_SnN[iCol] = np.sign(data.corrPCs_noise[data.max_IxN[iCol].astype(int),iCol])

        print( 'Shot Noise Generated, creating perturbances: Shot Noise Repetitions')
        data.reps_4_noise = np.tile(data.data[:,23],(data.data.shape[1],1)).T
        data.reps_4_noise = data.reps_4_noise + ((data.reps_4_noise**0.5 + 10) * data.shot_noise * 4) #noise scales by square root of intensity - use 100 offset so baseline not close to zero
        data.reps_4_noise_recon = data.pcaMC.reduced_Rank_Reconstruction( data.reps_4_noise , 10 )

        return data

