# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:44:41 2023

@author: rene
"""
import numpy as np
import scipy.stats as stat
import src.local_nipals as npls


### shot noise    
def  gen_data(
    data,
    ):
### **** Start Preprocessing
        data.nPC = 4
        data.SF = np.ones(data.data.shape[0])
        data.SF[range(73,84)] = np.arange(1,2.1,0.1)
        data.SF[range(84,350)] = data.SF[range(84,350)]*2
        data.SF[range(350,360)] = np.arange(2,1,-0.1)
        #offset the alyl modes but not their variation,to power of SF
        data.spectraCCpSF = ( data.data.copy().T+ data.simulated_Collagen ).T


        #scale the alyl modes across its dynamic range,to power of SF
        data.spectraCCxSF = data.data.copy()**2
        data.spectraCCxSF = data.spectraCCxSF*np.sum(data.data[520:570,:],axis=0)/np.sum(data.spectraCCxSF[520:570,:],axis=0)


### PCA models
        print( 'Scale and Offset Generated, PCA models: offset no preprocessing')
        # no preprocessing.
        data.PCACCpSF = npls.nipals(
            X_data=data.spectraCCpSF,
            maximum_number_PCs=data.nPC,
            preproc="none",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCACCpSF.calc_PCA()


        print( 'Scale and Offset Generated, PCA models: o no preprocessing')
        # peak offset mean centred.
        data.PCACCpSFMC = npls.nipals(
            X_data=data.spectraCCpSF,
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCACCpSFMC.calc_PCA()


        # peak range.
        data.PCACCxSF = npls.nipals(
            X_data=data.spectraCCxSF,
            maximum_number_PCs=data.nPC,
            preproc="none",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCACCxSF.calc_PCA()

        data.PCACCxSFMC = npls.nipals(
            X_data=data.spectraCCxSF,
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCACCxSFMC.calc_PCA()

        #  unit scaled version,
        data.pcaMCUV = npls.nipals(
            X_data = data.data,
            maximum_number_PCs = data.nPC,
            preproc = "MCUV",
            pixel_axis = data.wavelength_axis,
            spectral_weights = data.butter_profile,
            min_spectral_values = data.min_data,
        )
        data.pcaMCUV.calc_PCA()

        data.pca_1_noiseUV = npls.nipals(
            X_data=data.data_1_noise,
            maximum_number_PCs=data.nPC,
            preproc="MCUV",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pca_1_noiseUV.calc_PCA()

        data.pca_1_noiseSqrt = npls.nipals(
            X_data=(data.data_1_noise+100)**.5,
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pca_1_noiseSqrt.calc_PCA()

        data.pca_1_noiseLn = npls.nipals(
            X_data=np.log(data.data_1_noise+100),
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pca_1_noiseLn.calc_PCA()

        # unit scaled version of the dataset manipulated in one peak region
        data.PCACCxSFMCUV = npls.nipals(
            X_data = data.spectraCCxSF,
            maximum_number_PCs = data.nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "MCUV",
            pixel_axis = data.wavelength_axis,
            spectral_weights = data.butter_profile,
            min_spectral_values = data.min_data,
        )
        data.PCACCxSFMCUV.calc_PCA()

        # log transformed version of the manipulated data
        data.PCACCxSFLogMC = npls.nipals(
            X_data = np.log10(data.spectraCCxSF),
            maximum_number_PCs = data.nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "MC",
            pixel_axis = data.wavelength_axis,
            spectral_weights = data.butter_profile,
            min_spectral_values = data.min_data,
        )
        data.PCACCxSFLogMC.calc_PCA()

        data.pcaNonMC = npls.nipals(
            X_data = data.data,
            maximum_number_PCs = data.nPC,
            maximum_iterations_PCs = 25,
            iteration_tolerance = 0.000000000001,
            preproc = "NA",
            pixel_axis = data.wavelength_axis,
            spectral_weights = data.butter_profile,
            min_spectral_values = data.min_data,
        )
        data.pcaNonMC.calc_PCA()

        MAD_NonvspSF = np.absolute(data.PCACCpSF.spectral_loading[:3,:]-
                                   data.pcaNonMC.spectral_loading[:3,:]).mean() #mean absolute difference
        MAD_NonvsxSF = np.absolute(data.PCACCxSF.spectral_loading[:3,:]-
                                   data.pcaNonMC.spectral_loading[:3,:]).mean() #mean absolute difference
        MAD_MCvsxSFMC = np.absolute(data.PCACCxSFMC.spectral_loading[:3,:]-
                                    data.pcaMC.spectral_loading[:3,:]).mean() #mean absolute difference
        MAD_MCvspSFMC = np.absolute(data.PCACCpSFMC.spectral_loading[:3,:]-
                                    data.pcaMC.spectral_loading[:3,:]).mean() #mean absolute difference
        print('Mean Absolute Deviation w/o correcting offset with MC:' + 
              str(MAD_NonvspSF))
        print('Mean Absolute Deviation w/o correcting scale with MC:' + 
              str(MAD_NonvsxSF))
        print('Mean Absolute Deviation for correcting offset with MC:' + 
              str(MAD_MCvspSFMC))
        print('Mean Absolute Deviation for correcting scale with MC:' + 
              str(MAD_MCvsxSFMC))
        
# varition in reference data
        data.pcaFA = npls.nipals(
            X_data=data.simulated_FA,
            maximum_number_PCs=10,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.pcaFA.calc_PCA()
        data.FAnames = ['']*data.FA_properties["Carbons"].shape[1]
        for i in range(len(data.FAnames)):
            if data.FA_properties[ "Isomer" ][ 0, i ]==3:
                data.FAnames[i] = 'CLA'
            else:
                if data.FA_properties[ "Isomer" ][ 0, i ]==0:
                    isoNm = ''
                elif data.FA_properties[ "Isomer" ][ 0, i ]==1:
                    isoNm = 'c'
                else:
                    isoNm = 't'
                data.FAnames[i] = ( str( int(data.FA_properties[ "Carbons" ][ 0, i ] )) 
                + ':' + str( int(data.FA_properties[ "Olefins" ][ 0, i ]) ) + isoNm )
# create variation in intensity, then normalise to carbonyl, CHx and vector norm
        ints = stat.skewnorm.rvs(5,loc=1,scale=0.2,size=data.data.shape[1],random_state=6734325)
        data.data_Int = data.data*ints
        data.CO_Int = np.sum(data.data_Int[520:570,:],axis=0)
        data.CHx_Int = np.sum(data.data_Int[210:280,:],axis=0)
        data.Vec_Int = np.sum(data.data_Int**2,axis=0)**0.5
        data.PCA_Int = npls.nipals(
            X_data=data.data_Int,
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCA_Int.calc_PCA()
        data.PCA_IntCO = npls.nipals(
            X_data=data.data_Int/data.CO_Int,
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCA_IntCO.calc_PCA()
        data.PCA_IntCHx = npls.nipals(
            X_data=data.data_Int/data.CHx_Int,
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCA_IntCHx.calc_PCA()
        data.PCA_IntVec = npls.nipals(
            X_data=data.data_Int/data.Vec_Int,
            maximum_number_PCs=data.nPC,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,
        )
        data.PCA_IntVec.calc_PCA()

        return data

