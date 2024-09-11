# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:44:41 2023

@author: rene
"""
from sklearn.decomposition import PCA
import numpy as np
import src.local_nipals as npls

### clean_data    
def  gen_data(
    data,
    ):
        data.wavelength_axis = data.simplified_fatty_acid_spectra["FAXcal"][:,]
        data.simulated_FA = data.simplified_fatty_acid_spectra["simFA"][:,:].T
        data.simulated_Collagen = data.simulated_FA[:,27]*1000
        data.simulated_FA = data.simulated_FA[:,range(27)]
        data.FA_properties = data.simplified_fatty_acid_spectra["FAproperties"]
        
        data.min_spectral_values = np.tile(
            np.min(data.simulated_FA, axis=1), (np.shape(data.simulated_FA)[1], 1)
        )
# convert the mass base profile provided into a molar profile
        data.butter_profile = data.GC_data["ButterGC"][:,:] / data.FA_properties["MolarMass"][:,:].T
        data.butter_profile = 100.0 * data.butter_profile / sum(data.butter_profile)
        data.sam_codes = data.GC_data["sample_ID"][:,]
        data.Grouping = np.empty(data.sam_codes.shape[1],dtype=bool)
        data.feed = np.empty(np.size(data.Grouping))
        data.week = np.empty(np.size(data.Grouping))
        for iSam in range(data.sam_codes.shape[1]):
            data.Grouping[ iSam ] = data.sam_codes[ 0, iSam ]=='B'
            data.feed[ iSam ] = int( chr( data.sam_codes[ 1, iSam ] ) )*2
            data.week[ iSam ] = int( chr( data.sam_codes[ 2, iSam] ) + chr( data.sam_codes[ 3, iSam ] ) )

        data.adipose_profile = data.GC_data["AdiposeGC"] [:,:] / data.FA_properties["MolarMass"][:,:].T
        data.adipose_profile = 100.0 * data.adipose_profile / sum(data.adipose_profile)
        data.adipose_species = data.GC_data["AdiposeSpecies"][0].astype(int)
### generate simulated observational
# spectra for each sample by multiplying the simulated FA reference spectra by
# the Fatty Acid profiles. Note that the data.simplified_fatty_acid_spectra spectra
# have a standard intensity in the carbonyl mode peak (the peak with the
# highest pixel position)
        data.data = np.dot(data.simulated_FA, data.butter_profile)
        data.min_data = np.dot(
            np.transpose(data.min_spectral_values), data.butter_profile
        )  # will allow scaling of data.min_spectral_values to individual sample

        data.adipose_data = np.dot(data.simulated_FA, data.adipose_profile)

        data.N_FA = data.FA_properties["Carbons"].shape[1]

### calculate PCA
# with the full fatty acid covariance, comparing custom NIPALS and built in PCA function.
        print( 'Basic Data Generated, comparing datatypes: Raman')
        data.pcaMC = npls.nipals(
            X_data= data.data,
            maximum_number_PCs=10,
            maximum_iterations_PCs=100,
            iteration_tolerance=0.000000000001,
            preproc="MC",
            pixel_axis=data.wavelength_axis,
            spectral_weights=data.butter_profile,
            min_spectral_values=data.min_data,

        )
        data.pcaMC.figure_Settings( res=800,
                    frmt='tif', )
        data.pcaMC.calc_PCA()
        
        print( 'Basic Data Generated, comparing datatypes: GC')
        
        ### Spectra vs GC Variation
        data.pcaGC = PCA( n_components=11 )  # It uses the LAPACK implementation 
        # of the full SVD or a randomized truncated SVD by the method of Halko 
        # et al. 2009, depending on the shape of the input data and the number 
        # of components to extract.
        data.pcaGC.fit(np.transpose(data.butter_profile))
        data.GCsc=data.pcaGC.transform(np.transpose(data.butter_profile))

        return data

