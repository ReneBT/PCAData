# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:44:41 2023

@author: rene
"""
import numpy as np
from copulas.multivariate import GaussianMultivariate
from sklearn.decomposition import PCA

### shot noise    
def  gen_data(
    data,
    ):
### Cage of Covariance
#exploring the impact of perturbing the cage of covariance on model applicability
# need to have a test set with same cage then variations on the cage

# Butter validation set - same cage of covariance
# determine copula of original butter GC and use to generate a new test set
        copula = GaussianMultivariate()
        copula.fit(data.butter_profile.T)
        data.butter_profile_Val = np.array(copula.sample(100).T)
        data.spectra_butter_Val = np.dot(
            data.simulated_FA,data.butter_profile_Val)

# Butter with no covariance at all
        #create a data.butter_profile with no correlation
        data.FAmeans = np.mean(data.butter_profile,1)
        data.FAsd = np.std(data.butter_profile,1)
        # generate random numbers in array same size as data.butter_profile, scale by 1data.FAsd then add on
        # mean value
        data.butter_profile_NoCov = np.random.randn(*data.butter_profile.shape)
        data.butter_profile_NoCov = (data.butter_profile_NoCov.T*data.FAsd)+data.FAmeans
        data.butter_profile_NoCov = 100*data.butter_profile_NoCov.T/np.sum(data.butter_profile_NoCov,axis=1)

        data.spectraNoCov = np.dot(data.simulated_FA,data.butter_profile_NoCov)

# Butter with imputed Adipose covariance
        data.PCA_GC_Adipose = PCA(n_components=10,) #use of copula method retains central as well as marginal distribution covariances, so rebuild using covariance in PCA. This will mean FA means not exactly replicated as covariance constrains possible
        data.meanFA_adipose = np.mean(data.adipose_profile, axis=1) # mean centre for model
        data.PCA_GC_Adipose.fit(data.adipose_profile.T-data.meanFA_adipose)
#        data.butter_profileAdipose = (np.inner( np.inner(
#            data.butter_profile.T-data.meanFA_adipose, data.PCA_GC_Adipose.components_ ),
#            data.PCA_GC_Adipose.components_.T ) +data.meanFA_adipose).T
        data.butter_profileAdipose = (np.inner( np.inner(
            data.butter_profile.T-data.FAmeans, data.PCA_GC_Adipose.components_ ),
            data.PCA_GC_Adipose.components_.T ) +data.FAmeans).T
        data.spectra_butter_adiposeCov = np.dot(data.simulated_FA,data.butter_profileAdipose)

# also compare with adipose - this is an independent experiment, using a
# different lipid product (adipose vs milk derived butter), although one group
# is the same species (beef and butter both from cows)

        data.FA_profiles_sanity = np.vstack([ data.FAmeans,
                                        np.mean(data.butter_profile_Val,axis=1),
                                        np.mean(data.butter_profile_NoCov,axis=1),
                                        np.mean(data.butter_profileAdipose,axis=1),
                                        np.mean(data.adipose_profile,axis=1),
                                        ])
# apply butter model to each alternative
        data.TestSet_Train = data.pcaMC.project_model(data.data)
        data.TestSet_Val = data.pcaMC.project_model(data.spectra_butter_Val)
        data.TestSet_NoCov = data.pcaMC.project_model(data.spectraNoCov)
        data.TestSet_adiposeCov = data.pcaMC.project_model(data.spectra_butter_adiposeCov)
        data.TestSet_adipose = data.pcaMC.project_model(data.adipose_data)
        
        return data

