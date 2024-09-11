# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:44:41 2023

@author: rene
"""
import numpy as np
import scipy.stats as stat
import src.airPLS as air # https://github.com/zmzhang/airPLS
import src.local_nipals as npls

### class dirty_data    
def  gen_data(
    data,
):
    x=np.arange(601)+1
    data.backs = np.column_stack( (
        (-4.482e-18*10**(np.log10(x)*7) + 1.135e-14*10**(np.log10(x)*6) 
        - 1.098e-11*10**(np.log10(x)*5) + 5.222e-09*10**(np.log10(x)*4) 
        - 1.328e-6*10**(np.log10(x)*3) + 0.0001865*10**(np.log10(x)*2) 
        - 0.01498*x + 1.836),
        (-4.523e-18*10**(np.log10(x)*7) + 1.311e-14*10**(np.log10(x)*6) 
        -1.393e-11*10**(np.log10(x)*5) + 7.157e-09*10**(np.log10(x)*4) 
        -1.943e-06*10**(np.log10(x)*3) + 0.0002813*10**(np.log10(x)*2) 
        -0.02051*x + 1.842),
        (7.5212e-19*10**(np.log10(x)*7) - 6.518e-15*10**(np.log10(x)*6) 
        + 1.016e-11*10**(np.log10(x)*5) - 6.676e-09*10**(np.log10(x)*4) 
        + 2.178e-06*10**(np.log10(x)*3) - 0.0003526*10**(np.log10(x)*2) 
        +0.02651*x -0.08497),
        (-2.888e-17*10**(np.log10(x)*7) + 6.28e-14*10**(np.log10(x)*6) 
        -5.177e-11*10**(np.log10(x)*5) + 2.023e-08*10**(np.log10(x)*4) 
        -3.929e-06*10**(np.log10(x)*3) + 0.0003916*10**(np.log10(x)*2) 
        -0.01642*x + 0.1990),
        (-2.165e-17*10**(np.log10(x)*7) + 4.482e-14*10**(np.log10(x)*6) 
        -3.566e-11*10**(np.log10(x)*5) + 1.366e-08*10**(np.log10(x)*4) 
        -2.654e-06*10**(np.log10(x)*3) + 0.0002847*10**(np.log10(x)*2) 
        -0.02335*x + 2.265),
        0.8*np.exp((-(x-75)**2)/(12)**2)+np.exp((-(x-131)**2)/(20)**2)*0.4, # optics signal
        stat.norm.rvs( 0, 1, size=( 601,1 ))    
        ) )
    # define the weekly bias due to operational variation - first 3 solely by sample variation, 2nd 3 interact between week and sample
    temp = np.vstack( ( stat.norm.rvs( 280 , 120 , size=( 1,8 ) ),
                                   np.arange(450,169,-40),
                                   np.arange(3,18,2)**2, # directional bias - some mechanical drift in instrument, affecting stray light collection
                                   np.arange(800,299,-70)**0.5,
                                   np.arange(100,801,100)**0.5,
                                   )
                                 )
    data.simParams_weekly = np.vstack( ( (np.ones( ( 2, 88 ) ).T*np.array([540,380]).T).T, 
                                   np.tile( np.vstack( ( temp[0,:], temp[1,:],
                                                        temp[2,:], temp[3,:],  
                                                        temp[4,:])),(1,11)),
                                   )
                                 )
    
    data.simParams_weekly[ data.simParams_weekly<0 ] = 0 # no negative contributions
    data.simParams_Sample = np.vstack( (  
                                   stat.norm.rvs( 1 , 0.02 , size=( 1,88 ) ),
                                   stat.norm.rvs( 1 , 0.018 , size=( 1,88 ) ),
                                   stat.norm.rvs( 1 , 0.007 , size=( 1,88 ) ),
                                   stat.norm.rvs( 1 , 0.006 , size=( 1,88 ) ),
                                   stat.norm.rvs( 1 , 0.004 , size=( 1,88 ) ),
                                   stat.norm.rvs( 1 , 0.005 , size=( 1,88 ) ),
                                   stat.norm.rvs( 1 , 0.002 , size=( 1,88 ) ),
                                   )
                                 ) # individual fluctuations by sample
    data.simParams_Sample[ data.simParams_Sample<0 ] = 0 # no negative contributions
    data.simParams = data.simParams_Sample*data.simParams_weekly
    
    data.week_bias_backs = np.inner(data.backs,data.simParams.T)
    data.week_bias_data = data.data + data.week_bias_backs
    data.week_bias_noise = (data.week_bias_data**0.5) * data.shot_noise/4 #noise scales by square root of intensity - use 100 offset so baseline not close to zero
    data.week_bias_data_noise =  data.week_bias_data + data.week_bias_noise
    data.week_clean_data_noise =  data.data + data.week_bias_noise #noise affected by data.backs, but no baseline distoritions on spectra


    data.week_bias_data_airpls = data.week_bias_data_noise / 0
    for iS in range( 88 ):
        data.week_bias_data_airpls[ :, iS ] = ( data.week_bias_data_noise[ :, iS ] -
                                          air.airPLS( data.week_bias_data_noise[ :, iS ],
                                                     lambda_=9 ) )
    #plt.plot(data.week_bias_data_airpls)
    # do different train/test scenarios:
        # 1. non-random week selection - use case is testing if model has reached stable state against week
        # 3. random selection
    data.select_non_random_week = np.where(data.week<18)
    test_non_random_week = np.where(data.week>16)
    randIx = np.argsort(stat.norm.rvs( 1 , 1 , size=( 1,88 ),
                                      random_state=np.random.RandomState(seed=92867)))
    discard, randSel = np.where(randIx>(len(test_non_random_week[0])-1))
    discard, randSelTest = np.where(randIx<(len(test_non_random_week[0])))
    data.biasFeedSel = np.where(data.feed<12)
    data.biasFeedSelTest = np.where(data.feed>11)
    nPCs = 4
    data.pca_week_Bias_airPLS_train = npls.nipals(
        X_data=data.week_bias_data_airpls[:,data.select_non_random_week[0]],
        maximum_number_PCs=nPCs,
        maximum_iterations_PCs=100,
        iteration_tolerance=0.000000000001,
        preproc="MC",
        pixel_axis=data.wavelength_axis,
        spectral_weights=data.butter_profile,
        min_spectral_values=data.min_data,
    )
    data.pca_week_Bias_airPLS_train.calc_PCA()
    data.pca_week_Bias_airPLS_train_project = data.pca_week_Bias_airPLS_train.project_model( data.week_bias_data_airpls[:,data.select_non_random_week[0]] )
    data.pca_week_Bias_airPLS_test = data.pca_week_Bias_airPLS_train.project_model( data.week_bias_data_airpls[:,test_non_random_week[0]] )
    
    data.pca_week_Bias_clean_train = npls.nipals(
        X_data=data.week_clean_data_noise[:,data.select_non_random_week[0]],
        maximum_number_PCs=nPCs,
        maximum_iterations_PCs=100,
        iteration_tolerance=0.000000000001,
        preproc="MC",
        pixel_axis=data.wavelength_axis,
        spectral_weights=data.butter_profile,
        min_spectral_values=data.min_data,
    )
    data.pca_week_Bias_clean_train.calc_PCA()
    data.pca_week_Bias_clean_train_project = data.pca_week_Bias_clean_train.project_model( data.week_clean_data_noise[:,data.select_non_random_week[0]] )
    data.pca_week_Bias_clean_test = data.pca_week_Bias_clean_train.project_model( data.week_clean_data_noise[:,test_non_random_week[0]] )
    
    data.pca_week_Rand_airPLS_train = npls.nipals(
        X_data=data.week_bias_data_airpls[:,randSel],
        maximum_number_PCs=nPCs,
        maximum_iterations_PCs=100,
        iteration_tolerance=0.000000000001,
        preproc="MC",
        pixel_axis=data.wavelength_axis,
        spectral_weights=data.butter_profile,
        min_spectral_values=data.min_data,
    )
    data.pca_week_Rand_airPLS_train.calc_PCA()
    data.pca_week_Rand_airPLS_train_project = data.pca_week_Rand_airPLS_train.project_model( data.week_bias_data_airpls[:,randSel] )
    data.pca_week_Rand_airPLS_test = data.pca_week_Rand_airPLS_train.project_model( data.week_bias_data_airpls[:,randSelTest] )
    
    data.pca_feed_bias_train = npls.nipals(
        X_data=data.week_bias_data_airpls[:,data.biasFeedSel[0]],
        maximum_number_PCs=nPCs,
        maximum_iterations_PCs=100,
        iteration_tolerance=1e-12,
        preproc="MC",
        pixel_axis=data.wavelength_axis,
        spectral_weights=data.butter_profile,
        min_spectral_values=data.min_data,
    )
    data.pca_feed_bias_train.calc_PCA()
    data.pca_feed_bias_train_project = data.pca_feed_bias_train.project_model( data.week_bias_data_airpls[:,data.biasFeedSel[0]]  )
    data.pca_feed_bias_test = data.pca_feed_bias_train.project_model( data.week_bias_data_airpls[:,data.biasFeedSelTest[0]] )
    
    data.pca_clean_rand_train = npls.nipals(
        X_data=data.week_clean_data_noise[:,randSel],
        maximum_number_PCs=nPCs,
        maximum_iterations_PCs=100,
        iteration_tolerance=1e-12,
        preproc="MC",
        pixel_axis=data.wavelength_axis,
        spectral_weights=data.butter_profile,
        min_spectral_values=data.min_data,
    )
    data.pca_clean_rand_train.calc_PCA()
    data.pca_clean_rand_train_project = data.pca_clean_rand_train.project_model( data.week_clean_data_noise[:,randSel]  )
    data.pca_clean_rand_test = data.pca_clean_rand_train.project_model( data.week_clean_data_noise[:,randSelTest] )
    
    data.pca_feed_bias_clean_train = npls.nipals(
        X_data=data.week_clean_data_noise[:,data.biasFeedSel[0]],
        maximum_number_PCs=nPCs,
        maximum_iterations_PCs=100,
        iteration_tolerance=1e-12,
        preproc="MC",
        pixel_axis=data.wavelength_axis,
        spectral_weights=data.butter_profile,
        min_spectral_values=data.min_data,
    )
    data.pca_feed_bias_clean_train.calc_PCA()
    data.pca_feed_bias_clean_train_project = data.pca_feed_bias_clean_train.project_model( data.week_clean_data_noise[:,data.biasFeedSel[0]]  )
    data.pca_feed_bias_clean_test = data.pca_feed_bias_clean_train.project_model( data.week_clean_data_noise[:,data.biasFeedSelTest[0]] )

    return data

