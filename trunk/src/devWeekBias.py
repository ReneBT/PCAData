# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:44:41 2023

@author: rene
"""

# simulate non-random biases that may influence fold selection
x=np.arange(601)+1
backs = np.column_stack( (
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
simParams_weekly = np.vstack( ( (np.ones( ( 2, 88 ) ).T*np.array([540,380]).T).T, 
                               np.tile( np.vstack( ( temp[0,:], temp[1,:],
                                                    temp[2,:], temp[3,:],  
                                                    temp[4,:])),(1,11)),
                               )
                             )

simParams_weekly[ simParams_weekly<0 ] = 0 # no negative contributions
simParams_Sample = np.vstack( (  
                               stat.norm.rvs( 1 , 0.02 , size=( 1,88 ) ),
                               stat.norm.rvs( 1 , 0.018 , size=( 1,88 ) ),
                               stat.norm.rvs( 1 , 0.007 , size=( 1,88 ) ),
                               stat.norm.rvs( 1 , 0.006 , size=( 1,88 ) ),
                               stat.norm.rvs( 1 , 0.004 , size=( 1,88 ) ),
                               stat.norm.rvs( 1 , 0.005 , size=( 1,88 ) ),
                               stat.norm.rvs( 1 , 0.002 , size=( 1,88 ) ),
                               )
                             ) # individual fluctuations by sample
simParams_Sample[ simParams_Sample<0 ] = 0 # no negative contributions
simParams = simParams_Sample*simParams_weekly

week_bias_backs = np.inner(backs,simParams.T)
week_bias_data = data + week_bias_backs
week_bias_noise = (week_bias_data**0.5) * shot_noise/4 #noise scales by square root of intensity - use 100 offset so baseline not close to zero
week_bias_data_noise =  week_bias_data + week_bias_noise
week_clean_data_noise =  data + week_bias_noise #noise affected by backs, but no baseline distoritions on spectra

week_bias_data_airpls = week_bias_data_noise / 0
for iS in range( 88 ):
    week_bias_data_airpls[ :, iS ] = ( week_bias_data_noise[ :, iS ] -
                                      air.airPLS( week_bias_data_noise[ :, iS ],
                                                 lambda_=9 ) )
#plt.plot(week_bias_data_airpls)
# do different train/test scenarios:
    # 1. non-random week selection - use case is testing if model has reached stable state against week
    # 3. random selection
select_non_random_week = np.where(week<18)
test_non_random_week = np.where(week>16)
randIx = np.argsort(stat.norm.rvs( 1 , 1 , size=( 1,88 ),
                                  random_state=np.random.RandomState(seed=92867)))
discard, randSel = np.where(randIx>(len(test_non_random_week[0])-1))
discard, randSelTest = np.where(randIx<(len(test_non_random_week[0])))
biasFeedSel = np.where(feed<12)
biasFeedSelTest = np.where(feed>11)
nPCs = 4
pca_week_Bias_airPLS_train = npls.nipals(
    X_data=week_bias_data_airpls[:,select_non_random_week[0]],
    maximum_number_PCs=nPCs,
    maximum_iterations_PCs=100,
    iteration_tolerance=0.000000000001,
    preproc="MC",
    pixel_axis=wavelength_axis,
    spectral_weights=butter_profile,
    min_spectral_values=min_data,
)
pca_week_Bias_airPLS_train.calc_PCA()
pca_week_Bias_airPLS_train_project = pca_week_Bias_airPLS_train.project_model( week_bias_data_airpls[:,select_non_random_week[0]] )
pca_week_Bias_airPLS_test = pca_week_Bias_airPLS_train.project_model( week_bias_data_airpls[:,test_non_random_week[0]] )

pca_week_Bias_clean_train = npls.nipals(
    X_data=week_clean_data_noise[:,select_non_random_week[0]],
    maximum_number_PCs=nPCs,
    maximum_iterations_PCs=100,
    iteration_tolerance=0.000000000001,
    preproc="MC",
    pixel_axis=wavelength_axis,
    spectral_weights=butter_profile,
    min_spectral_values=min_data,
)
pca_week_Bias_clean_train.calc_PCA()
pca_week_Bias_clean_train_project = pca_week_Bias_clean_train.project_model( week_clean_data_noise[:,select_non_random_week[0]] )
pca_week_Bias_clean_test = pca_week_Bias_clean_train.project_model( week_clean_data_noise[:,test_non_random_week[0]] )

pca_week_Rand_airPLS_train = npls.nipals(
    X_data=week_bias_data_airpls[:,randSel],
    maximum_number_PCs=nPCs,
    maximum_iterations_PCs=100,
    iteration_tolerance=0.000000000001,
    preproc="MC",
    pixel_axis=wavelength_axis,
    spectral_weights=butter_profile,
    min_spectral_values=min_data,
)
pca_week_Rand_airPLS_train.calc_PCA()
pca_week_Rand_airPLS_train_project = pca_week_Rand_airPLS_train.project_model( week_bias_data_airpls[:,randSel] )
pca_week_Rand_airPLS_test = pca_week_Rand_airPLS_train.project_model( week_bias_data_airpls[:,randSelTest] )

pca_feed_bias_train = npls.nipals(
    X_data=week_bias_data_airpls[:,biasFeedSel[0]],
    maximum_number_PCs=nPCs,
    maximum_iterations_PCs=100,
    iteration_tolerance=1e-12,
    preproc="MC",
    pixel_axis=wavelength_axis,
    spectral_weights=butter_profile,
    min_spectral_values=min_data,
)
pca_feed_bias_train.calc_PCA()
pca_feed_bias_train_project = pca_feed_bias_train.project_model( week_bias_data_airpls[:,biasFeedSel[0]]  )
pca_feed_bias_test = pca_feed_bias_train.project_model( week_bias_data_airpls[:,biasFeedSelTest[0]] )

pca_clean_rand_train = npls.nipals(
    X_data=week_clean_data_noise[:,randSel],
    maximum_number_PCs=nPCs,
    maximum_iterations_PCs=100,
    iteration_tolerance=1e-12,
    preproc="MC",
    pixel_axis=wavelength_axis,
    spectral_weights=butter_profile,
    min_spectral_values=min_data,
)
pca_clean_rand_train.calc_PCA()
pca_clean_rand_train_project = pca_clean_rand_train.project_model( week_clean_data_noise[:,randSel]  )
pca_clean_rand_test = pca_clean_rand_train.project_model( week_clean_data_noise[:,randSelTest] )

pca_feed_bias_clean_train = npls.nipals(
    X_data=week_clean_data_noise[:,biasFeedSel[0]],
    maximum_number_PCs=nPCs,
    maximum_iterations_PCs=100,
    iteration_tolerance=1e-12,
    preproc="MC",
    pixel_axis=wavelength_axis,
    spectral_weights=butter_profile,
    min_spectral_values=min_data,
)
pca_feed_bias_clean_train.calc_PCA()
pca_feed_bias_clean_train_project = pca_feed_bias_clean_train.project_model( week_clean_data_noise[:,biasFeedSel[0]]  )
pca_feed_bias_clean_test = pca_feed_bias_clean_train.project_model( week_clean_data_noise[:,biasFeedSelTest[0]] )

plt.plot( np.vstack( (
    np.mean(np.abs(pca_week_Bias_airPLS_test["residual"]),axis=0), 
    np.mean(np.abs(pca_feed_bias_test["residual"]),axis=0), 
    np.mean(np.abs(pca_week_Rand_airPLS_test["residual"]),axis=0), 
    np.mean(np.abs(pca_clean_rand_test["residual"]),axis=0), 
    np.mean(np.abs(pca_week_Bias_clean_test["residual"]),axis=0), 
    np.mean(np.abs(pca_feed_bias_clean_test["residual"]),axis=0), 
    )).T)
plt.legend(('Week Bias','Feed Bias','Random','Clean Random','Clean Week Bias','Clean Feed Bias'))

plt.plot( np.vstack( ( 
    np.mean(np.abs(pca_feed_bias_clean_test["residual"]),axis=0), 
    np.mean(np.abs(pca_clean_rand_test["residual"]),axis=0), 
    np.mean(np.abs(pca_week_Bias_clean_test["residual"]),axis=0))).T)

figBias, axBias = plt.subplots(1, 2, figsize=pcaMC.fig_Size)
axBias[0] = plt.subplot2grid((1, 11), (0, 0), colspan=5)
axBias[1] = plt.subplot2grid((1, 11), (0, 6), colspan=5)
axBias[0].plot( wavelength_axis ,
               week_clean_data_noise  , linewidth = 0.75, color=[0,0,0.7])
axBias[0].plot( wavelength_axis ,
               week_bias_data_airpls  , linewidth = 0.75, color=[0.7,0,0])
axBias[0].legend(('Clean Data','Dirty Data'))
axBias[1].plot( wavelength_axis ,
               np.vstack( (
                   np.mean(np.abs(pca_week_Bias_airPLS_test["residual"]),axis=0), 
                   np.mean(np.abs(pca_feed_bias_test["residual"]),axis=0), 
                   np.mean(np.abs(pca_week_Rand_airPLS_test["residual"]),axis=0), 
                   np.mean(np.abs(pca_week_Bias_clean_test["residual"]),axis=0), 
                   np.mean(np.abs(pca_feed_bias_clean_test["residual"]),axis=0), 
                   np.mean(np.abs(pca_clean_rand_test["residual"]),axis=0), 
                   )).T , linewidth = 0.75)
axBias[1].legend(('Dirty Bias:Week','Dirty Bias:Feed','Dirty Random','Clean Bias:Week','Clean Bias:Feed','Clean Unbiased'))
axBias[0].set_ylabel("Intensity / Counts")
axBias[0].set_xlabel("Raman Shift cm$^{-1}$")
axBias[1].set_ylabel("Residual Intensity / Counts")
axBias[1].set_xlabel("Raman Shift cm$^{-1}$")
image_name = " Selection Residuals."
full_path = os.path.join(images_folder, pcaMC.fig_Project +
                        image_name + pcaMC.fig_Format)
plt.savefig(full_path,
                 dpi=pcaMC.fig_Resolution)        # plt.show()
plt.close()
