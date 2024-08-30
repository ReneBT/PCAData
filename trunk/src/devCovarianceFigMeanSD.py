        figVal, axVal = plt.subplots(2, 2, figsize=pcaMC.fig_Size)
        axVal[0,0] = plt.subplot2grid((23, 11), (0, 0), colspan=5, rowspan=10)
        axVal[0,1] = plt.subplot2grid((23, 11), (0, 6), colspan=5, rowspan=10)
        axVal[1,0] = plt.subplot2grid((23, 11), (13, 0), colspan=5, rowspan=10)
        axVal[1,1] = plt.subplot2grid((23, 11), (13, 6), colspan=5, rowspan=10)
        figVal.subplots_adjust(left=0.08,right=0.98,top=0.94,bottom=0.09)

        # Generated GC profiles
        axVal[0,0].plot(np.arange(0,27)-0.2 , adipose_profile,'dg', markersize=2)
        axVal[0,0].plot(np.arange(0,27)-0.1 , butter_profileAdipose,'^c',alpha=0.5, markersize=2)
        axVal[0,0].plot(np.arange(0,27) , butter_profileUncorr,'*b',alpha=0.5, markersize=2)
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
        truncTicks = np.concatenate([np.arange(0,101,50),np.arange(180,271,50),np.arange(400,461,50)])#,np.arange(490,541,50)])
        tempDat = np.mean(adipose_data[trunc,:],axis=1)
        tempIx = np.concatenate([np.arange(96,106),np.arange(187,197)])#,np.arange(247,257)])
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'g')
        tempDat = np.mean(spectra_butter_Val[trunc,:],axis=1)
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'r')
        tempDat = np.mean(spectra_butter_adiposeCov[trunc,:],axis=1)
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'c')
        tempDat = np.mean(spectraNoCov[trunc,:],axis=1)
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'--b')

        tempDat = 2*np.std(adipose_data[trunc,:],axis=1)-200
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'g')
        tempDat = 2*np.std(spectra_butter_Val[trunc,:],axis=1)-200
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'r')
        tempDat = 2*np.std(spectra_butter_adiposeCov[trunc,:],axis=1)-200
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'c')
        tempDat = 2*np.std(spectraNoCov[trunc,:],axis=1)-200
        tempDat[tempIx] = np.nan
        axVal[0,1].plot(tempDat,'--b')
        truncTicks = np.concatenate([np.arange(0,96,25),[95],np.arange(106,180,25),[186],np.arange(197,247,25),[246]])#,np.arange(257,304,25),[303]])
        axVal[0,1].set_xticks(truncTicks)
        axVal[0,1].set_xticklabels(wavelength_axis[trunc[truncTicks]], rotation=90)

