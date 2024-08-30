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
            axGCscores[0].plot(pcaMC.component_weight[ix, 0],pcaMC.component_weight[ix, 1],grps[2][iS],fillstyle=fillS)
            axGCscores[1].plot(-GCsc[ix, 0],-GCsc[ix, 1],grps[2][iS],fillstyle=fillS)
            axGCscores[2].plot(-GCsc[ix, 0],pcaMC.component_weight[ix, 0],grps[2][iS],fillstyle=fillS)
        axGCscores[0].set_xlabel('t[1]Spectral',labelpad=-1)
        axGCscores[0].set_ylabel('t[2]Spectral',labelpad=-1)
        axGCscores[0].legend(['0mg E','0mg L','2mg','_','4mg','_','6mg'],
                              fontsize=pcaMC.fig_Text_Size*0.65,framealpha=0.5,
                              borderpad=0.2)
        axGCscores[1].set_xlabel('t[1]GC',labelpad=-1)
        axGCscores[1].set_ylabel('t[2]GC',labelpad=-1)
        axGCscores[2].set_xlabel('t[1]GC',labelpad=-1)
        axGCscores[2].set_ylabel('t[1]Spectral',labelpad=-1)
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

        SpecGCLoad = np.inner(butter_profile,pcaMC.component_weight[:,:11].T)
        SpecGCLoad = (SpecGCLoad/np.sum(SpecGCLoad**2,axis=0)**0.5)
        lSOff = np.hstack((0,np.cumsum(np.floor(np.min(SpecGCLoad[:,0:2]-SpecGCLoad[:,1:3],axis=0)*20)/20)))
        xval = np.empty(27)
        for i in range(N_FA):
            xval[i] = GC_data["N_Carbon"][i][0] + GC_data["N_Isomer"][i][0]/4 + GC_data["N_Olefin"][i][0]/40
        xIx = np.argsort(xval)
        axGCdata[0,1].plot(np.arange(len(xIx)),[-1,-1,1]*pcaGC.components_[0:3, xIx].T+lSOff,'k')
        axGCdata[0,1].plot(np.arange(len(xIx)),SpecGCLoad[xIx,0:3]+lSOff,'--',color=[0.6,0.6,0.6])
        axGCdata[0,1].plot(axGCdata[0,1].get_xlim(),np.tile(lSOff,(2,1)),'--',color=[0.8,0.8,0.8])
        axGCdata[0,1].set_xticks(range(len(xIx)))
        axGCdata[0,1].set_xticklabels(labels=FAnames,rotation=90, fontsize=8)
        axGCdata[0,1].set_xlim([0,len(xIx)-1])



        axGCdata[1,0].plot(-GCSpecLoad[:,0],pcaMC.spectral_loading[0,:],'.k',markersize=5)
        axGCdata[1,0].plot(-GCSpecLoad[:,1],pcaMC.spectral_loading[1,:],'+',color=[0.5,0.5,0.5],markersize=5)
        axGCdata[1,0].plot(GCSpecLoad[:,2],pcaMC.spectral_loading[2,:],'<',color=[0.8,0.8,0.8],markersize=5)
        xran = (axGCdata[1,0].get_xlim()[0]*0.95,axGCdata[1,0].get_xlim()[1]*0.95)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(-GCSpecLoad[:,0].T,pcaMC.spectral_loading[0,:],1))(xran),'--k',lw=0.5)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(-GCSpecLoad[:,1].T,pcaMC.spectral_loading[1,:],1))(xran),'--',color=[0.5,0.5,0.5],lw=0.5)
        axGCdata[1,0].plot(xran,np.poly1d(np.polyfit(GCSpecLoad[:,2].T,pcaMC.spectral_loading[2,:],1))(xran),'--',color=[0.8,0.8,0.8],lw=0.5)

        axGCdata[1,1].plot(-pcaGC.components_[0, :],SpecGCLoad[:,0],'.k')
        axGCdata[1,1].plot(-pcaGC.components_[1, :],SpecGCLoad[:,1],'+',color=[0.5,0.5,0.5])
        axGCdata[1,1].plot(pcaGC.components_[2, :],SpecGCLoad[:,2],'<',color=[0.8,0.8,0.8])
        xran = (axGCdata[1,1].get_xlim()[0]*0.95,axGCdata[1,1].get_xlim()[1]*0.95)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(-pcaGC.components_[0,:],SpecGCLoad[:,0],1))(xran),'--k',lw=0.5)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(-pcaGC.components_[1,:],SpecGCLoad[:,1],1))(xran),'--',color=[0.5,0.5,0.5],lw=0.5)
        axGCdata[1,1].plot(xran,np.poly1d(np.polyfit(pcaGC.components_[2,:],SpecGCLoad[:,2],1))(xran),'--',color=[0.8,0.8,0.8],lw=0.5)


        axGCdata[0,0].set_xlabel('Raman Shift (cm$^{-1}$)',labelpad=-1)
        axGCdata[0,0].set_ylabel('W',labelpad=-1)
        axGCdata[0,1].set_ylabel('W',labelpad=-1)
        axGCdata[1,0].set_xlabel('W spectral',labelpad=-1)
        axGCdata[1,0].set_ylabel('W GCX',labelpad=-1)

        axGCdata[1,1].set_xlabel('W GC',labelpad=-1)
        axGCdata[1,1].set_ylabel('W SpectralX',labelpad=-1)

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
