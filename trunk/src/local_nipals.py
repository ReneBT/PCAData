import numpy as np
import matplotlib.pyplot as plt
import os
#from matplotlib.patches import ConnectionPatch
from matplotlib import transforms, patches

#from pathlib import Path

# This expects to be called inside the jupyter project folder structure.
from src.file_locations import images_folder


### class nipals
class nipals:
# 3 hash comments with no indent are intended to assist navigation in spyder IDE
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned
    # along the columns and each row corresponds to different variables

    # comments include references to relevant lines in the pseudocode listed in the paper

### __init__
    def __init__(
        self,
        X_data,
        maximum_number_PCs,
        maximum_iterations_PCs=100,
        iteration_tolerance=0.000000000001,
        preproc="None",
        pixel_axis="None",
        spectral_weights=None,
        min_spectral_values=None,
    ):
        # requires a data matrX_data and optional additional settings
        # X_data      X, main data matrX_data as specified in pseudocode
        # maximum_number_PCs    is max_i, desired maximum number of PCs
        # maximum_iterations_PCs   is max_j, maximum iterations on each PC
        # iteration_tolerance    is tol, acceptable tolerance for difference between j and j-1 estimates of the PCs
        # preproc is defining preprocessing stpng desired prior to PCA. Current options are
        #                     mean centering ('MC')
        #                     median centering ('MdnC')
        #                     scaling to unit variance ('UV')
        #                     scaling to range ('MinMax')
        #                     scaling to largest absolute value ('MaxAbs')
        #                     combining these (e.g.'MCUV').
        #            if other types are desired please handle these prior to calling this class
        #            Note that only one centering method and one scaling method will be implmented. If more are present then the
        #            implmented one will be the first in the above list.
        # self.pixel_axis    is a vector of values for each pixel in the x-axis or a tuple specifying a fixed interval spacing
        #         (2 values for unit spacing, 3 values for non-unit spacing)
        # spectral_weights  is the array of the weights used to combine the simulated reference spectra into the simulated sample spectra

        if X_data is not None:
            if "MC" in preproc:
                self.centring = np.mean(X_data, 1)  # calculate the mean of the data
                self.mean = 1
            elif "MdnC" in preproc:
                self.centring = np.median(X_data, 1)  # calculate the mean of the data
                self.mean = 2
            else:
                self.centring = np.zeros(X_data.shape[0])  # calculate the mean of the data
                self.mean = 0
            X_data = (
                X_data.T - self.centring
            ).T  # mean centre data

            if "UV" in preproc:
                self.scale = X_data.std(1)
            elif "MinMax" in preproc:
                self.scale = X_data.max(1) - X_data.min(1)
            elif "MaxAbs" in preproc:
                self.scale = abs(X_data).max(1)
            else:
                self.scale = 1
            X_data = (X_data.T / self.scale).T

            self.data = X_data
            self.N_Vars, self.N_Obs = (
                X_data.shape
            )  # calculate v (N_Vars) and o (N_Obs), the number of variables and observations in the data matrX_data
        else:
            print("No data provided")
            return

        # maximum_number_PCs is the maximum number of principle components to calculate
        if maximum_number_PCs is not None:
            #            print(type(maximum_number_PCs))
            self.N_PC = min(
                X_data.shape[0], X_data.shape[1], maximum_number_PCs
            )  # ensure max_i is achievable (minimum of v,o and max_i)

        else:
            self.N_PC = min(X_data.shape[0], X_data.shape[1])

        self.positive_sum = np.empty([self.N_Vars, self.N_PC])
        self.positive_sum[:] = np.nan
        self.negative_sum = np.copy(self.positive_sum)
        self.orthogonal_sum = np.copy(self.positive_sum)
        self.global_minimum = np.copy(self.positive_sum)
        self.positive_sumS = np.copy(self.positive_sum)
        self.negative_sumS = np.copy(self.positive_sum)
        self.orthogonal_sumS = np.copy(self.positive_sum)
        self.global_minimumS = np.copy(self.positive_sum)
        self.optSF = np.empty(self.N_PC)
        self.optSF[:] = np.nan
        self.Eigenvalue = np.copy(self.optSF)

        self.fig_Size = [8,5]
        self.fig_Resolution = 800
        self.fig_Format = 'png'
        self.fig_k = range( 0 , self.N_Obs , np.ceil(self.N_Obs/10).astype(int) )
        self.fig_i = range( 0 , np.min((self.N_PC , 5 ) ) )
        self.fig_Y_Label = ''
        self.fig_X_Label = ''
        self.fig_Show_Labels =  False
        self.fig_Show_Values =  False
        self.fig_Text_Size =  10
        self.fig_Project = 'Graphical PCA Demo'


        # maximum_iterations_PCs is the maximum number of iterations to perform for each PC
        if maximum_iterations_PCs is not None:
            self.Max_It = maximum_iterations_PCs

            # iteration_tolerance is the tolerance for decieding if subsequent iterations are significantly reducing the residual variance
        if iteration_tolerance is not None:
            #           print('iteration_tolerance not Empty')
            self.Tol = iteration_tolerance
        else:
            print("iteration_tolerance Empty")

        if pixel_axis is not None:
            self.pixel_axis = pixel_axis
        else:
            self.pixel_axis = list(range(self.N_Vars))

        self.spectral_loading = np.ndarray(
            (self.N_PC, self.N_Vars)
        )  # initialise array for right eigenvector (loadings/principal components/latent factors for column oriented matrices, note is transposed wrt original data)
        self.component_weight = np.ndarray(
            (self.N_Obs, self.N_PC)
        )  # initialise array for left eigenvector (scores for column oriented matrices)
        self.icomponent_weight = np.copy(
                self.component_weight.T
                )
        self.r = (
            []
        )  # initialise list for residuals for each PC (0 will be initial data, so indexing will equal the PC number)
        self.pc = (
            []
        )  # initialise list for iterations on the right eigenvectors for each PC (0 will be initial data, so indexing will equal the PC number)
        self.w = (
            []
        )  # initialise list for iterations on the left eigenvectors for each PC (0 will be initial data, so indexing will equal the PC number)
        self.rE = np.ndarray(
            (self.N_PC + 1, self.N_Vars)
        )  # residual error at each pixel for each PC (0 will be initial data, so indexing will equal the PC number)
        self.spectral_weights = spectral_weights
        self.min_spectral_values = min_spectral_values

        return

### figure_Settings
    def figure_Settings(
        self,
        size=None,
        res=None,
        frmt=None,
        k=None,
        i=None,
        Ylab=None,
        Xlab=None,
        Show_Labels=None,
        Show_Values=None,
        TxtSz=None,
        Project=None
    ):
        # NOTE that the user is expected to update all settings in one call, otherwise excluded settings will revert to default
        if self.data is None:
                print('Data must be loaded into object in order to define plot parameters')
                return

        if size is not None:
            if np.ndim(size) == 1 and np.shape(size)[0] == 2:
                self.fig_Size = size
            else:
                print('Size should be a two value vector, reverting to default size of 8 x 8 cm')
                self.fig_Size = [ 8 , 5 ]
        else:
            self.fig_Size = [ 8 , 5 ]

        if res is not None:
            if np.ndim(res) == 0:
                self.fig_Resolution = res
            else:
                print('Resolution should be a scalar, reverting to default resolution of 800 dpi')
                self.fig_Resolution = 800
        else:
            self.fig_Resolution = 800

        # dynamically get list of supported formats, based on accepted answer at https://stackoverflow.com/questions/7608066/in-matplotlib-is-there-a-way-to-know-the-list-of-available-output-format
        # class has imported matplotlib so shouldn't need imported as in the answer
        fig = plt.figure()
        availFrmt = fig.canvas.get_supported_filetypes()
        if frmt is not None:
            if frmt in availFrmt:
                self.fig_Format = frmt
            else:
                print('Format string not recognised, reverting to default of png')
                self.fig_Format = 'png'
        else:
            self.fig_Format = 'png'

        default_K = range( 0 , self.N_Obs , np.ceil(self.N_Obs/10).astype(int))
        if k is not None:
            if np.shape(k)[0]<3:
                print('a minimum of 3 samples must be plotted to observe variation. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            elif np.shape(k)[0]>10:
                print('a maximum of 10 samples permitted to prevent overcrowding. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            elif np.min(k)<0:
                print('Cannot start indexing before 0. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            elif np.max(k)>self.N_Obs:
                print('Cannot index beyond number of observations. Defaulting to stepping every 1/10th of dataset')
                self.fig_k = default_K
            else:
                self.fig_k = k
        else:
            self.fig_k = default_K

        default_i = range( 0 , np.min((self.N_PC , 4, np.max(self.fig_k)-1) ) ) # N_PCs constrained to be less than k in local_NIPALS init
        if i is not None:
            if np.shape(i)[0] < 2:
                print('a minimum of 2 pcs must be plotted to observe variation. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            elif np.max(i) > (np.max(self.fig_k)-1):
                print('a maximum of 9 pcs permitted to prevent overcrowding. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            elif np.min(i)<0:
                print('Cannot start indexing before 0. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            elif np.max(i)>self.N_PC:
                print('Cannot index beyond number of PCs. Defaulting to 1st (max) 5 PCs')
                self.fig_i = default_i
            else:
                self.fig_i = i
        else:
            self.fig_i = default_i


        if Ylab is not None:
            if isinstance(Ylab, str):
                if len(Ylab)<33:
                    self.fig_Y_Label = Ylab
                else:
                    print('Y label too long, truncating to 32 characters')
                    self.fig_Y_Label = Ylab[:31]
            else:
                print('Y label is not a string, no Y label will be displayed')
                self.fig_Y_Label = ''
        else:
            self.fig_Y_Label = ''

        if Xlab is not None:
            if isinstance(Xlab, str):
                if len(Xlab)<33:
                    self.fig_X_Label = Xlab
                else:
                    print('X label too long, truncating to 32 characters')
                    self.fig_X_Label = Xlab[:31]
            else:
                print('X label is not a string, no X label will be displayed')
                self.fig_X_Label = ''
        else:
            self.fig_X_Label = ''

        if Show_Labels is not None:
            if Show_Labels and (len(self.fig_Y_Label)>0 or len(self.fig_X_Label)>0):
                self.fig_Show_Labels =  True
            else:
                self.fig_Show_Labels =  False
        else:
            self.fig_Show_Labels =  False

        if Show_Values is not None:
            if Show_Values:
                self.fig_Show_Values =  True
            else:
                self.fig_Show_Values =  False
        else:
            self.fig_Show_Values =  False

        if  TxtSz is not None:
            if TxtSz<6:
                print('Specified text size is below limit of 6, defaulting to 6')
                self.fig_Text_Size =  6
            elif TxtSz>18:
                print('Specified text size is above limit of 18, defaulting to 18')
                self.fig_Text_Size =  18
            else:
                self.fig_Text_Size =  TxtSz
        else:
            self.fig_Text_Size =  8
        plt.rcParams.update({'font.size': self.fig_Text_Size})
        plt.rcParams.update({'font.sans-serif': "Arial"})
        plt.rcParams.update({'lines.linewidth': "0.75"})
        plt.rcParams.update({'axes.xmargin': "0"})
        plt.rcParams.update({'axes.ymargin': "0"})

        if Project is not None:
            if isinstance(Project, str):
                if len(Project)<25:
                    self.fig_Project = Project
                else:
                    print('Project code too long, truncating to 25 characters')
                    self.fig_Project = Project[:24]
            else:
                print('Project code is not a string, default code "Graphical PCA Demo" applied')
                self.fig_Project = 'Graphical PCA Demo'
        else:
            self.fig_Project = 'Graphical PCA Demo'

        self.prepare_Data() #update offset data for plotting equations


### calc_PCA
    def calc_PCA(self):
        #        print('initialising NIPALS algorithm')
        self.r.append(self.data)  # initialise the residual_i as the raw input data
        self.rE[0, :] = np.sum(
            self.r[0] ** 2, 1
        )  # calculate total variance (initial residual variance)

        for component_array_index in range(self.N_PC):  # for i = 0,1,... max_i
            pc = np.ndarray((self.N_Vars, self.Max_It))
            w = np.ndarray((self.Max_It, self.N_Obs))
            jIt = 0  # iteration counter initialised
            pc[:, jIt] = (
                np.sum(self.r[component_array_index] ** 2, 1) ** 0.5
            )  # pc_i,j = norm of sqrt(sum(residual_i^2))
            pc[:, jIt] = (
                pc[:, jIt] / sum(pc[:, jIt] ** 2) ** 0.5
            )  # convert to unit vector for initial right eigenvector guess

            itChange = sum(
                abs(pc[:, 0])
            )  # calculate the variance in initialisation iteration, itChange = |pc_i,j|

            while True:
                if (
                    jIt < self.Max_It - 1 and itChange > self.Tol
                ):  # Max_It-1 since 0 is the first index: for j = 1,2,... max_j, while itChange<tol
                    #                    print(str(jIt)+' of '+str(self.Max_It)+' iterations')
                    #                    print(str(diff)+' compared to tolerance of ',str(self.Tol))
                    w[jIt, :] = (
                        self.r[component_array_index].T @ pc[:, jIt]
                    )  # calculate Scores from Loadings: w_i,j = outer product( residual_i , pc_i,j )
                    jIt += 1  # reset iteration counter, j, to 0
                    pc[:, jIt] = np.inner(
                        w[jIt - 1, :].T, self.r[component_array_index]
                    )  # update estimate of Loadings using Scores resulting from previous estimate: pc_i,j = inner product( w'_i,j-1 , residual_i)
                    ml = sum(pc[:, jIt] ** 2) ** 0.5  # norm of Loadings
                    pc[:, jIt] = (
                        pc[:, jIt] / ml
                    )  # convert Loadings to unit vectors:  pc_i,j = pc_i,j/|pc_i,j|
                    itChange = sum(
                        abs(pc[:, jIt] - pc[:, jIt - 1])
                    )  # total difference between iterations: itChange = sum(|pc_i,j - pc_i,j-1|)
                else:
                    break
                # endfor

            self.pc.append(
                pc[:, 0 : jIt - 1]
            )  # truncate iteration vectors to those calculated prior to final iteration
            self.spectral_loading[component_array_index, :] = pc[
                :, jIt
            ]  # store optimised Loading: rEV_i = pc_i,j

            self.w.append(
                w[0 : jIt - 1, :]
            )  # truncate iteration vectors to those calculated
            self.component_weight[:, component_array_index] = (
                self.r[component_array_index].T @ pc[:, jIt]
            )  # store Scores, = outer product( residual_i , pc_i,j )
            self.Eigenvalue[component_array_index] = np.dot(
                    self.component_weight[:, component_array_index].T,
                    self.component_weight[:, component_array_index]
            ) #calculate and store eigenvalue
            self.icomponent_weight[ component_array_index , :] = (
                self.component_weight[:, component_array_index]/
                self.Eigenvalue[component_array_index]
            )  # store inverse Scores, = scores scaled to norm (U scores in SVD)
            self.r.append(
                self.r[component_array_index]
                - np.outer(self.component_weight[:, component_array_index].T, self.spectral_loading[component_array_index, :].T).T
            )  # update residual:     couldn't get np.inner or np.dot to work
            self.rE[component_array_index + 1, :] = np.sum(
                self.r[component_array_index + 1] ** 2, 1
            )  # calculate residual variance
        self.prepare_Data()
### Project model
    def project_model(self, data):

        datamc = data.T - self.centring
        output = dict()
        output["scores"] = np.inner(datamc, self.spectral_loading)
        q2=np.empty(self.N_PC)
        for iPC in range(self.N_PC):
            output["recon"] = np.inner(output["scores"][:,:iPC+1], self.spectral_loading[:iPC+1,:].T)
            output["residual"]= datamc - output["recon"]
            q2[iPC] = 1 - (np.cov( output["residual"]).trace()/
                            np.cov( datamc ).trace() )
        output["Q2"] =q2
        return output

### prepare_Data
    def prepare_Data(self):
        data4plot = self.data[:,self.fig_k]
        data_spacing = (np.arange(np.shape(self.fig_k)[0]))*(np.mean(np.max(data4plot,axis=0))/2)
        self.data4plot = data4plot + data_spacing
        self.data0lines = np.tile([data_spacing],(2,1))

        dataSq4plot = self.data[:,self.fig_k]**2
        dataSq_spacing = (np.arange(np.shape(self.fig_k)[0]))*(np.mean(np.max(dataSq4plot,axis=0))/2)
        self.dataSq4plot = dataSq4plot + dataSq_spacing
        self.dataSq0lines = np.tile([dataSq_spacing],(2,1))

        Loadings4plot = self.spectral_loading[self.fig_i,:].T
        Loading_spacing = -(np.arange(np.shape(self.fig_i)[0]))*(np.mean(np.max(Loadings4plot,axis=1))*4)
        self.Loadings4plot = Loadings4plot + Loading_spacing
        self.Loading0lines = np.tile([Loading_spacing],(2,1))

        Scores4plot = self.component_weight[self.fig_k,:]
        Scores4plot = Scores4plot[:,self.fig_i]
        Score_spacing = (np.arange(np.shape(self.fig_k)[0]))*(np.mean(np.max(Scores4plot,axis=1))*1)
        self.Scores4plot = (Scores4plot.T + Score_spacing).T
        self.Score0lines = np.tile([Score_spacing],(2,1))

        iScores4plot = self.icomponent_weight[:,self.fig_k]
        iScores4plot = iScores4plot[self.fig_i,:].T
        iScore_spacing = (np.arange(np.shape(self.fig_k)[0]))*(np.mean(np.max(iScores4plot,axis=1))*1)
        self.iScores4plot = (iScores4plot.T + iScore_spacing).T
        self.iScore0lines = np.tile([iScore_spacing],(2,1))

### reduced_Rank_Reconstruction
    def reduced_Rank_Reconstruction( self , X , nPC ):

        Xc = (X.T-self.centring).T
        Scores = np.inner( Xc.T ,self.spectral_loading[:nPC,:])
        X_recon = np.inner( Scores , self.spectral_loading[:nPC,:].T).T
        X_recon = (X_recon.T+self.centring).T
        return X_recon

### calc_Constituents
    def calc_Constituents(self, nPC):
        # calculate the constituents that comprise the PC, splitting them into 3 parts:
        # Positive score weighted summed spectra - positive contributors to the PC
        # -Negative score weighted summed spectra - negative contributors to the PC
        # Sparse reconstruction ignoring current PC - common contributors to both the positive and negative constituent
        # Note that this implmenation is not generalisable, it depends on knowledge that is not known in real world datasets,
        # namely what the global minimum intensity at every pixel is. This means it is only applicable for simulated datasets.
        if self.mean != 0:
            print(
                "Warning: using subtracted spectra, such as mean or median centred data, will result in consitutents that are still"
                + " subtraction spectra - recontructing these to positive signals requires addition of the mean or median"
            )
        for component_array_index in range(nPC):#        print('extracting contributing Score features')
            posSc = self.component_weight[:, component_array_index] > 0

            self.negative_sum[:, component_array_index] = -np.sum(
                self.component_weight[posSc == False, component_array_index] * self.data[:, posSc == False], axis=1
            )
            self.positive_sum[:, component_array_index] = np.sum(
                self.component_weight[posSc, component_array_index] * self.data[:, posSc], axis=1
            )
            # tried the mean only. The shape is identical but the scaling is different.
            # The mean was 46.25458682703735 times larger than the reconstructed version
            # the ratio of Eigenvalues was 1385.6306160811262
            # the ratio of singular values was 37.22405963998454
            # so neither could be used to rescale and not disturb the below version.
            # The reconstructed version ensures the scale of the common
            # constituent is the same as for the p anc negative_sums
            self.orthogonal_sum[:, component_array_index] = np.inner(
                np.inner(
                    np.mean([self.negative_sum[:, component_array_index], self.positive_sum[:, component_array_index]], axis=0),
                    self.spectral_loading[range(component_array_index), :],
                ),                                                              # returns scores for all previous loadings
                self.spectral_loading[range(component_array_index), :].T,
            )                                                                   # These are used to adjust the mean offset of the pCon and nCon spectral weightings.
            self.global_minimum[:, component_array_index] = np.sum(
                self.component_weight[posSc, component_array_index] * self.min_spectral_values[:, posSc], axis=1
            )  # minimum score vector
            if component_array_index>0: #not applicable to PC1
                tSF = np.ones(11)
                res = np.ones(11)
                for X_data in range(
                    0, 9
                ):  # search across 10x required precision of last minimum
                    tSF[X_data] = (X_data + 1) / 10
                    negative_sumS = (
                        self.negative_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index] * tSF[X_data]
                    )  # negative constituent corrected for common signal
                    positive_sumS = (
                        self.positive_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index] * tSF[X_data]
                    )  # positive constituent corrected for common signal
                    global_minimumS = self.global_minimum[:, component_array_index] * (1 - tSF[X_data])
                    orthogonal_sumS = self.orthogonal_sum[:, component_array_index] * (1 - tSF[X_data])

                    res[X_data] = np.min([negative_sumS - global_minimumS, positive_sumS - global_minimumS]) / np.max(orthogonal_sumS)
                res[res < 0] = np.max(
                    res
                )  # set all negative residuals to max so as to bias towards undersubtraction
                optSF = tSF[np.nonzero(np.abs(res) == np.min(np.abs(res)))]

                for iPrec in range(2, 10):  # define level of precsision required
                    tSF = np.ones(19)
                    res = np.ones(19)
                    for X_data in range(
                        -9, 10
                    ):  # serach across 10x required precision of last minimum
                        tSF[X_data + 9] = optSF + X_data / 10 ** iPrec
                        negative_sumS = (
                            self.negative_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index] * tSF[X_data + 9]
                        )  # - constituent corrected for common signal
                        positive_sumS = (
                            self.positive_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index] * tSF[X_data + 9]
                        )  # + constituent corrected for common signal
                        orthogonal_sumS = self.orthogonal_sum[:, component_array_index] * (1 - tSF[X_data + 9])
                        global_minimumS = self.global_minimum[:, component_array_index] * (1 - tSF[X_data + 9])

                        res[X_data + 9] = np.min([negative_sumS - global_minimumS, positive_sumS - global_minimumS]) / np.max(orthogonal_sumS)
                    res[res < 0] = np.max(
                        res
                    )  # set all negative residuals to max so as to bias towards undersubtraction
                    optSF = tSF[np.nonzero(np.abs(res) == np.min(np.abs(res)))]
                self.optSF[component_array_index] = optSF[0]
                self.negative_sumS[:, component_array_index] = (
                    self.negative_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index] * optSF
                )  # - constituent corrected for common signal
                self.positive_sumS[:, component_array_index] = (
                    self.positive_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index] * optSF
                )  # + constituent corrected for common signal
                self.orthogonal_sumS[:, component_array_index] = self.orthogonal_sum[:, component_array_index] * (1 - optSF)
                self.global_minimumS[:, component_array_index] = self.global_minimum[:, component_array_index] * (1 - optSF)
            else: #no subtraction in 1st PC
                optSF = 0
                self.optSF[component_array_index] = 0
                self.negative_sumS[:, component_array_index] = (
                    self.negative_sum[:, component_array_index]
                )  # - constituent corrected for common signal
                self.positive_sumS[:, component_array_index] = (
                    self.positive_sum[:, component_array_index]
                )  # + constituent corrected for common signal
                self.orthogonal_sumS[:, component_array_index] = self.orthogonal_sum[:, component_array_index] * (1 - optSF)
                self.global_minimumS[:, component_array_index] = self.global_minimum[:, component_array_index] * (1 - optSF)

        ## need to work out how to handle min_spectral_values

        # print('extracted positive and negative Score features')

### figure_DSLT
    def figure_DSLT(self, arrangement):
        grid_Column = np.array( [8, 1, 4, 1, 8] )
        sub_Fig = [ "a" , "", "b" , "" , "c" ]
        s_Str = ") $S$" # default name for score matrix
        Lstr = r"$L{^\top}$" #default name for loading matrix
        v_Ord = np.array([0,1,2,3,4]) #Variable axis order for equation plot
       #        grid_Row = np.array([5, 5, 3])
        if arrangement == "DSLT":#column data (convention in maths)
            #                   D           =           S           .           LT
            txt_Positions = [[0.17, 0.9],[0.43, 0.5],[0.47, 0.9],[0.51, 0.5],[0.66, 0.9]]
            matrix_Dims = [ r"$n\times m$" , "", r"$n\times p$" , "" , r"$p\times m$" ]
        elif arrangement == "SDL": # scores for row data matrix
            v_Ord = np.array([2,1,0,3,4])
            txt_Positions = [[0.35, 0.9],[0.2, 0.5],[.14, 0.9],[0.51, 0.5],[0.66, 0.9]]
            matrix_Dims = [ r"$n\times p$" , "", r"$n\times m$" , "" , r"$m\times p$" ]
            Lstr = r"$L$"
        elif arrangement == "LTSiD": # loadings for row data matrix
            v_Ord = np.array([4,1,2,3,0])
            #                   D           =           S           .           LT
            txt_Positions = [[0.66, 0.9],[0.43, 0.5],[0.47, 0.9],[0.55, 0.5],[0.17, 0.9]]
            matrix_Dims = [ r"$p\times m$" , "", r"$p\times n$" , "" , r"$n\times m$" ]
            s_Str = ") $S^{+}$" #overwrite S matrix name to indicate pseudo-inverse
        else:
            print(str(arrangement)+" is not a valid option. Use DSLT, SDL or LTSiD")
        #    return
        v_Ix = np.argsort(v_Ord) #index of sorted variable order for reverse lookup
        columns_ordered = [0, grid_Column[v_Ix[:1]][0],np.sum(grid_Column[v_Ix[:2]]),np.sum(grid_Column[v_Ix[:3]]),np.sum(grid_Column[v_Ix[:4]])]
        #determine correct column starting positions
        figDSLT, axDSLT = plt.subplots(1, 5,figsize=self.fig_Size)
        axDSLT[v_Ord[0]] = plt.subplot2grid((6, 22), (0, columns_ordered[v_Ord[0]]), colspan=8, rowspan=6)
        axDSLT[v_Ord[1]] = plt.subplot2grid((6, 22), (0, columns_ordered[v_Ord[1]]), colspan=1, rowspan=6)
        axDSLT[v_Ord[2]] = plt.subplot2grid((6, 22), (0, columns_ordered[v_Ord[2]]), colspan=4, rowspan=6)
        axDSLT[v_Ord[3]] = plt.subplot2grid((6, 22), (0, columns_ordered[v_Ord[3]]), colspan=1, rowspan=6)
        axDSLT[v_Ord[4]] = plt.subplot2grid((6, 22), (0, columns_ordered[v_Ord[4]]), colspan=8, rowspan=6)

        axDSLT[v_Ord[0]].plot(self.pixel_axis[[0,-1]], self.data0lines,  "-.", linewidth=0.5)
        axDSLT[v_Ord[0]].plot(self.pixel_axis, self.data4plot)

        alphas = 0.75*(self.fig_i/np.max(self.fig_i))**0.5 #want biggest alpha to be slightly transparent
        if arrangement == "LTSiD":
            axDSLT[v_Ord[2]].plot(self.iScores4plot.T, ".",
                  transform=transforms.Affine2D().rotate_deg(270) +
                  axDSLT[v_Ord[2]].transData, markersize=7)
            axDSLT[v_Ord[2]].plot([-0.15,np.max(self.fig_i)+0.15],self.iScore0lines, "-.",
                  transform=transforms.Affine2D().rotate_deg(270) +
                  axDSLT[v_Ord[2]].transData, linewidth=0.5)
            xlims = axDSLT[v_Ord[2]].get_xlim()
            axDSLT[v_Ord[2]].set_xlim((xlims[0]*1.1,xlims[1]*1.1))
            for iP in self.fig_i:
                axDSLT[v_Ord[2]].add_patch(patches.Rectangle((xlims[0]*1.1,-iP-0.035) , (xlims[1]-xlims[0])*1.1, 0.07,
                                                   alpha = alphas[iP], color = 'w', edgecolor=None, zorder=10))
            axDSLT[v_Ord[4]].annotate(
                'C=C$_c$',
                xy=(0.2, 0.04),
                xytext=(0.07, 0.85),
                textcoords="axes fraction",
                xycoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                color=[1, 0, 0],
                horizontalalignment="left",
                rotation=90,
                va="center",
            )
            axDSLT[v_Ord[4]].annotate(
                'C-H$_2$',
                xy=(0.2, 0.04),
                xytext=(0.12, 0.97),
                textcoords="axes fraction",
                xycoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                color=[0.8, 0.65, 0.3],
                horizontalalignment="left",
                rotation=90,
                va="center",
            )
            axDSLT[v_Ord[4]].annotate(
                'C-H$_x$',
                xy=(0.2, 0.04),
                xytext=(0.34, 0.885),
                textcoords="axes fraction",
                xycoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                color=[0.87, 0.72, 0.38],
                horizontalalignment="left",
                va="center",
            )
            axDSLT[v_Ord[4]].annotate(
                'C=C$_c$',
                xy=(0.2, 0.04),
                xytext=(0.72, 0.9),
                textcoords="axes fraction",
                xycoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                color=[1, 0, 0],
                horizontalalignment="left",
                rotation=90,
                va="center",
            )
            axDSLT[v_Ord[4]].annotate(
                'C=C$_t$',
                xy=(0.2, 0.04),
                xytext=(0.78, 0.8),
                textcoords="axes fraction",
                xycoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                color=[0, 1, 0],
                horizontalalignment="left",
                rotation=90,
                va="center",
            )
            axDSLT[v_Ord[4]].annotate(
                'C=O',
                xy=(0.2, 0.04),
                xytext=(0.9, 0.78),
                textcoords="axes fraction",
                xycoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                color=[0, 0, 1],
                horizontalalignment="left",
                rotation=90,
                va="center",
            )
        else:
            axDSLT[v_Ord[2]].plot(self.Scores4plot.T, ".", markersize=7)
            axDSLT[v_Ord[2]].plot([-0.15,np.max(self.fig_i)+0.15],self.Score0lines, "-.", linewidth=0.5)
            axDSLT[v_Ord[2]].set_ylim(axDSLT[v_Ord[2]].get_ylim()[0]*1.1,axDSLT[v_Ord[2]].get_ylim()[1]*1.1)
# currently have xcf files in GIMP that do the shading.
# To apply to an updated figure:
# 1. paste in the new png generated by Python
# 2. set as a new layer
# 3. move down to just above the base layer (remove any previous edits)
# 4. for inverse loadings (LTSiD) you need to select the scores area and paste
#    in a new layer the flip horizontally
# 5. export as png to replace the python generated version
            ylims = axDSLT[v_Ord[2]].get_ylim()*0.9
            for iP in range(1,np.shape(self.fig_i)[0]):
                axDSLT[v_Ord[2]].add_patch(patches.Rectangle((iP-0.5,ylims[0]), 1 , ylims[1]-ylims[0],
                                                   alpha = alphas[iP], color = 'w', edgecolor=None, zorder=10))
            axDSLT[v_Ord[2]].add_patch(patches.Rectangle((iP+0.5,ylims[0]), 0.5 , ylims[1]-ylims[0],
                                               alpha = 1, color = 'w', edgecolor=None, zorder=10))

        if arrangement == "SDL":
            axDSLT[v_Ord[4]].plot(self.pixel_axis[[0,-1]],self.Loading0lines,'-.',
                  transform=transforms.Affine2D().rotate_deg(90) +
                  axDSLT[v_Ord[4]].transData, linewidth=0.5)
            axDSLT[v_Ord[4]].plot(self.pixel_axis,self.Loadings4plot,
                  transform=transforms.Affine2D().rotate_deg(90) +
                  axDSLT[v_Ord[4]].transData)
        else:
            axDSLT[v_Ord[4]].plot(self.pixel_axis[[0,-1]],self.Loading0lines,'-.', linewidth=0.5)
            axDSLT[v_Ord[4]].plot(self.pixel_axis,self.Loadings4plot)

        for iC in range(np.shape(self.fig_i)[0]):
            axDSLT[v_Ord[4]].lines[iC].set_color(str(0 + iC / 5)) #shade loadings
            axDSLT[v_Ord[4]].lines[iC+np.shape(self.fig_i)[0]].set_color(str(0 + iC / 5)) #shade zero lines



        axDSLT[v_Ord[0]].annotate(
            sub_Fig[v_Ix[0]]+") $D$",
            xy=(txt_Positions[0]),
            xytext=(txt_Positions[0]),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDSLT[v_Ord[1]].annotate(
            "=",
            xy=(0.5,0.5),
            xytext=(0.5, 0.5),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size*1.5,
            horizontalalignment="center",
        )
        axDSLT[v_Ord[2]].annotate(
            sub_Fig[v_Ix[2]] + s_Str,
            xy=(txt_Positions[2]),
            xytext=(txt_Positions[2]),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",

        )
        axDSLT[v_Ord[3]].annotate(
            r"$\cdot$",
            xy=(0,0.5),
            xytext=(0.5, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*3,
            horizontalalignment="center",

        )
        axDSLT[v_Ord[4]].annotate(
            sub_Fig[v_Ix[4]]+") "+Lstr,
            xy=(txt_Positions[4]),
            xytext=(txt_Positions[4]),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",

        )
        axDSLT[v_Ord[0]].annotate(
            matrix_Dims[v_Ix[0]],
            xy=(0.15, 0.12),
            xytext=(0.5, 1.02),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",

        )
        axDSLT[v_Ord[2]].annotate(
            matrix_Dims[v_Ix[2]],
            xy=(0.52, 0.12),
            xytext=(0.5, 1.02),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",
        )

        axDSLT[v_Ord[4]].annotate(
            matrix_Dims[v_Ix[4]],
            xy=(0.75, 0.12),
            xytext=(0.5,1.02),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            va="center",

        )
        if arrangement == "DSLT": #only put dimensions on the main plot
            axDSLT[v_Ord[0]].annotate(
                "$o=1$",
                xy=(0.12, 0.15),
                xytext=(0.12, 0.22),
                textcoords="figure fraction",
                xycoords="figure fraction",
                arrowprops=dict(facecolor="black",arrowstyle="->",
                            connectionstyle="arc3",lw=1),
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",

            )
            axDSLT[v_Ord[0]].annotate(
                "$o=n$",
                xy=(0.12, 0.06),
                xytext=(0.12, 0.14),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",

            )
            axDSLT[v_Ord[0]].annotate(
                "$v=1$",
                xy=(0.27, 0.12),
                xytext=(0.15, 0.12),
                textcoords="figure fraction",
                xycoords="figure fraction",
                arrowprops=dict(facecolor="black",arrowstyle="->",
                            connectionstyle="arc3",lw=1),
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                va="center",

            )
            axDSLT[v_Ord[0]].annotate(
                "$v=m$",
                xy=(0.27, 0.12),
                xytext=(0.27, 0.12),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                va="center",

            )
            axDSLT[v_Ord[2]].annotate(
                "$o=1$",
                xy=(0.45, 0.15),
                xytext=(0.45, 0.22),
                textcoords="figure fraction",
                xycoords="figure fraction",
                arrowprops=dict(facecolor="black",arrowstyle="->",
                            connectionstyle="arc3",lw=1),
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",

            )
            axDSLT[v_Ord[2]].annotate(
                "$o=n$",
                xy=(0.45, 0.14),
                xytext=(0.45, 0.14),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",

            )
            axDSLT[v_Ord[2]].annotate(
                "$i=1$",
                xy=(0.55, 0.12),
                xytext=(0.48, 0.12),
                textcoords="figure fraction",
                xycoords="figure fraction",
                arrowprops=dict(facecolor="black",arrowstyle="->",
                            connectionstyle="arc3",lw=1),
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                va="center",

            )
            axDSLT[v_Ord[2]].annotate(
                "$i=p$",
                xy=(0.55, 0.12),
                xytext=(0.55, 0.12),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                va="center",

            )

            axDSLT[v_Ord[4]].annotate(
                "$i=1$",
                xy=(0.62, 0.15),
                xytext=(0.62, 0.22),
                textcoords="figure fraction",
                xycoords="figure fraction",
                arrowprops=dict(facecolor="black",arrowstyle="->",
                            connectionstyle="arc3",lw=1),
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",

            )
            axDSLT[v_Ord[4]].annotate(
                "$i=p$",
                xy=(0.62, 0.14),
                xytext=(0.62, 0.14),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",

            )
            axDSLT[v_Ord[4]].annotate(
                "$v=1$",
                xy=(0.78, 0.12),
                xytext=(0.645, 0.12),
                textcoords="figure fraction",
                xycoords="figure fraction",
                arrowprops=dict(facecolor="black",arrowstyle="->",
                            connectionstyle="arc3",lw=1),
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                va="center",

            )
            axDSLT[v_Ord[4]].annotate(
                "$v=m$",
                xy=(0.78, 0.12),
                xytext=(0.78, 0.12),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                va="center",

            )

        if not self.fig_Show_Values:
            for iax in range(len(axDSLT)):
                axDSLT[iax].axis("off")

        if self.fig_Show_Labels:
            axDSLT[v_Ord[0]].set_ylabel(self.fig_Y_Label)
            axDSLT[v_Ord[0]].set_xlabel(self.fig_X_Label)
            axDSLT[v_Ord[2]].set_ylabel('Score / Arbitrary')
            axDSLT[v_Ord[2]].set_xlabel('Sample #')
            axDSLT[v_Ord[4]].set_ylabel('Weights / Arbitrary')
            axDSLT[v_Ord[4]].set_xlabel(self.fig_X_Label)

        image_name = " " + arrangement + " Eqn."
        full_path = os.path.join(images_folder, self.fig_Project +
                                image_name + self.fig_Format)
        figDSLT.savefig(full_path,
                         dpi=self.fig_Resolution)

        plt.close()

### figure_sldi
    def figure_sldi(self, component_index):
            # plots the vector process for how scores are calculated to compliment the
    # relevant math equation, rather than directly represent it.
    # component_index is the PC number, not index so starts at 1 for the first PC

    # vector plots differ radically so not simple to create common function
        if component_index is None:
            component_array_index = self.fig_i[0] #internal list is python index so need to add 1 for compatibility
            print('No PC specified for vector figure (figure_dsli). Defaulting to first PC used in matrix figures')
        else:
            component_array_index = component_index-1
        figsldi, axsldi = plt.subplots(1, 5, figsize=self.fig_Size)
        axsldi[0] = plt.subplot2grid((1, 20), (0, 0), colspan=8)
        axsldi[1] = plt.subplot2grid((1, 20), (0, 8), colspan=1)
        axsldi[2] = plt.subplot2grid((1, 20), (0, 9), colspan=8)
        axsldi[3] = plt.subplot2grid((1, 20), (0, 17), colspan=1)
        axsldi[4] = plt.subplot2grid((1, 20), (0, 18), colspan=2)

        iSamMin = np.argmin(self.component_weight[:, component_array_index])
        iSamMax = np.argmax(self.component_weight[:, component_array_index])
        iSamZer = np.argmin(
            np.abs(self.component_weight[:, component_array_index])
        )  # Sam = 43 #the ith sample to plot
        sf_iSam = np.mean(
            [
                sum(self.data[:, iSamMin] ** 2) ** 0.5,
                sum(self.data[:, iSamMax] ** 2) ** 0.5,
                sum(self.data[:, iSamZer] ** 2) ** 0.5,
            ]
        )  # use samescaling factor to preserve relative intensity
        offset = np.max(self.spectral_loading[component_array_index, :]) - np.min(
            self.spectral_loading[component_array_index, :]
        )  # offset for clarity
        axsldi[0].plot(
            self.pixel_axis,
            self.spectral_loading[component_array_index, :] + offset * 1.25,
            "k",
            self.pixel_axis,
            self.data[:, iSamMax] / sf_iSam + offset / 4,
            "r",
            self.pixel_axis,
            self.data[:, iSamZer] / sf_iSam,
            "b",
            self.pixel_axis,
            self.data[:, iSamMin] / sf_iSam - offset / 4,
            "g",
            )
        axsldi[0].plot(
            self.pixel_axis[[0,-1]],
            np.tile(offset *1.25,(2,1)),
            "-.k",
            self.pixel_axis[[0,-1]],
            np.tile(offset /4,(2,1)),
            "-.r",
            self.pixel_axis[[0,-1]],
            np.tile(0,(2,1)),
            "-.b",
            self.pixel_axis[[0,-1]],
            np.tile(-offset / 4,(2,1)),
            "-.g", linewidth=0.5, zorder=1,
        )
        axsldi[0].legend(("$l_i$", "$d_{max}$", "$d_{zero}$", "$d_{min}$"), loc=8)
        temp = self.spectral_loading[component_array_index, :] * self.data[:, iSamZer]
        offsetProd = np.max(temp) - np.min(temp)
        axsldi[2].plot(
            self.pixel_axis,
            self.spectral_loading[component_array_index, :] * self.data[:, iSamMax] + offsetProd,
            "r",
            self.pixel_axis,
            self.spectral_loading[component_array_index, :] * self.data[:, iSamZer],
            "b",
            self.pixel_axis,
            self.spectral_loading[component_array_index, :] * self.data[:, iSamMin] - offsetProd,
            "g",
            )
        axsldi[2].plot(
            self.pixel_axis[[0,-1]],
            np.tile(offsetProd,(2,1)),
            "-.r",
            self.pixel_axis[[0,-1]],
            np.tile(0,(2,1)),
            "-.b",
            self.pixel_axis[[0,-1]],
            np.tile(-offsetProd,(2,1)),
            "-.g", linewidth=0.5, zorder=1
        )

        PCilims = np.tile(
            np.array(
                [
                    np.average(self.component_weight[:, component_array_index])
                    - 1.96 * np.std(self.component_weight[:, component_array_index]),
                    np.average(self.component_weight[:, component_array_index]),
                    np.average(self.component_weight[:, component_array_index])
                    + 1.96 * np.std(self.component_weight[:, component_array_index]),
                ]
            ),
            (2, 1),
        )
        axsldi[4].plot(
            [0, 10],
            PCilims,
            "k--",
            5,
            self.component_weight[iSamMax, component_array_index],
            "r.",
            5,
            self.component_weight[iSamZer, component_array_index],
            "b.",
            5,
            self.component_weight[iSamMin, component_array_index],
            "g.",
            markersize=7,
        )
        ylimLEV = (
            np.abs(
                [
                    self.component_weight[:, component_array_index].min(),
                    self.component_weight[:, component_array_index].max(),
                ]
            ).max()
            * 1.05
        )
        axsldi[4].set_ylim([-ylimLEV, ylimLEV])

        axsldi[0].annotate(
            "a) PC"+str(component_index)+" loading & data",
            xy=(0.2, 0.95),
            xytext=(0.25, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        #connection arrows showing
        xpos = np.round(0.5*self.N_Vars).astype('int')
        ypos = np.mean( [ self.spectral_loading[component_array_index, xpos] + offset * 1.25,
                         self.data[xpos, iSamMax] / sf_iSam + offset / 4])
        axsldi[0].annotate(
            r"$\times$",
            xy=(self.pixel_axis[xpos], 0.5),
            xytext=(self.pixel_axis[xpos], ypos),
            xycoords="data",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            color = (0.7, 0.7, 0.7)
        )
        for xpos in (np.arange(0.05,1,0.1)*self.N_Vars).astype('int'):
            ypos0 = self.spectral_loading[component_array_index, xpos] + offset * 1.25
            ypos1 = self.data[xpos, iSamMax] / sf_iSam + offset / 4
            axsldi[0].annotate(
                "",
                xy=(self.pixel_axis[xpos], ypos1),
                xytext=(self.pixel_axis[xpos], ypos0),
                xycoords="data",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = (0.7, 0.7, 0.7)),
            )

        axsldi[0].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), color = (0.5,0.5,0.5)
        )
        axsldi[1].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldi[1].annotate(
            r"$l_i \times d_o$",
            xy=(0.5, 0.5),
            xytext=(0.5, 0.52),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldi[2].annotate(
            "b) PC weighted data",
            xy=(0.55, 0.95),
            xytext=(0.6, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        ypos = np.max(self.spectral_loading[component_array_index, :] * self.data[:, iSamMax] + offsetProd)*0.9
        axsldi[2].annotate(
            "",
            xy=(self.pixel_axis[-1], ypos),
            xytext=(self.pixel_axis[0], ypos),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = (0.7, 0.7, 0.7)),
        )
        axsldi[2].annotate(
            "+",
            xy=(self.pixel_axis[0], ypos*1.1),
            xytext=(self.pixel_axis[np.round((self.N_Vars/2)).astype('int')], ypos*1.1),
            xycoords="data",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            color = (0.7, 0.7, 0.7)
        )

        for xpos in (np.arange(0.05,1,0.1)*self.N_Vars).astype('int'):
            ypos0 = self.spectral_loading[component_array_index, xpos] + offset * 1.25
            ypos1 = self.data[xpos, iSamMax] / sf_iSam + offset / 4
            axsldi[0].annotate(
                "",
                xy=(self.pixel_axis[xpos], ypos1),
                xytext=(self.pixel_axis[xpos], ypos0),
                xycoords="data",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = (0.7, 0.7, 0.7)),
            )


        axsldi[3].annotate(
            "$\Sigma _{v=1}^{v=m}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.52),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldi[3].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldi[0].annotate(
            "c) Score",
            xy=(0.3, 0.95),
            xytext=(0.87, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldi[4].annotate(
            "$U95CI$",
            xy=(5, PCilims[0, 2]),
            xytext=(10, PCilims[0, 2]),
            xycoords="data",
            textcoords="data",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axsldi[4].annotate(
            "$\overline{S_{p}}$",
            xy=(0, 0.9),
            xytext=(1, 0.49),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axsldi[4].annotate(
            "$L95CI$",
            xy=(5, PCilims[0, 0]),
            xytext=(10, PCilims[0, 0]*1.05),
            xycoords="data",
            textcoords="data",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )

        if not self.fig_Show_Values:
            for iax in range(len(axsldi)):
                axsldi[iax].axis("off")

        if self.fig_Show_Labels:
            axsldi[0].set_ylabel(self.fig_Y_Label)
            axsldi[0].set_xlabel(self.fig_X_Label)
            axsldi[2].set_ylabel('PC Weighted ' + self.fig_Y_Label)
            axsldi[2].set_xlabel(self.fig_X_Label)
            axsldi[4].set_ylabel('Weights / Arbitrary')

        figsldi.savefig(
                str(images_folder) + "\\" +
                self.fig_Project +
                " sldi Eqn." +
                self.fig_Format,
                dpi=self.fig_Resolution
                )

        plt.close()

        figsldiRes, axsldiRes = plt.subplots(1, 3, figsize=self.fig_Size)
        axsldiRes[0] = plt.subplot2grid((1, 17), (0, 0), colspan=8)
        axsldiRes[1] = plt.subplot2grid((1, 17), (0, 8), colspan=1)
        axsldiRes[2] = plt.subplot2grid((1, 17), (0, 9), colspan=8)

        iSamResMax = self.data[:, iSamMax] - np.inner(
            self.component_weight[iSamMax, component_array_index],
            self.spectral_loading[component_array_index, :],
        )
        iSamResZer = self.data[:, iSamZer] - np.inner(
            self.component_weight[iSamZer, component_array_index],
            self.spectral_loading[component_array_index, :],
        )
        iSamResMin = self.data[:, iSamMin] - np.inner(
            self.component_weight[iSamMin, component_array_index],
            self.spectral_loading[component_array_index, :],
        )
#        offsetRes = np.max(iSamResZer) - np.min(iSamResZer)

        axsldiRes[0].plot(
            self.pixel_axis,
            self.data[:, iSamMax] / sf_iSam + offset / 4,
            "r",
            self.pixel_axis,
            self.data[:, iSamZer] / sf_iSam,
            "b",
            self.pixel_axis,
            self.data[:, iSamMin] / sf_iSam - offset / 4,
            "g",
            self.pixel_axis,
            np.inner(
                self.component_weight[iSamMax, component_array_index],
                self.spectral_loading[component_array_index, :],
            )
            / sf_iSam
            + offset / 4,
            "k--",
            )
        axsldiRes[0].plot(
            self.pixel_axis[[0,-1]],
            np.tile(offset / 4,(2,1)),
            "-.r",
            self.pixel_axis[[0,-1]],
            np.tile(0,(2,1)),
            "-.b",
            self.pixel_axis[[0,-1]],
            np.tile(-offset / 4,(2,1)),
            "-.g", linewidth=0.5,
            )
        axsldiRes[0].plot(
            self.pixel_axis,
            np.inner(
                self.component_weight[iSamZer, component_array_index],
                self.spectral_loading[component_array_index, :],
            )
            / sf_iSam,
            "k--",
            self.pixel_axis,
            np.inner(
                self.component_weight[iSamMin, component_array_index],
                self.spectral_loading[component_array_index, :],
            )
            / sf_iSam
            - offset / 4,
            "k--", linewidth=1,
        )
        axsldiRes[0].legend(("$d_{max}$", "$d_{zero}$", "$d_{min}$", r"$s_o \times l_i$"))

        axsldiRes[2].plot(
            self.pixel_axis,
            iSamResMax/sf_iSam + offset / 4,
            "r",
            self.pixel_axis[[0,-1]],
            np.tile(offset / 4,(2,1)),
            "-.r",
            self.pixel_axis,
            iSamResZer/sf_iSam,
            "b",
            self.pixel_axis[[0,-1]],
            np.tile(0,(2,1)),
            "-.b",
            self.pixel_axis,
            iSamResMin/sf_iSam - offset / 4,
            "g",
             self.pixel_axis[[0,-1]],
            np.tile(-offset / 4,(2,1)),
            "-.g",
        )
        axsldiRes[2].set_ylim(axsldiRes[0].get_ylim())

        axsldiRes[0].annotate(
            "a) Data & PC"+str(component_index)+" score weighted loading",
            xy=(0.2, 0.95),
            xytext=(0.3, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRes[1].annotate(
            "",
            xy=(0.55, 0.5),
            xytext=(0.47, 0.5),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axsldiRes[1].annotate(
            r"$d_o-$",
            xy=(0.5, 0.5),
            xytext=(0.51, 0.52),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRes[1].annotate(
            r"$(s_o \times l_i)$",
            xy=(0.5, 0.5),
            xytext=(0.51, 0.46),
            xycoords="figure fraction",
            textcoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axsldiRes[2].annotate(
            "b) PC"+str(component_index)+" residual",
            xy=(0.2, 0.95),
            xytext=(0.75, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )

        if not self.fig_Show_Values:
            for iax in range(len(axsldiRes)):
                axsldiRes[iax].axis("off")

        if self.fig_Show_Labels:
            axsldiRes[0].set_ylabel(self.fig_Y_Label)
            axsldiRes[0].set_xlabel(self.fig_X_Label)
            axsldiRes[2].set_ylabel('PC Weighted ' + self.fig_Y_Label)
            axsldiRes[2].set_xlabel(self.fig_X_Label)
            axsldiRes[4].set_ylabel('Weights / Arbitrary')

        image_name = " sldi Residual Eqn."
        full_path = os.path.join(images_folder, self.fig_Project +
                                image_name + self.fig_Format)
        figsldiRes.savefig(full_path,
                         dpi=self.fig_Resolution)

        plt.close()

### figure_lsdi
    def figure_lsdi( self , component_index ):
        ###################        START lsdiScoreEqn          #######################
        # FIGURE for the ith Loading equation Li = S^{-1}i*D
        if component_index is None:
            component_array_index = self.fig_i[0] #internal list is python index so need to add 1 for compatibility
            print('No PC specified for vector figure (figure_dsli). Defaulting to first PC used in matrix figures')
        else:
            component_array_index = component_index-1
        figlsdi, axlsdi = plt.subplots(1, 6, figsize=self.fig_Size)
        axlsdi[0] = plt.subplot2grid((1, 21), (0, 0), colspan=1)
        axlsdi[1] = plt.subplot2grid((1, 21), (0, 1), colspan=6)
        axlsdi[2] = plt.subplot2grid((1, 21), (0, 7), colspan=1)
        axlsdi[3] = plt.subplot2grid((1, 21), (0, 8), colspan=6)
        axlsdi[4] = plt.subplot2grid((1, 21), (0, 14), colspan=1)
        axlsdi[5] = plt.subplot2grid((1, 21), (0, 15), colspan=6)

        c_Inv_Score = self.icomponent_weight[component_array_index,:]
        PCilims = np.tile(np.array([np.nanmean(c_Inv_Score)-1.96*np.nanstd(c_Inv_Score),
                                    0,
                                    np.nanmean(c_Inv_Score)+1.96*np.nanstd(c_Inv_Score)]),
                          (2,1))

        axlsdi[1].plot(
            self.pixel_axis,
            4*self.data[:,self.fig_k]/self.Eigenvalue[component_index] + c_Inv_Score[self.fig_k],
            [self.pixel_axis[0],self.pixel_axis[-1]],
            np.tile(c_Inv_Score[self.fig_k],(2,1)),
            "-."
            )
        axlsdi[0].scatter(
            np.tile([0],(np.size(self.fig_k),1)),
            c_Inv_Score[self.fig_k],
            c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        )
        axlsdi[0].plot(
            [-1, 1],
            PCilims,
            "--k", linewidth=0.5
        )
        axlsdi[0].set_ylim((np.min(c_Inv_Score)*1.1,np.max(c_Inv_Score)*1.1))
        axlsdi[0].annotate(
            "$L95\%CI$",
            xy=(0.11, 0.35),
            xytext=(0.11, 0.16),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="right",
        )
        axlsdi[0].annotate(
            "$0$",
            xy=(0.11, 0.57),
            xytext=(0.11, 0.5),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="right",
        )
        axlsdi[0].annotate(
            "$U95\%CI$",
            xy=(0.11, 0.78),
            xytext=(0.11, 0.84),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="right",
        )

        axlsdi[0].annotate(
            "a) $s^{-1}_{o,i}$",
            xy=(0.12, 0.95),
            xytext=(0.12, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axlsdi[1].annotate(
            "b) $d_o$",
            xy=(0.27, 0.95),
            xytext=(0.27, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axlsdi[3].plot(c_Inv_Score[self.fig_k] * self.data[:,self.fig_k])
        axlsdi[3].annotate(
            r"c) $s^{-1}_{o,i} \times d_o$",
            xy=(0.52, 0.95),
            xytext=(0.52, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            alpha = 0.95
        )
        axlsdi[5].plot(self.spectral_loading[component_array_index, :], c='k')
        axlsdi[5].plot(axlsdi[5].get_xlim(),(0,0),'-.',linewidth=0.5, c=[0.2,0.2,0.2],zorder=1)
        axlsdi[5].annotate(
            "d) $l_i$",
            xy=(0.8, 0.5),
            xytext=(0.8, 0.95),
            textcoords="figure fraction",
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )

        axlsdi[2].annotate(
            r"$s^{-1}_{o,i} \times d_o$",
            xy=(0, 0.5),
            xytext=(0.52, 0.52),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axlsdi[2].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axlsdi[4].annotate(
            "$\Sigma _{o=1}^{o=n}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.52),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axlsdi[4].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        if not self.fig_Show_Values:
            for iax in range(len(axlsdi)):
                axlsdi[iax].axis("off")

        if self.fig_Show_Labels:
            axlsdi[0].set_ylabel('Scores / Arbitrary')
            axlsdi[1].set_ylabel(self.fig_Y_Label)
            axlsdi[1].set_xlabel(self.fig_X_Label)
            axlsdi[3].set_ylabel('PC Weighted ' + self.fig_Y_Label)
            axlsdi[3].set_xlabel(self.fig_X_Label)
            axlsdi[5].set_xlabel(self.fig_X_Label)
            axlsdi[5].set_ylabel('Weights / Arbitrary')

        image_name = " lsdi Eqn."
        full_path = os.path.join(images_folder, self.fig_Project +
                                image_name + self.fig_Format)
        figlsdi.savefig(full_path,
                         dpi=self.fig_Resolution)
        plt.close()



### figure_lpniCommonSignalScalingFactors
    def figure_lpniCommonSignalScalingFactors(self, nPC, xview):
        ###################       START lpniCommonSignalScalingFactors       #######################
        # FIGURE of the scaling factor calculated for subtracting the common signal from the positive
        # and negative constituents of a PC
        if xview is None:
            xview = [0,np.length(self.pixel_axis)]
        figlpniS, axlpniS = plt.subplots(
            1, 6, figsize=self.fig_Size
        )  # Extra columns to match spacing in
        for component_array_index in range(1, nPC):
            iFig = (
                (component_array_index % 6) - 1
            )  # modulo - determine where within block the current PC is
            if iFig == -1:
                iFig = 5  # if no remainder then it is the last in the cycle
            axlpniS[iFig] = plt.subplot2grid((1, 6), (0, iFig), colspan=1)
            axlpniS[iFig].plot(self.pixel_axis[[xview[0], xview[-0]]], [0, 0], "--")
            axlpniS[iFig].plot(
                self.pixel_axis[xview], self.negative_sumS[xview, component_array_index], "b", linewidth=1
            )
            axlpniS[iFig].plot(
                self.pixel_axis[xview], self.positive_sumS[xview, component_array_index], "y", linewidth=1
            )
            axlpniS[iFig].plot(
                self.pixel_axis[xview], self.orthogonal_sumS[xview, component_array_index], "g", linewidth=0.5
            )
            txtpos = [
                np.mean(axlpniS[iFig].get_xlim()),
                axlpniS[iFig].get_ylim()[1] * 0.9,
            ]
            axlpniS[iFig].annotate(
                "PC " + str(component_array_index+1),
                xy=(txtpos),
                xytext=(txtpos),
                textcoords="data",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
            )

            if iFig == 5:
                if not self.fig_Show_Values:
                    for iax in range(len(axlpniS)):
                        axlpniS[iax].axis("off")
#                        axlpniU[iax].axis("off")
                if self.fig_Show_Labels:
                    axlpniS[iax].set_ylabel("Weighted " + self.fig_Y_Label)
#                    axlpniU[iax].set_ylabel("Weighted " + self.fig_Y_Label)
                    for iax in range(len(axlpniS)):
                        axlpniS[iax].set_xlabel(self.fig_X_Label)
#                        axlpniU[iax].set_xlabel(self.fig_X_Label)

                image_name = f" lpni common corrected PC{str(component_array_index - 4)} to {str(component_array_index + 1)}."
                full_path = os.path.join(images_folder, self.fig_Project +
                                        image_name + self.fig_Format)
                figlpniS.savefig( full_path,
                                 dpi=self.fig_Resolution)

#                plt.show()
                plt.close()
                #create new figures
                figlpniS, axlpniS = plt.subplots(
                    1, 6, figsize=self.fig_Size
                )  # Extra columns to match spacing in

        plt.close()
        figlpniU, axlpniU = plt.subplots(
            1, 6, figsize=self.fig_Size
        )  # Extra columns to match spacing
        for component_array_index in range(1, nPC):
            iFig = (
                (component_array_index % 6) - 1
            )  # modulo - determine where within block the current PC is
            if iFig == -1:
                iFig = 5  # if no remainder then it is the last in the cycle
            axlpniU[iFig] = plt.subplot2grid((1, 6), (0, iFig), colspan=1)
            axlpniU[iFig].plot(self.pixel_axis[[xview[0], xview[-0]]], [0, 0], "--")
            axlpniU[iFig].plot(
                self.pixel_axis[xview], self.negative_sum[xview, component_array_index], "b", linewidth=1
            )
            axlpniU[iFig].plot(
                self.pixel_axis[xview], self.positive_sum[xview, component_array_index], "y", linewidth=1
            )
            axlpniU[iFig].plot(
                self.pixel_axis[xview], self.orthogonal_sum[xview, component_array_index], "g", linewidth=0.5
            )
            axlpniU[iFig].annotate(
                "PC " + str(iFig + 2),
                xy=(
                    np.mean(axlpniU[iFig].get_xlim()),
                    axlpniU[iFig].get_ylim()[1] * 0.9,
                ),
                xytext=(
                    np.mean(axlpniU[iFig].get_xlim()),
                    axlpniU[iFig].get_ylim()[1] * 0.9,
                ),
                textcoords="data",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
            )
            if iFig == 5:
                if not self.fig_Show_Values:
                    for iax in range(len(axlpniU)):
                        axlpniU[iax].axis("off")
                if self.fig_Show_Labels:
                    axlpniU[iax].set_ylabel("Weighted " + self.fig_Y_Label)
                    for iax in range(len(axlpniU)):
                        axlpniU[iax].set_xlabel(self.fig_X_Label)
                image_name = f" lpni common raw PC{str(component_array_index - 4)} to {str(component_array_index + 1)}."
                full_path = os.path.join(images_folder, self.fig_Project +
                                        image_name + self.fig_Format)
                figlpniU.savefig(full_path,
                                 dpi=self.fig_Resolution)
#                plt.show()

                plt.close()
                figlpniU, axlpniU = plt.subplots(
                    1, 6, figsize=self.fig_Size
                )

        plt.close()
        plt.figure(figsize=self.fig_Size)
        plt.plot(range(2, np.shape(self.optSF)[0] + 1), self.optSF[1:], ".")
        drp = np.add(np.nonzero((self.optSF[2:] - self.optSF[1:-1]) < 0), 2)
        if np.size(drp) != 0:
            plt.plot(drp[0][0] + 1, self.optSF[drp[0][0]], "or")
            plt.plot([2, nPC], [self.optSF[drp[0][0]], self.optSF[drp[0][0]]], "--")
        image_name = f" lpni common signal scaling factors PC2 to {str(nPC)}."
        full_path = os.path.join(images_folder, self.fig_Project +
                                image_name + self.fig_Format)
        plt.savefig(full_path,
                         dpi=self.fig_Resolution)

        plt.close()

        # copy scalingAdjustment.py into its own cell after running this main cell in Jupyter then you
        # can manually adjust the scaling factors for each PC to determine what is the most appropriate method

        ###### Plot positive, negative score and common  signals without any  common signal subtraction ######
        ###################         END lpniCommonSignalScalingFactors           #######################

### figure_lpniScoreEqn
    def figure_lpniScoreEqn(self, component_index):
        # this class function prints out images comparing the score magnitude weighted summed spectra for
        # positive and negative score spectra. The class must have already calculated the positive, negative
        # and common consitituents
        if component_index is None:
            component_array_index = self.fig_i[1] #internal list is python index so need to add 1 for compatibility
            print('No PC specified for vector figure (figure_dsli). Defaulting to second PC used in matrix figures')
        else:
            component_array_index = component_index-1

        if ~np.isnan(self.orthogonal_sum[0, component_array_index]):

            figlpni, axlpni = plt.subplots(1, 6, figsize=self.fig_Size)
            axlpni[0] = plt.subplot2grid((1, 21), (0, 0), colspan=1)
            axlpni[1] = plt.subplot2grid((1, 21), (0, 1), colspan=6)
            axlpni[2] = plt.subplot2grid((1, 21), (0, 7), colspan=1)
            axlpni[3] = plt.subplot2grid((1, 21), (0, 8), colspan=6)
            axlpni[4] = plt.subplot2grid((1, 21), (0, 14), colspan=1)
            axlpni[5] = plt.subplot2grid((1, 21), (0, 15), colspan=6)
            posSc = self.component_weight[:, component_array_index] > 0  # skip PC1 as not subtraction
            if any(np.sum(posSc==True)==[0,np.shape(posSc)[0]]) or any(np.sum(posSc==False)==[0,np.shape(posSc)[0]]):
                component_index = 2
                component_array_index = component_index-1
                print('Data not mean centered, so assuming input data is all positive then no positive/negative split will be observed. Defaulting to PC2')
                #this switches PC to 2 if PC1 has no combination of positive or negative
            axlpni[0].plot([-10, 10], np.tile(0, (2, 1)), "k")
            axlpni[0].plot(np.tile(0, sum(posSc)), self.component_weight[posSc, component_array_index], ".y")
            axlpni[0].plot(
                np.tile(0, sum(posSc == False)),
                self.component_weight[posSc == False, component_array_index],
                ".b",
            )

            axlpni[1].plot(self.pixel_axis, self.data[:, posSc], "y")
            axlpni[1].plot(self.pixel_axis, self.data[:, posSc == False], "--b", lw=0.1)

            axlpni[3].plot(self.pixel_axis, self.negative_sum[:, component_array_index], "b")
            axlpni[3].plot(self.pixel_axis, self.positive_sum[:, component_array_index], "y")

            pnegative_sum = self.positive_sum[:, component_array_index] - self.negative_sum[:, component_array_index]
            pnegative_sum = pnegative_sum / np.sum(pnegative_sum ** 2) ** 0.5 # unit vector
            axlpni[5].plot(self.spectral_loading[component_array_index, :], "m")
            axlpni[5].plot(pnegative_sum, "--c", linewidth=0.75)
            axlpni[5].plot(self.spectral_loading[component_array_index, :] - pnegative_sum, "-.k",linewidth=0.5)

            # subplot headers
            axlpni[0].annotate(
                "a) $s^{-1}_{o,i}$",
                xy=(0.12, 0.95),
                xytext=(0.12, 0.9),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
            )
            axlpni[1].annotate(
                "b) $d_o$",
                xy=(0.27, 0.95),
                xytext=(0.2, 0.9),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
            )
            axlpni[3].annotate(
                r"c) $|s^{-1}_{o,i\pm}| \times d_o = l_{i\pm}$",
                xy=(0.52, 0.95),
                xytext=(0.5, 0.9),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                alpha = 0.95
            )
            axlpni[5].annotate(
                "d) $l_i$",
                xy=(0.8, 0.5),
                xytext=(0.7, 0.9),
                textcoords="figure fraction",
                xycoords="figure fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
            )

            # other annotation
            axlpni[1].annotate(
                "$s_{o,i}^{-1} >0$",
                xy=(0.37, 0.8),
                xytext=(0.4, 0.84),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
                color="y",
            )
            axlpni[1].annotate(
                "$s_{o,i}^{-1} <0$",
                xy=(0.35, 0.78),
                xytext=(0.4, 0.8),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
                color="b",
            )

            axlpni[3].annotate(
                "$l_{i+}$",
                xy=(0.37, 0.8),
                xytext=(0.4, 0.84),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
                color="y",
            )
            axlpni[3].annotate(
                "$l_{i-}$",
                xy=(0.35, 0.78),
                xytext=(0.4, 0.8),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="left",
                color="b",
            )


#            ylim = np.max(pnegative_sum)
            axlpni[5].legend(("$l_i$","$l_{i+}-l_{i-}$","$l_i-(l_{i+}-l_{i-})$"),
                             loc=(0.22,0.78),
                             fontsize=self.fig_Text_Size*0.75)
            totdiff = "{:.0f}".format(
                np.log10(np.mean(np.abs(self.spectral_loading[component_array_index, :] - pnegative_sum)))
            )
            axlpni[5].annotate(
                "$\Delta_{l_i,(l_{i+}-l_{i-})}$",
                xy=(0.1, 0.9),
                xytext=(0.42, 0.54),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                color="k",
            )
            axlpni[5].annotate(
                "$=10^{" + totdiff + "}$",
                xy=(0.1, 0.9),
                xytext=(0.46, 0.5),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="left",
                color="k",
            )

            axlpni[2].annotate(
                r"$\Sigma(s_{i+}^{-1} \times d_{i+})$",
                xy=(0, 0.5),
                xytext=(0.5, 0.53),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="center",
            )
            axlpni[2].annotate(
                r"$\Sigma(-s_{i-}^{-1} \times d_{i-})$",
                xy=(0, 0.5),
                xytext=(0.5, 0.44),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size*0.75,
                horizontalalignment="center",
            )
            axlpni[2].annotate(
                "",
                xy=(1, 0.5),
                xytext=(0, 0.5),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )
            axlpni[4].annotate(
                "",
                xy=(1, 0.5),
                xytext=(0, 0.5),
                textcoords="axes fraction",
                fontsize=self.fig_Text_Size,
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )

            if not self.fig_Show_Values:
                for iax in range(len(axlpni)):
                    axlpni[iax].axis("off")

            if self.fig_Show_Labels:
                axlpni[0].set_xlabel(self.fig_X_Label)

            image_name = f" positive negative contributions PC_{str(component_index)}."
            full_path = os.path.join(images_folder, self.fig_Project +
                                    image_name + self.fig_Format)
            figlpni.savefig(full_path,
                             dpi=self.fig_Resolution)
            plt.close()

        else:
            print(
                "Common, Positive and Negative Consituents must be calculated first using calorthogonal_sums"
            )

### figure_lpniCommonSignal
    def figure_lpniCommonSignal(self, component_index, SF = None):
# component_index allows control of which PC is plotted
# SF allows control of the scaling factor tested. Leaving no SF will default to the value calculated by calc_Constituents

        # this class function prints out images comparing the score magnitude weighted summed spectra for
        # positive and negative score spectra corrected for the common consitituents, compared with the common
        # consituents itself and the scaled global minimum
        if component_index is None:
            component_index = 2
            print('No PC defined for lpniCommonSignal. Defaulting to PC2')
        component_array_index = component_index-1

        if SF is None:
            SF = self.optSF[component_array_index]
            plt.plot(self.pixel_axis, self.negative_sumS[:, component_array_index], "b")
            plt.plot(self.pixel_axis, self.positive_sumS[:, component_array_index], "y")
            plt.plot(self.pixel_axis, self.orthogonal_sumS[:, component_array_index], "g")
            plt.plot(self.pixel_axis, self.global_minimumS[:, component_array_index], "--c", linewidth=0.5)
        else:
            plt.plot(self.pixel_axis, self.negative_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index]*SF, "b")
            plt.plot(self.pixel_axis, self.positive_sum[:, component_array_index] - self.orthogonal_sum[:, component_array_index]*SF, "y")
            plt.plot(self.pixel_axis, self.orthogonal_sum[:, component_array_index] * (1-SF), "g")
            plt.plot(self.pixel_axis, self.global_minimum[:, component_array_index] * (1-SF), "--c", linewidth=0.5)

        image_name = " Common Signal Subtraction PC" + str(component_index) + " Scale Factor " + str(SF)
        plt.title(image_name)
        plt.legend(
            ("-ve Constituent", "+ve Constituent", "Common Signal", "Global Minimum")
        )

        if not self.fig_Show_Values:
            plt.gca().axis("off")

        if self.fig_Show_Labels:
            plt.gca().set_ylabel(self.fig_Y_Label)
            plt.gca().set_xlabel(self.fig_X_Label)

        image_name = image_name.replace(".","_") + "."
        full_path = os.path.join(images_folder, self.fig_Project +
                                image_name + self.fig_Format)
        plt.savefig(full_path,
                         dpi=self.fig_Resolution)

        plt.close()

###  figure_DTD
    def figure_DTD(self,):
        ###################                  START DTDscoreEqn                  #######################
        # FIGURE showing how the inner product of the data forms the sum of squares
        figDTD, axDTD = plt.subplots(1, 5, figsize=self.fig_Size)
        axDTD[0] = plt.subplot2grid((1, 20), (0, 0), colspan=6)
        axDTD[1] = plt.subplot2grid((1, 20), (0, 6), colspan=1)
        axDTD[2] = plt.subplot2grid((1, 20), (0, 7), colspan=6)
        axDTD[3] = plt.subplot2grid((1, 20), (0, 13), colspan=1)
        axDTD[4] = plt.subplot2grid((1, 20), (0, 14), colspan=6)


        axDTD[0].plot(self.pixel_axis[[0,-1]], self.data0lines,"-.",lw=0.5)
        axDTD[0].plot(self.pixel_axis, self.data4plot)
        axDTD[2].plot(self.pixel_axis[[0,-1]], self.dataSq0lines,"-.",lw=0.5)
        axDTD[2].plot(self.pixel_axis, self.dataSq4plot)
        axDTD[4].plot(self.pixel_axis,  np.sum(self.data**2,1))

        axDTD[0].annotate(
            "a) $d_{o=1...n}$",
            xy=(0.1, 0.95),
            xytext=(0, 1.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )

        axDTD[2].annotate(
            r"b) $d_o\times d_o$",
            xy=(0.1, 0.95),
            xytext=(0, 1.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )

        axDTD[4].annotate(
            "c) Sum of Squares",
            xy=(0.1, 0.9),
            xytext=(0, 1.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )

        axDTD[1].annotate(
            "$d_o^2$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDTD[1].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axDTD[3].annotate(
            "$\Sigma _{o=1}^{o=n}(d_o^2)$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
        )
        axDTD[3].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        if not self.fig_Show_Values:
            for iax in range(len(axDTD)):
                axDTD[iax].axis("off")

        if self.fig_Show_Labels:
            axDTD[0].set_ylabel(self.fig_Y_Label)
            axDTD[0].set_xlabel(self.fig_X_Label)
            axDTD[2].set_ylabel(self.fig_Y_Label + "$^2$")
            axDTD[2].set_xlabel(self.fig_X_Label)
            axDTD[4].set_ylabel(self.fig_Y_Label + "$^2$")
            axDTD[4].set_xlabel(self.fig_X_Label)

        image_name = " DTD Eqn."
        full_path = os.path.join(images_folder, self.fig_Project +
                                image_name + self.fig_Format)
        figDTD.savefig(full_path,
                         dpi=self.fig_Resolution)


        plt.close()
        ###################                  END DTDscoreEqn                  #######################
### figure_DTDw
    def figure_DTDw(self,):
        ###################                  START D2DwscoreEqn               #######################
        # FIGURE for the illustration of the NIPALs algorithm, with the aim of iteratively calculating
        # each PCA to minimise the explantion of the sum of squares
        figD2Dw, axD2Dw = plt.subplots(3, 5, figsize=[self.fig_Size[0],self.fig_Size[0]])
        #needs to be square plot
        axD2Dw[0, 0] = plt.subplot2grid((13, 20), (0, 0), colspan=6, rowspan=6)
        axD2Dw[0, 1] = plt.subplot2grid((13, 20), (0, 6), colspan=1, rowspan=6)
        axD2Dw[0, 2] = plt.subplot2grid((13, 20), (0, 7), colspan=6, rowspan=6)
        axD2Dw[0, 3] = plt.subplot2grid((13, 20), (0, 13), colspan=1, rowspan=6)
        axD2Dw[0, 4] = plt.subplot2grid((13, 20), (0, 14), colspan=6, rowspan=6)
        axD2Dw[1, 0] = plt.subplot2grid((13, 20), (7, 0), colspan=7, rowspan=1)
        axD2Dw[1, 1] = plt.subplot2grid((13, 20), (7, 7), colspan=7, rowspan=1)
        axD2Dw[1, 2] = plt.subplot2grid((13, 20), (7, 14), colspan=6, rowspan=1)
        axD2Dw[2, 0] = plt.subplot2grid((13, 20), (8, 0), colspan=6, rowspan=6)
        axD2Dw[2, 1] = plt.subplot2grid((13, 20), (8, 6), colspan=1, rowspan=6)
        axD2Dw[2, 2] = plt.subplot2grid((13, 20), (8, 7), colspan=6, rowspan=6)
        axD2Dw[2, 3] = plt.subplot2grid((13, 20), (8, 13), colspan=1, rowspan=6)
        axD2Dw[2, 4] = plt.subplot2grid((13, 20), (8, 14), colspan=6, rowspan=6)

        # data plots
        # initial data (PC=0)
        axD2Dw[0, 0].plot(self.pixel_axis, self.r[0])
        ylims0_0 = np.max(np.abs(axD2Dw[0, 0].get_ylim()))
        axD2Dw[0, 0].set_ylim(
            -ylims0_0, ylims0_0
        )  # tie the y limits so scales directly comparable

        # sum of squares for residual after PCi (for raw data before 1st PCA i=0)
        axD2Dw[0, 2].plot(self.pixel_axis, self.pc[1][:, 0], "m")
        axD2Dw[0, 2].plot(self.pixel_axis, self.pc[0][:, 0])

        # scores in current iteration (j) if current PC(i)
        axD2Dw[0, 4].plot(self.w[1][0, :], ".m")
        axD2Dw[0, 4].plot(self.w[0][0, :] / 10, ".")#divide by 10 so on visually comparable scale
        axD2Dw[0, 4].plot(self.w[0][1, :], ".c")
        axD2Dw[0, 4].legend(["$pc_2$,$it_1$","$pc_1$,$it_2$","$pc_1$,$it_1$"],
              handletextpad = 0.1,
              loc='best', bbox_to_anchor=(0, 0.5, 0.5, 0.5)
              )

        # current iteration j of loading i
        axD2Dw[2, 4].plot(self.pixel_axis, self.pc[1][:, 1], "m")
        axD2Dw[2, 4].plot(self.pixel_axis, self.pc[0][:, 1])
        axD2Dw[2, 4].plot(self.pixel_axis, self.pc[0][:, 2], "c")
        ylims2_4 = np.max(np.abs(axD2Dw[2, 4].get_ylim()))
        axD2Dw[2, 4].set_ylim(
            -ylims2_4, ylims2_4
        )  # tie the y limits so scales directly comparable

        # Iteration j-1 to j change in loading i
        axD2Dw[2, 2].plot(
            self.pixel_axis,
            np.abs(self.pc[1][:, 1] - self.pc[1][:, 0]),
            "m",
        )
        axD2Dw[2, 2].plot(
            self.pixel_axis,
            np.abs(self.pc[0][:, 1] - self.pc[0][:, 0]),
        )
        ylims2_2 = np.max(np.abs(axD2Dw[2, 2].get_ylim())) * 1.1
        axD2Dw[2, 2].plot(
            self.pixel_axis,
            np.abs(self.pc[0][:, 2] - self.pc[0][:, 1]),
            "c",
        )
        axD2Dw[2, 2].set_ylim([0 - ylims2_2 * 0.1, ylims2_2])
        axD2Dw[2, 0].plot(self.pixel_axis, self.r[1])
        axD2Dw[2, 0].set_ylim(
            -ylims0_0, ylims0_0
        )  # tie the y limits so scales directly comparable

        # subplot headers
        axD2Dw[0, 0].annotate(
            "a) $R_0=D_{-\mu}$",
            xy=(0.25,0.95),
            xytext=(0, 1.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axD2Dw[0, 2].annotate(
            "b) $\widehat{SS_{R_{pc}}}$",
            xy=(0.5,0.95),
            xytext=(0, 1.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axD2Dw[0, 4].annotate(
            "c) $S_{pc,it}$",
            xy=(0.75,0.95),
            xytext=(0, 1.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axD2Dw[2, 4].annotate(
            "d) $L_{pc,it}^T$",
            xy=(0.75,0.1),
            xytext=(0, -0.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axD2Dw[2, 2].annotate(
            "e) $\Delta L^T_{it,it-1}$",
            xy=(0.5,0.1),
            xytext=(0, -0.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )
        axD2Dw[2, 0].annotate(
            "f) $R_{pc}$",
            xy=(0.25,0.1),
            xytext=(0, -0.05),
            xycoords="axes fraction",
            fontsize=self.fig_Text_Size,
            horizontalalignment="left",
        )

        # other information
        axD2Dw[0, 1].annotate(
            "$\widehat{\Sigma(R_0^2)}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
        )
        axD2Dw[0, 1].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        axD2Dw[0, 3].annotate(
            "$R_{pc}\widehat{SS}$",
            xy=(0, 0.5),
            xytext=(0.5, 0.55),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
        )
        axD2Dw[0, 3].annotate(
            "",
            xy=(1, 0.5),
            xytext=(0, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[0, 3].annotate(
            "$pc=pc+1$",
            xy=(0, 0.5),
            xytext=(0.5, 0.45),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
        )
        axD2Dw[0, 3].annotate(
            "$it=it+1$",
            xy=(0, 0.5),
            xytext=(0.5, 0.40),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
        )
        axD2Dw[1, 2].annotate(
            "",
            xy=(0.5, 0),
            xytext=(0.5, 2),
            textcoords="axes fraction",
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[1, 2].annotate(
            r"$S_{pc,it}^{-1}\times R_{pc-1}$",
            xy=(0.55, 0.5),
            xytext=(0.57, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            rotation=90,
        )
        axD2Dw[2, 3].annotate(
            "",
            xy=(0, 0.5),
            xytext=(1, 0.5),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[2, 3].annotate(
            "$|L_{pc,it}-L_{pc,it-1}|$",
            xy=(0.53, 0.55),
            xytext=(0.5, 0.42),
            textcoords="axes fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
        )
        axD2Dw[2, 2].annotate(
            "$\Sigma|\Delta L^T_{it,it-1}|<Tol$",
            xy=(0.48,0.375),
            xytext=(0.48,0.38),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
        )
        axD2Dw[2, 2].annotate(
            "OR $it=max\_{it}$",
            xy=(0.45,0.36),
            xytext=(0.48,0.355),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
        )
        axD2Dw[2, 2].annotate(
            "$False$",
            xy=(0.65,0.55),
            xytext=(0.5,0.4),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            color="r",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[1, 1].annotate(
            r"$R_{pc-1}^T\times L_{pc,it}$",
            xy=(0.65,0.55),
            xytext=(0.57,0.47),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            rotation=45,
        )
        axD2Dw[1, 1].annotate(
            "$it=it+1$",
            xy=(0.65,0.55),
            xytext=(0.59,0.45),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            rotation=45,
        )

        axD2Dw[2, 2].annotate(
            "$True$",
            xy=(0.35,0.25),
            xytext=(0.44,0.34),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            color="g",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

        axD2Dw[2, 1].annotate(
            r"$R_{pc-1}-S_{pc}\times L_{pc}^T$",
            xy=(0.65,0.55),
            xytext=(0.38,0.27),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            rotation=45,
        )

        axD2Dw[1, 0].annotate(
            "",
            xy=(0.5,0.53),
            xytext=(0.3,0.33),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            rotation=45,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        axD2Dw[1, 0].annotate(
            "$\widehat{\Sigma(R_{pc}^2)}$",
            xy=(0.305,0.535),
            xytext=(0.405,0.435),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            rotation=45,
        )
        axD2Dw[1, 0].annotate(
            "$it=0$",
            xy=(0.32,0.52),
            xytext=(0.42,0.42),
            xycoords="figure fraction",
            fontsize=self.fig_Text_Size*0.75,
            horizontalalignment="center",
            rotation=45,
        )
        if not self.fig_Show_Values:
            for iaxr in range(np.shape(axD2Dw)[0]):
                for iaxc in range(np.shape(axD2Dw)[1]):
                    axD2Dw[iaxr,iaxc].axis("off")

        if self.fig_Show_Labels:
            axD2Dw[0,0].set_ylabel(self.fig_Y_Label)
            axD2Dw[0,0].set_xlabel(self.fig_X_Label)
            axD2Dw[0,2].set_ylabel(self.fig_Y_Label + "$^2$")
            axD2Dw[0,2].set_xlabel(self.fig_X_Label)
            axD2Dw[0,4].set_ylabel("Score / Arbitrary")
            axD2Dw[0,4].set_xlabel("Sample Index")
            axD2Dw[2,0].set_ylabel("Residual "+self.fig_Y_Label)
            axD2Dw[2,0].set_xlabel(self.fig_X_Label)
            axD2Dw[2,2].set_ylabel(self.fig_Y_Label + "$^2$")
            axD2Dw[2,2].set_xlabel(self.fig_X_Label)
            axD2Dw[2,4].set_ylabel("Weighting / " + self.fig_Y_Label)
            axD2Dw[2,4].set_xlabel(self.fig_X_Label)

        image_name = " NIPALS Algorithm."
        full_path = os.path.join(images_folder, self.fig_Project +
                                image_name + self.fig_Format)
        figD2Dw.savefig(full_path,
                         dpi=self.fig_Resolution)

        plt.close()
       ###################                  END D2DwscoreEqn                  #######################
