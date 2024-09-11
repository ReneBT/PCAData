import h5py
import src.DGP_data as dgp
import src.DGP_shot_noise as shot
import src.preprocessing as pp
import src.cage_covariance as cage
import src.dirty_data as dirty_data
import src.figure as fig
import src.figure1 as fig1
import src.figure2 as fig2
import src.figure4 as fig4
import src.figure5 as fig5
import src.figure6 as fig6
import src.figure7 as fig7
import src.figure8 as fig8
import src.figure9 as fig9
import src.figure10 as fig10
import src.figure11 as fig11
import src.figure12 as fig12
import src.figure13 as fig13
import src.figure14 as fig14
from src.file_locations import data_folder


# Using pathlib we can handle different filesystems (mac, linux, windows) using a common syntax.
# file_path = data_folder / "raw_data.txt"
# More info on using pathlib:
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f


class graphicalPCA_Data:
### ******      START CLASS      ******
    # base class for a NIPALs implmentation of PCA intended for training purposes on small datasets as it creates many
    # intermediate attributes not usually retained in efficent code
    # original data must be oriented such that sample spectra are aligned
    # along the columns and each row corresponds to different variables

    # comments include references to relevant lines in the pseudocode listed in the paper

    def __init__( self ):
### ***   START  Data Calculations   ***
    ### Read in data
    # simulated fatty acid spectra and associated experimental concentration data
        
        print( 'Initialising PCA graphical Data, loading reference data')
        
        # Gas Chromatograph (GC) data is modelled based on 
        # Beattie et al. Lipids 2004 Vol 39 (9):897-906
        # it is reconstructed with 4 underlying factors
        self.GC_data = h5py.File(data_folder / "AllGC.mat")
        # simplified_fatty_acid_spectrkma are simplified spectra of fatty acid
        # methyl esters built from the properties described in
        # Beattie et al. Lipids  2004 Vol 39 (5): 407-419
        self.simplified_fatty_acid_spectra = h5py.File(data_folder / "FA spectra.mat")
        
        # use loaded underlying initialisation and process data to generate 
        # basic 'observed' data
        self = dgp.gen_data( self )
        print( 'Basic Data Generated, creating perturbances: Shot Noise')

        # generate unbiased shot noise as an example of a common well defined 
        # perturbance
        self = shot.gen_data( self )
        print( 'Shot Noise Complete, creating perturbances: scale and offset')
        
        # generate data preprocessed in a variety of ways to explore the impact 
        # and relevance of common preprocessing steps
        self = pp.gen_data( self )
        print( 'Preprocessing Data Generation Complete: Commencing Cage of Covariance')
        
        # generate alternative fine covariance structures to explore how changes 
        # to the cage of covariance impacts the model
        self = cage.gen_data( self )
        print( 'Cage of Covariance generated: commencing dirty data generation')

        # generate biased noise to explore the implications of biased effects
        self = dirty_data.gen_data( self )
        print( 'Dirty Data Generated')
        print( 'Ready for Plotting')

        return 

    def plots( self ):
        self.fig_settings = fig.figurePCA_Data()        
        fig1.fig( self ) # basic spectra and basic generated dataset
        fig2.fig( self ) # Plot data to demonstrate Offset and Scale
        # Figure 3 is a schematic generated in Powerpoint 
        fig4.fig( self ) # Data to score variation
        fig5.fig( self ) # Data to demonstrate GC spectral crossover
        fig6.fig( self ) # Data to demonstrate tecnological covariance
        fig7.fig( self ) # PCA models to demonstrate impact of mean centring
        fig8.fig( self ) # PCA models to demonstrate impact of scaling
        fig9.fig( self ) # PCA models to demonstrate impact of normalisation
        fig10.fig( self ) # PCA results to demonstrate imapct of shot noise
        fig11.fig( self ) # Reconstruction of noisy data by PCA
        fig12.fig( self ) # PCA results showing correlation smearing of noise
        fig13.fig( self ) # PCA residuals for clean and baised models
        fig14.fig( self ) # Impact of cage of covariance perturbances for validation
        print( 'All Figures completed')
### ******      END CLASS      ******
        return
    
#PCAdata  = graphicalPCA_Data.graphicalPCA_Data
#src.figure1.figure1( test )