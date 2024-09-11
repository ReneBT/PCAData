# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 07:12:18 2024

@author: rene
"""
import matplotlib as mpl
import os


class figurePCA_Data:
    def __init__( self ):
        mpl.rcParams['lines.markersize'] = 3
        self.WD = os.getcwd()
        self.plot_show = False
        self.fig_Size = [8,5]
        self.fig_Resolution = 800
        self.fig_Format = 'tif'
        self.fig_Y_Label = 'Relative Intensity'
        self.fig_X_Label = 'Raman Shift cm^{-1}'
        self.fig_Text_Size =  12
        self.fig_Project = 'Graphical PCA Data'
        self.fig_Show_Values = False
        self.fig_Show_Labels = False
        return
