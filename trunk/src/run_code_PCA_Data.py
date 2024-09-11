# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 05:16:20 2024

@author: rene
"""
# module versions used for publication
# copulas       v0.8.0
# sklearn       v1.2.1
# numpy         v1.24.2
# h5py          v3.8.0
# airPLS        v2.0
# scipy         v1.10.0
# matplotlib    v3.6.3
# pathlib       v1.0.1
from src import graphicalPCA_Data

data  = graphicalPCA_Data.graphicalPCA_Data()
data.plots()