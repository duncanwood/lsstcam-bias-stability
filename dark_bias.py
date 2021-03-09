import sys
import os
#from astropy.io import fits
import fitsio
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import astropy.time
from lsst.daf.persistence import Butler
import lsst.afw
import numpy as np
import lsst.eotest.image_utils as imutils
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.eotest.sensor.AmplifierGeometry import makeAmplifierGeometry
import lsst.ip.isr as isr
import sys

import sklearn.linear_model
from sklearn.decomposition import PCA
import scipy.optimize
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter

def mean_serial(this_bias_data):
    return np.mean(this_bias_data['serial_arr'][:,2:], axis=1)
def mean_parallel(this_bias_data):
    return np.mean(this_bias_data['parallel_arr'][2:], axis=0)
def project_principal_components(pca, arr):
    return pca.inverse_transform(pca.transform(arr))
def remove_pca_both(this_bias_data, serial_pca, parallel_pca):
    this_mean_serial = mean_serial(this_bias_data)
    this_mean_parallel = mean_parallel(this_bias_data)
    mean_bias = np.mean(np.concatenate([this_mean_parallel.flat,this_mean_serial.flat]))
    return mean_bias + (this_bias_data['im_arr']-project_principal_components(serial_pca, this_mean_serial.reshape(1, -1)).T) \
-project_principal_components(parallel_pca, this_mean_parallel.reshape(1, -1))

def get_pixel_data(exp, amp_num):
    
    det = exp.getDetector()
    amp = det[amp_num]
    im = exp.getMaskedImage()[amp.getRawBBox()]
    arr_imaging  = im[amp.getRawDataBBox()].getImage().getArray()
    arr_parallel = im[amp.getRawParallelOverscanBBox()].getImage().getArray()
    arr_serial   = im[amp.getRawHorizontalOverscanBBox()].getImage().getArray()

    metadata = exp.getMetadata()
    
    visit_result = [im, arr_imaging, arr_parallel, arr_serial, metadata]
    visit_result_dict = {}
    datatypes = [('masked_im', object), ('im_arr', object),('parallel_arr', object),('serial_arr', object), ('metadata', object)]
    for i, item in enumerate(visit_result):
            visit_result_dict[datatypes[i][0]] = item
        
    return visit_result_dict

class PCA_Corrector:
    serial_pca = None
    parallel_pca = None
    amp_num = None
    n_components = None
    
    def __init__(self, biases, ampnum, ncomponents=7):
        n_components = ncomponents
        self.serial_pca = PCA(n_components=n_components)
        self.parallel_pca = PCA(n_components=n_components)
        self.amp_num = ampnum
        
        bias_data = []
        for exp in biases:
            bias_data.append(get_pixel_data(exp, self.amp_num))
        
        #datatypes = [('masked_im', object), ('im_arr', object),('parallel_arr', object),('serial_arr', object), ('metadata', object)]
        #bias_data_arr = np.asarray(bias_data,dtype=datatypes)
        bias_data_arr = pd.DataFrame(bias_data)
        
        serial_means = [mean_serial(this_bias_data) for this_bias_data in bias_data_arr.to_dict('records')]
        parallel_means = [mean_parallel(this_bias_data) for this_bias_data in bias_data_arr.to_dict('records')]
        
        self.serial_pca.fit(serial_means)
        self.parallel_pca.fit(parallel_means)
        
    def correct(self, exp: lsst.afw.image.exposure.ExposureF):
        return remove_pca_both(get_pixel_data(exp,self.amp_num), self.serial_pca, self.parallel_pca)
    
    
    
    