import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.time
from lsst.daf.persistence import Butler
import lsst.afw
try:
    import lsst.eotest.image_utils as imutils
except:
    import eotest.image_utils as imutils
    pass
from pathlib import Path

import sklearn.linear_model
from sklearn.decomposition import PCA

import random
from astropy.visualization import (MinMaxInterval, PercentileInterval, SqrtStretch,
                                   ImageNormalize, LogStretch)

import pickle
from astropy.stats import sigma_clip



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

def get_pixel_data(exp, amp):
    
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
    amp = None
    n_comp_x = None
    n_comp_y = None
    
    def __init__(self, biases, amp, ncomp_x=6, ncomp_y=8):
        self.n_comp_x = ncomp_x
        self.n_comp_y = ncomp_x
        self.serial_pca = PCA(n_components=ncomp_x)
        self.parallel_pca = PCA(n_components=ncomp_y)
        self.amp = amp
        
        bias_data = []
        for exp in biases:
            bias_data.append(get_pixel_data(exp, self.amp))
        
        bias_data_arr = pd.DataFrame(bias_data)
        
        serial_means = [mean_serial(this_bias_data) for this_bias_data in bias_data_arr.to_dict('records')]
        parallel_means = [mean_parallel(this_bias_data) for this_bias_data in bias_data_arr.to_dict('records')]
        
        self.serial_pca.fit(serial_means)
        self.parallel_pca.fit(parallel_means)
        
    def correct(self, exp: lsst.afw.image.exposure.ExposureF):
        return remove_pca_both(get_pixel_data(exp,self.amp), self.serial_pca, self.parallel_pca)
    
def plot_superbias(superbias, nbiases, dataname, title=None, outname=None, show_plots=False):
    if title is None:
        title = f'Superbias: mean of {nbiases} exposures post-PCA oscan in {dataname}'
    fig = plt.figure(dpi=150, facecolor='white')
    plt.subplot(211)
    norm = ImageNormalize(superbias, interval=PercentileInterval(99))
    plt.imshow(np.rot90(superbias),norm=norm, cmap='gist_heat')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('ADU')
    
    plt.subplot(212)
    plt.hist(superbias.flat, bins=100);
    plt.xlabel('ADU')
    plt.ylabel('Bin count')
    plt.yscale('log')
    plt.suptitle(title, wrap=True)
    
    if outname is not None:
        plt.savefig(outname)
    if show_plots:
        plt.show()
    plt.close()
    
def do_dark_clipping(corrected_dark_arr):
    original_shape = corrected_dark_arr.shape
    corrected_dark_arr_clipped = sigma_clip(corrected_dark_arr.reshape(corrected_dark_arr.shape[0], -1), sigma=3, masked=False, axis=0)
    pixel_means = np.nanmean(corrected_dark_arr_clipped, axis=0) #fill clipped values with mean of the rest 
    clipped_inds = np.where(np.isnan(corrected_dark_arr_clipped))
    corrected_dark_arr_clipped[clipped_inds] = np.take(pixel_means, clipped_inds[1])
    corrected_dark_arr = corrected_dark_arr_clipped.reshape(corrected_dark_arr.shape)
    return corrected_dark_arr

def make_superbias_and_dark_from_corrected_clipped(corrected_bias_arr, corrected_dark_arr, bias_dark_times, dark_times):
                        
    flats = np.concatenate([[np.ravel(arr) for arr in corrected_bias_arr], [np.ravel(arr) for arr in corrected_dark_arr]])
    params = sklearn.linear_model.LinearRegression().fit(np.concatenate([bias_dark_times, dark_times]).reshape(-1, 1),flats)
    return np.reshape(params.intercept_ , corrected_bias_arr[0].shape), np.reshape(params.coef_, corrected_bias_arr[0].shape), params

def do_PCA_correction(exps, corrector, verbose=False):
    corrected_arr = []
    i=1
    for exp in exps:
        if verbose:
            print(f'Correcting {i} of {len(exps)}',end='\r')
        corrected = corrector.correct(exp)
        corrected_arr.append(corrected)
        i=i+1
    corrected_arr = np.asarray(corrected_arr)
    if verbose:
        print(f'Done correcting {len(correcteds_arr)} biases')
    return corrected_arr

def make_superbias_and_dark_regression(amp, dataname, bias_data, dark_data, sensor_out_dir, clip_darks=True, median_darks=False, median_biases=False, show_plots=False, verbose=False):

    corrector = PCA_Corrector(bias_data, amp)

    corrected_bias_arr = do_PCA_correction(bias_data, corrector, verbose=verbose)

    superbias = np.mean(corrected_bias_arr, axis=0)

    plot_superbias(superbias, len(corrected_bias_arr), dataname, outname=f'{sensor_out_dir}/{dataname}-superbias.png', show_plots=show_plots)
    pickle.dump(superbias, open(f'{sensor_out_dir}/{dataname}-superbias.pickle', "wb"))

    corrected_dark_arr = do_PCA_correction(dark_data, corrector, verbose=verbose)
    if verbose:
        print(f'Done correcting {len(corrected_dark_arr)} darks')

    dark_times = np.array([exp.getMetadata()['DARKTIME'] for exp in dark_data])
    if median_darks:
        dark_times = np.asarray([np.median(dark_times)])
        corrected_dark_arr = np.median(corrected_dark_arr, axis=0)
        corrected_dark_arr = np.asarray([corrected_dark_arr])

    bias_dark_times = np.asarray([exp.getMetadata()['DARKTIME'] for exp in bias_data])
    if median_biases:
        bias_dark_times = np.asarray([np.mean(bias_dark_times)])
        corrected_bias_arr = np.mean(corrected_bias_arr, axis=0)
        corrected_bias_arr = np.asarray([corrected_bias_arr])

    if clip_darks:
        corrected_dark_arr = do_dark_clipping(corrected_dark_arr)

    regression_superbias, regression_dark_current, params = \
        make_superbias_and_dark_from_corrected_clipped(corrected_bias_arr, corrected_dark_arr, bias_dark_times, dark_times)

    pickle.dump(regression_superbias, open(f'{sensor_out_dir}/{dataname}-superbias-regression.pickle', "wb"))

    plot_superbias(regression_superbias, len(corrected_bias_arr)+len(corrected_dark_arr), 
                   dataname, title=f'Superbias: regression intercept of {len(corrected_bias_arr)} biases and {len(corrected_dark_arr)} darks post-PCA oscan in {dataname}', outname=f'{sensor_out_dir}/{dataname}-superbias-regression.png', show_plots=show_plots)
    
    return superbias, regression_superbias, regression_dark_current, params

def get_selected_bias_and_dark_visits(butler, run, raft, sensor):
        visits = butler.queryMetadata('raw', ['visit', 'imageType'], 
                                        dataId={'run': run,'raftName': raft,'detectorName': sensor})#6790D
        biases = butler.queryMetadata('raw', ['visit'], 
                                    dataId={'run': run, 'imageType' : 'BIAS','raftName': raft,'detectorName': sensor})
        if len(biases) < 8:
            print('Too few biases, skipping...')
            return None
        
        darks = butler.queryMetadata('raw', ['visit'], 
                                    dataId={'run': run, 'imageType' : 'DARK','raftName': raft,'detectorName': sensor})
        if len(darks) < 3:
            print('Too few darks, skipping...')
            return None
        visit_arr = np.asarray(visits, dtype=[('visit',int), ('test','S10')])
        visit_arr = np.sort(visit_arr)
        non_saturated_biases_mask = np.logical_not(visit_arr[np.arange(0,len(visit_arr))[visit_arr['test']==b'BIAS']-1]['test']==b'FLAT')
        non_saturated_biases = np.arange(0,len(biases))[non_saturated_biases_mask]
        non_saturated_bias_visits = list(np.sort(np.asarray(biases))[non_saturated_biases])[1:]
        
        return non_saturated_bias_visits, darks
    
def run_dark_regression_analysis_on_a_sensor(butler, run, raft, sensor, sensor_out_dir, show_plots=False, nbiases=None):
    visit_selection = get_selected_bias_and_dark_visits(butler, run, raft, sensor)
    if visit_selection is None:
        return None
    bias_selection, darks = visit_selection

    if nbiases is not None:
        bias_selection = random.sample(bias_selection, nbiases)

    bias_data = []
    exp=None
    i=0
    for visit in bias_selection:
        print(f'Importing bias {i+1}/{len(bias_selection)}    ',end='\r')
        testtype = 'BIAS'
        try:
            exp = butler.get('raw',dataId={'run': run, 'visit':int(visit),'raftName': raft,'detectorName': sensor})
            bias_data.append(exp)
        except:
            print(f'Butler failed to get bias visit {visit}')
        i=i+1
    print(f'Done importing {i}/{len(bias_selection)} biases   ')

    dark_data = []
    i=0
    for visit in darks:
        print(f'Importing dark {i+1}/{len(darks)}    ',end='\r')
        testtype = 'DARK'
        try:
            exp = butler.get('raw',dataId={'run': run, 'visit':visit,'raftName': raft,'detectorName': sensor})

            dark_data.append(exp)
        except:
            print(f'Butler failed to get dark visit {visit}')
        i=i+1
    print(f'Done importing {i}/{len(darks)} darks        ')


    det = exp.getDetector()
    for amp_num, amp in enumerate(det):
        amp_name = amp.getName()
        dataname = f'{run}-{raft}-{sensor}-{amp_name}'
        superbias, regression_superbias, regression_dark_current, params = \
            make_superbias_and_dark_regression(amp, dataname, bias_data, dark_data, sensor_out_dir, show_plots=show_plots)
    
def run_dark_regression_analysis_on_all_sensors(butler, runs, sensors, nbiases = None, show_plots=False, color_lims = (-5,5), \
                                                last_amp_name = 'C00', base_dir = 'dark_bias', \
                                                median_darks = False, median_biases = False, clip_darks = True):
    
    sensors_selection = sensors

    try:
        os.mkdir(base_dir)
    except:
        pass

    for raft, sensor in sensors_selection:

        print(f'{raft} - {sensor}')
        try:
            os.mkdir(f'{base_dir}/{raft}-{sensor}')
        except:
            pass

        sensor_out_dir = f'{base_dir}/{raft}-{sensor}'

        for run in runs:
            data_prefix = f'{sensor_out_dir}/{run}-{raft}-{sensor}-{last_amp_name}'
            if os.path.exists(f'{data_prefix}-superbias.png'):
                print(f'{run}-{raft}-{sensor} already present in directory, skipping...')
                continue
            print(run)

            analysis_results = run_dark_regression_analysis_on_a_sensor(butler, run, raft, sensor, sensor_out_dir, show_plots=show_plots, nbiases=nbiases)
            if analysis_results is None:
                continue
            superbias, regression_superbias, regression_dark_current, params = analysis_results


            

def __main__(args):
    repo_directory = "/lsstdata/offline/teststand/BOT/gen2repo"
    butler = Butler(repo_directory)
    runs = ['12672','12673', '12844', '12845','12853','12855']
    run = runs[0]
    sensors = set(butler.queryMetadata('raw', ['raftName','detectorName'], dataId={'run': run, 'imageType' : 'BIAS'}))
    
    run_dark_regression_analysis_on_all_sensors(butler, runs, sensors, show_plots=False)