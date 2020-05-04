import matplotlib.pyplot as plt
import lsst.afw
import lsst.afw.math as afwMath
import numpy as np
import lsst.eotest.image_utils as imutils
from pathlib import Path
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

import csv

from lsst.eotest.sensor.AmplifierGeometry import makeAmplifierGeometry
import lsst.ip.isr as isr

def draw_bias_sequence(exposures, amp_num=None, overscan_correct=False, \
                save_fig=False, same_scale=True, run_num=None, lsst_num=None, \
                raft=None, detector_name=None, plot_medians=False):

    figdims = (10,8)
    if amp_num is None:
        figdims = (15,5)
    
    fig, axes = plt.subplots(nrows=1, ncols=len(exposures), figsize=figdims)
    clow, chigh = None, None
    if run_num is None: run_num = exposures[0].getInfo().getMetadata()['RUNNUM']
    if lsst_num is None: lsst_num = exposures[0].getInfo().getMetadata()['LSST_NUM']    
    if raft is None: raft = exposures[0].getInfo().getMetadata()['RAFTBAY']
    if detector_name is None: detector_name = exposures[0].getInfo().getMetadata()['CCDSLOT']
    

    arrays = []
    medians = []
    ampnames = []
    
    for i in range(len(exposures))[::-1]:
        
        raw = exposures[i]
        detector = raw.getDetector()
        
        array = None
        
        if amp_num is not None:
            amp = detector[amp_num]
            im = raw.getMaskedImage()[amp.getRawBBox()].clone()
            if overscan_correct:
                isr.overscanCorrection(im[amp.getRawDataBBox()], im[amp.getRawHorizontalOverscanBBox()], fitType='MEDIAN_PER_ROW')
            array = im[amp.getRawDataBBox()].getImage().getArray().copy()
            
            if plot_medians:
                medians.append(np.median(array))
            del im
        else:
            raw_clone = raw.clone()
            
            median_by_amp = []
            
            if plot_medians:
                for amp in detector:
                    median = afwMath.makeStatistics(raw.getMaskedImage()[amp.getRawDataBBox()], afwMath.MEDIAN).getValue()
                    median_by_amp.append(median)
                    if not len(ampnames) == 16:
                        ampnames.append(amp.getName())
                
            #print(median_by_amp)
            medians.append(median_by_amp)
            
            task = isr.isrTask.IsrTask()
            task.config.doAssembleCcd = True
            task.assembleCcd.config.doTrim = True
            task.config.overscanFitType = 'MEDIAN_PER_ROW'
            task.config.doOverscan = overscan_correct
            task.config.doBias = False
            task.config.doLinearize = False
            task.config.doDark = False
            task.config.doFlat = False
            task.config.doDefect = False
            
            assembled = task.run(raw_clone).exposure
            array = assembled.getImage().getArray().copy()
            del assembled, raw_clone
            
            
        
        if not same_scale: 
            implot = axes[i].imshow(array,  cmap='gist_heat')  
            axes[i].set_axis_off()
            divider = make_axes_locatable(axes[i])
            cax = divider.new_vertical(size="5%", pad=0.1, pack_start=True)
            fig.add_axes(cax)
            cbar = fig.colorbar(implot, cax=cax, orientation="horizontal")
            cbar.set_label('Raw ADU'.format(' (Median Row Overscan Corrected)' if overscan_correct else ''))
            continue
            
        arrays.append(array)
        
        this_clow = np.amin(array)
        this_chigh = np.amax(array)
        if clow is None:
            clow = this_clow
        elif this_clow < clow:
            clow = this_clow
        if chigh is None:
            chigh = this_chigh
        elif this_chigh > chigh:
            chigh = this_chigh

    fig.patch.set_facecolor('white')
    fig.suptitle('Consecutive biases in run {}, {} {} {} : {}'.format(run_num, raft, detector_name, '' if (amp_num is None) else 'Amp {}'.format(amp_num), lsst_num))
        
    if same_scale:            
        for i in range(len(exposures))[::-1]:
            array = arrays[::-1][i]
            im = axes[i].imshow(array, clim=(clow,chigh), cmap='gist_heat')
            axes[i].set_axis_off()
        if amp_num is None: fig.tight_layout(rect=[0,0,1,1.1]) 
        else: fig.tight_layout(rect=[0,0,1,0.95]) 
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7)
        cbar.set_label('Raw ADU{}'.format(' (Median Row Overscan Corrected)' if overscan_correct else ''))
    else:
        fig.tight_layout(rect=[0,0,1,.95])
    
    
    if save_fig:
        im_path = '{}/{}/{}'.format(run_num, raft, detector_name)
        Path(im_path).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/biases_{}_{}_{}_Amp{:02d}'.format(im_path, run_num, raft, detector_name, amp_num))

    plt.show()    
    
    if plot_medians:
        fig = plt.figure()
        plt.plot([str(x) for x in range(1,len(exposures)+1)],medians[::-1])
        fig.patch.set_facecolor('white')
        plt.xlabel('Exposure')
        plt.ylabel('Median Pixel Value (ADU)')
        plt.legend(ampnames, bbox_to_anchor=(1.25, 1))
        plt.show()
    plt.close()
    return np.array(medians)



def bias_sequence(exposures, amp_num=None, overscan_correct=False, \
                save_fig=False, same_scale=True, run_num=None, lsst_num=None, \
                raft=None, detector_name=None, plot_medians=True, outfile=None):
    
    if run_num is None: run_num = exposures[0].getInfo().getMetadata()['RUNNUM']
    if lsst_num is None: lsst_num = exposures[0].getInfo().getMetadata()['LSST_NUM']    
    if raft is None: raft = exposures[0].getInfo().getMetadata()['RAFTBAY']
    if detector_name is None: detector_name = exposures[0].getInfo().getMetadata()['CCDSLOT']
    

    arrays = []
    medians = []
    higherrs = []
    lowerrs = []
    ampnames = []
    
    for i in range(len(exposures)):
        
        raw = exposures[i]
        detector = raw.getDetector()
        
        array = None
        
        if amp_num is not None:
            amp = detector[amp_num]
            im = raw.getMaskedImage()[amp.getRawBBox()].clone()
            if overscan_correct:
                isr.overscanCorrection(im[amp.getRawDataBBox()], im[amp.getRawHorizontalOverscanBBox()], fitType='MEDIAN_PER_ROW')
            array = im[amp.getRawDataBBox()].getImage().getArray().copy()
            median = np.median(array)
            higherr = np.percentile(array, 84)
            lowerr  = np.percentile(array, 16)            
            
            medians.append(median)
            higherrs.append(higherr)
            lowerrs.append(lowerr)
                     
            del im
        else:
            
            median_by_amp = []
            higherr_by_amp = []
            lowerr_by_amp = []
            
            if plot_medians:
                for amp in detector:
                    
                    im = raw.getMaskedImage()[amp.getRawDataBBox()]
                    array = im.getImage().getArray().copy()
                    median = np.median(array)
                    higherr = np.percentile(array, 84) - median
                    lowerr  = np.percentile(array, 16) - median  
                    
                    median_by_amp.append(median)
                    higherr_by_amp.append(higherr)
                    lowerr_by_amp.append(lowerr)
                    
                    if not len(ampnames) == 16:
                        ampnames.append(amp.getName())
                
            medians.append(median_by_amp)
            higherrs.append(higherr_by_amp)
            lowerrs.append(lowerr_by_amp)
            
    
    medians_arr = np.array(medians)
    higherrs_arr = np.array(higherrs)
    lowerrs_arr = np.array(lowerrs)
    
    
    if plot_medians:
        fig = plt.figure()
        for i in range(len(ampnames)):
            plt.errorbar([str(x) for x in range(1,len(exposures)+1)],medians_arr[:,i], yerr=[lowerrs_arr[:,i], higherrs_arr[:,i]])
        fig.patch.set_facecolor('white')
        plt.xlabel('Exposure')
        plt.ylabel('Median Pixel Value (ADU)')
        plt.legend(['{}: std = {:.1f} ADU'.format(ampnames[i], np.std(medians_arr[1:,i])) for i in range(len(ampnames))], bbox_to_anchor=(1.25, 1))
        fig.suptitle('Consecutive biases in run {}, {} {} {} : {}'.format(run_num, raft, detector_name, '' if (amp_num is None) else 'Amp {}'.format(amp_num), lsst_num))
        plt.show()
    plt.close()
    
    if not outfile is None:
        filename = 'bias_seqs/{}/bias_seqs_{}_{}_{}_{}.csv'.format(lsst_num, run_num, raft, detector_name,lsst_num)
        if not outfile == True:
            filename = outfile
        else: 
            outdir = 'bias_seqs/{}'.format(lsst_num)
            os.makedirs(outdir, exist_ok=True)
            
        out = csv.writer(open(filename,'w'))
        header = ['Amp', 'Exposure Number', 'Median Bias (ADU)', 'High Error (ADU)', 'Low Error (ADU)']
        out.writerow(header)
        for ampnum in range(len(ampnames)):
            for expnum in range(len(exposures)):
                row = [ampnames[ampnum], expnum,  medians_arr[expnum, ampnum], higherrs_arr[expnum, ampnum], lowerrs_arr[expnum, ampnum]]
                out.writerow(row)
    return medians_arr, higherrs_arr, lowerrs_arr




def find_bias_anomalies(exposures, amp_num=None, verbose=False,\
                run_num=None, lsst_num=None, raft=None, detector_name=None, \
                        plot_medians=False, outfile=None, anomaliesfile=None, recompute=False):
    
    if run_num is None: run_num = exposures[0].getInfo().getMetadata()['RUNNUM']
    if lsst_num is None: lsst_num = exposures[0].getInfo().getMetadata()['LSST_NUM']    
    if raft is None: raft = exposures[0].getInfo().getMetadata()['RAFTBAY']
    if detector_name is None: detector_name = exposures[0].getInfo().getMetadata()['CCDSLOT']
    
    
    if not anomaliesfile is None:
        filename = 'bias_anomalies/bias_anomalies.csv'
        if not outfile == True:
            filename = anomaliesfile
        else: 
            outdir = 'bias_anomalies'
            os.makedirs(outdir, exist_ok=True)
        
        try:
            with open(filename, 'r') as in_anomalies:
                read_anomalies = csv.reader(in_anomalies)
                for line in read_anomalies:
                    if line[0] == run_num and line[1] == raft \
                    and line[2] == detector_name and line[3] == lsst_num:
                        if recompute == False:
                            print('Sensor {} - {} - {} has been checked for run {}'.format(raft, detector_name,lsst_num, run_num))
                            return
        except IOError:
            with open(filename,'w') as f:
                out_anomalies = csv.writer(f)
                header = ['Run', 'Raft Slot', 'CCD Slot', 'LSST Name', 'Anomaly Significance', 'Structure']
                out_anomalies.writerow(header)
        anomaliesfile = filename
            
           
    if not outfile is None:
        filename = 'bias_anomalies/{}/bias_anomalies_{}_{}_{}_{}.csv'.format(run_num, run_num, raft, detector_name,lsst_num)
        if not outfile == True:
            filename = outfile
        else: 
            outdir = 'bias_anomalies/{}'.format(run_num)
            os.makedirs(outdir, exist_ok=True)
        try:
            with open(filename, 'r') as f:
                print('Sensor {} - {} - {} has been checked for run {}'.format( raft, detector_name,lsst_num, run_num))
                return
        except IOError:
            with open(filename,'w') as f:
                out = csv.writer(f)
                header = ['Amp', 'First Exposure Bias (ADU)', 'Median Without First (ADU)', 'Median Variance (ADU)', 'Variation of First (ADU)', 'Anomaly Significance', 'Structure']
                out.writerow(header)
        outfile = filename
        
    try: 
        f = open("bias_anomalies/checked.csv",'r')
        f.close()
    except IOError:
        with open("bias_anomalies/checked.csv", 'w') as f:
            out = csv.writer(f)
            header = ['Run', 'Raft Slot', 'CCD Slot', 'LSST Name', 'Anomaly Significance', 'Structure']
            out.writerow(header)

    
    ampnames = []
    
    first_biases  = []
    medians       = []
    median_devs   = []
    diffs         = []
    significances = []
    structures    = []
    
    
    detector = exposures[0].getDetector()
    
    for ampnum in range(16):
        
        if (amp_num is not None) and not ampnum == amp_num: continue
        
        amp = detector[ampnum]
        ampnames.append(amp.getName())
        
        im_list = []
        
        for expnum in range(1, len(exposures)):
            im_list.append(exposures[expnum].getMaskedImage()[amp.getRawDataBBox()].getImage().getArray())
        
        median_im = np.median(np.array(im_list), axis=0)
        
        median = np.median(median_im)
        first_median = np.median(exposures[0].getMaskedImage()[amp.getRawDataBBox()].getImage().getArray())  
        
        med_dev = np.mean(np.square(np.array([im - median_im for im in im_list])))
        
        diff_im = exposures[0].getMaskedImage()[amp.getRawDataBBox()].getImage().getArray() - median_im
        
        diff_ms = np.mean(np.square(diff_im))
        diff_rms = np.sqrt(diff_ms)
        
        structure = np.sqrt(diff_ms - (median - first_median)**2)/med_dev
        
        if diff_rms/med_dev >= 1 and verbose:
            print('Amp {} - Exposure 0 deviation= {}; Mean of 1-4 deviations= {}'.format(ampnames[ampnum], diff_rms, med_dev))
  
        
        first_biases.append(first_median)
        medians.append(median)
        median_devs.append(med_dev)
        diffs.append(diff_rms)
        significances.append(diff_rms/med_dev)
        structures.append(structure)
        
    if outfile is not None:
        with open(outfile, 'a') as f:
            out = csv.writer(f)
            for i in range(len(medians)):
                out.writerow([ampnames[i], first_biases[i], medians[i], median_devs[i], diffs[i], significances[i], structures[i]])
             
    if max(significances) > 1:
        with open(anomaliesfile, 'a')  as f:
            out = csv.writer(f)
            out.writerow([run_num, raft, detector_name, lsst_num, max(significances), max(structures)])
            
    try: 
        with open("bias_anomalies/checked.csv", 'a') as f:
            out = csv.writer(f)
            out.writerow([run_num, raft, detector_name, lsst_num, max(significances), max(structures)])
    except IOError:
        with open("bias_anomalies/checked.csv", 'w') as f:
            out = csv.writer(f)
            header = ['Run', 'Raft Slot', 'CCD Slot', 'LSST Name', 'Anomaly Significance', 'Structure']
            out.writerow(header)
            out.writerow([run_num, raft, detector_name, lsst_num, max(significances), max(structures)])
    
    return

def is_checked(run, raft, ccd):
    try:
        with open('bias_anomalies/checked.csv', 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if line[:3] == [run,raft,ccd]:
                    return True
            return False
    except IOError:
        return False
    
def get_bias_exposures(detector_name, raft_name, chosen_runs, nexps=5, verbose=False):
    exposures = []
    for run in chosen_runs:
        num_visits = min(len(biases_by_run[run]), nexps)
        if num_visits <= 2: 
            if verbose: print('Too few exposures in run {}'.format(run))
            continue
        first_few_visits = biases_by_run[run][:num_visits]
        first_few_images = []
        if verbose: print('Run {}'.format(run))
        for visit in first_few_visits:
            dId = {'visit': visit, 'raftName': raft_name,'detectorName': detector_name}
            if verbose: print(dId)
            raw = butler.get('raw', dId)
            first_few_images.append(raw)
        exposures.append(first_few_images)
    return exposures

def find_anomalies_in_runs(runs):
    logger = logging.basicConfig(filename='anomalies.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    chosen_runs = runs
    fail_count = 0
    success_count = 0
    for run in chosen_runs:
        print("Run {}".format(run))
        for raft in rafts:
            for ccd in slots:
                logging.info('*** Trying: {}, {}, {} ***'.format(run,raft,ccd))
                fail_count += 1
                if is_checked(run, raft, ccd):
                    logging.info('Already checked: {}, {}, {}'.format(run,raft,ccd))
                    continue
                try:
                    exposures = get_bias_exposures(ccd, raft, [run], verbose=False)
                    exposure = exposures[0]
                except :
                    logging.warning("NoResult error\n")
                    continue
                bias_tools.find_bias_anomalies(exposure, plot_medians=True, outfile=True, anomaliesfile=True)
                del exposures
                logging.info("Success\n")
                success_count += 1
                fail_count -= 1

        
    
        