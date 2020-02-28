import matplotlib.pyplot as plt
import lsst.afw
import numpy as np
import lsst.eotest.image_utils as imutils
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.eotest.sensor.AmplifierGeometry import makeAmplifierGeometry
import lsst.ip.isr as isr

def draw_biases(exposures, amp_num=None, raft=None, overscan_correct=False, save_fig=False, same_scale=True):
    figdims = (10,8)
    if amp_num is None:
        figdims = (15,5)
    
    fig, axes = plt.subplots(nrows=1, ncols=len(exposures), figsize=figdims)
    clow, chigh = None, None
    run_num = exposures[0].getInfo().getMetadata()['RUNNUM']
    lsst_num = exposures[0].getInfo().getMetadata()['LSST_NUM']    
    raft = exposures[0].getInfo().getMetadata()['RAFTBAY']
    detector_name = exposures[0].getInfo().getMetadata()['CCDSLOT']
    

    arrays = []
    
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
            del im
        else:
            raw_clone = raw.clone()
            
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
    plt.close()
