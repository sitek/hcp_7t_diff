import os
from glob import glob

from nipype import config
config.set('execution', 'remove_unnecessary_outputs', 'false')
config.set('execution', 'crashfile_format', 'txt')
from nipype import (Node, MapNode, Function, DataGrabber,
                    Workflow, IdentityInterface)
from nipype.interfaces.io import SelectFiles, DataSink

''' define data '''
t_r = 1
vox_size = 1.6

target_atlas = os.path.join('/om2/user/ksitek/reference/',
                            'mni_icbm152_nlin_asym_09b/',
                            'sub-bigbrain_MNI_100um_bstem_'\
                            'corrected_rois_KRS02_OFG02_conjunction.nii.gz')

''' set smoothing inputs '''
#fwhm_list = [round(x,1) for x in [1*vox_size, 2*vox_size, 3*vox_size]]
#fwhm_list = [vox_size]
fwhm = vox_size*2

low_pass = 0.1
high_pass = 0.001

''' define directories '''
analysis = 'rs_corr_fwhm-%.01f_temp-0.001-0.1Hz'%fwhm
work_dir = os.path.join('/om2/scratch/ksitek/hcp/', analysis)
project_dir = os.path.abspath('/om2/user/ksitek/hcp_7t/')
out_dir = os.path.join(project_dir, 'derivatives', analysis)

''' define subjects '''
#sub_list = [os.path.basename(x) for x in sorted(glob(project_dir+'/data/13*'))]
sub_list = ['131217','131722','132118'] # test on one subject


''' set up nodes '''
# set up iterables
infosource = Node(IdentityInterface(fields=['subject_id']),#, 'acq_dir', 'fwhm']),
                                    name='infosource')
infosource.iterables = [('subject_id', sub_list)]#,
                        #('acq_dir', acq_dir_list),
                        #('fwhm', fwhm_list)]

# Create DataGrabber node
dg = Node(DataGrabber(infields=['subject_id'],#, 'acq_dir'],
                      outfields=['anat', 'atlas', 'func']),
          name='datagrabber')

# Location of the dataset folder
dg.inputs.base_directory = os.path.join(project_dir, 'data')

# Necessary default parameters
dg.inputs.template = '*'
dg.inputs.sort_filelist = True

dg.inputs.template_args = {'anat': [['subject_id']],
                           'atlas' : [['subject_id']],
                           'func': [['subject_id']]}
dg.inputs.field_template = {'anat': '%s/MNINonLinear/T1w_restore.1.60.nii.gz',
                            'atlas': '%s/MNINonLinear/ROIs/Atlas_ROIs.1.60.nii.gz',
                            'func': '%s/MNINonLinear/Results/rfMRI_REST*_7T_*/'\
                                    'rfMRI_REST*_7T_*.nii.gz'}

''' create subject-specific subcortical mask '''
def make_mask(atlas):
    import nibabel as nib
    import numpy as np
    import os
    subcort_labels = [16, 28, 60, 10, 49] # brainstem, L/R VentralDC, L/R thalamus

    atlas_img = nib.load(atlas)
    atlas_data = atlas_img.get_data()
    affine = atlas_img.affine

    mask_data = np.isin(atlas_data, subcort_labels).astype('uint8')
    mask_img = nib.Nifti1Image(mask_data, affine=affine)
    mask_filename = os.path.abspath('subcortical_mask.nii.gz')
    nib.save(mask_img, mask_filename)

    return mask_filename
masker = Node(Function(input_names=['atlas'],
                       output_names=['mask_filename'],
                       function=make_mask), name='masker')


''' filter functional images '''
# clean images
def run_clean_img(filename, t_r, low_pass, high_pass):
    import os
    import nibabel as nib
    from nilearn.image import clean_img

    func_bandpassed = clean_img(filename, detrend=True, t_r=t_r,
                                 low_pass=low_pass, high_pass=high_pass)

    clean_func_fname = os.path.abspath('filtered_func.nii.gz')
    nib.save(func_bandpassed, clean_func_fname)
    print('bandpassed run saved to %s'%clean_func_fname)

    return clean_func_fname
img_cleaner = MapNode(Function(input_names=['filename', 't_r',
                                        'low_pass', 'high_pass'],
                            output_names=['clean_func'],
                            function=run_clean_img), name='img_cleaner',
                            iterfield=['filename'])
img_cleaner.inputs.t_r = t_r
img_cleaner.inputs.low_pass = low_pass
img_cleaner.inputs.high_pass = high_pass

''' extract auditory label regions '''
def extract_region_mask(target_atlas, anat):
    import os
    import nibabel as nib
    import numpy as np
    from nilearn.image import resample_to_img

    atlas_img = nib.load(target_atlas)
    atlas_affine = atlas_img.affine
    anat_img = nib.load(anat)
    anat_affine = anat_img.affine

    atlas_data = atlas_img.get_fdata().round()
    atlas_masks = []
    #for rval in [8]:
    for rval in np.unique(atlas_data):
        if rval == 0:
            continue
        mask_data = np.isin(atlas_data, rval).astype('uint8')
        mask_img = nib.Nifti1Image(mask_data, affine=atlas_affine)

        '''
        # save in original atlas space
        mask_fname = os.path.abspath('roi-%02d_mask_native-sample.nii.gz'%rval)
        nib.save(mask_img, mask_fname)
        '''

        # resample to subject space
        mask_resample = resample_to_img(mask_img, anat_img)
        mask_fname = os.path.abspath('roi-%02d_mask_resampled.nii.gz'%rval)
        nib.save(mask_resample, mask_fname)

        atlas_masks.append(mask_fname)
    return atlas_masks
region_extracter = Node(Function(input_names=['target_atlas','anat'],
                                 output_names=['atlas_masks'],
                                 function=extract_region_mask),
                        name='region_extracter')
region_extracter.inputs.target_atlas = target_atlas

''' extract timeseries from auditory label regions '''
def extract_mean_ts(functional_run, atlas_masks, fwhm):
    import nibabel as nib
    import numpy as np
    from nilearn.input_data import NiftiMasker
    import os

    func_img = nib.load(functional_run)
    func_data = func_img.get_fdata()

    mean_ts = []
    for ax, atlas_mask in enumerate(sorted(atlas_masks)):
        mask_img = nib.load(atlas_mask)
        region_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=fwhm,
                                     mask_strategy='epi')
        func_data_2D = region_masker.fit_transform(func_img)

        # compute mean timeseries
        single_mean_ts = func_data_2D.mean(axis=1)

        # save mean time series to txt file
        single_mean_ts_file = os.path.abspath('roi-%02d_mean_ts.txt'%ax)
        np.savetxt(single_mean_ts_file, single_mean_ts, fmt='%f')

        mean_ts.append(single_mean_ts_file)
    return mean_ts
# create node that takes single run and all masks
# (can't do different-length iterfields with MapNode)
mean_ts_extracter = MapNode(Function(input_names=['functional_run',
                                                  'atlas_masks',
                                                  'fwhm'],
                                     output_names=['mean_ts'],
                                     function=extract_mean_ts),
                            name='mean_ts_extracter',
                            iterfield=['functional_run'])
mean_ts_extracter.inputs.fwhm = fwhm

''' correlate ROI timeseries with rest of subcortical voxels '''
def correlate_roi_voxels(functional_run, mask, mean_ts, fwhm):
    import nibabel as nib
    import numpy as np
    import os

    func_img = nib.load(functional_run)
    func_data = func_img.get_fdata()
    affine = func_img.affine

    # flatten 4D data to 2D using NiftiMasker
    from nilearn.input_data import NiftiMasker
    subcort_masker = NiftiMasker(mask_img=mask,
                                 smoothing_fwhm=fwhm,
                                 mask_strategy='epi')
    func_data_2D = subcort_masker.fit_transform(func_img)

    all_roi_corr_files = []
    for mx, mean_ts_file in enumerate(mean_ts):
        print('roi %d mean ts file: %s'%(mx, mean_ts_file))
        single_roi_mean_ts = np.loadtxt(mean_ts_file)
        print(single_roi_mean_ts.shape)

        # turn list into 2D array:
        single_ts = np.asarray(single_roi_mean_ts).reshape(-1,1)

        # correlate ROI timeseries with 2D functional data
        # must have single timeseries first! that way can just take
        # output[0,1:] later for all the correlation coefficients
        corr_data_mat = np.corrcoef(single_ts, func_data_2D, rowvar=False)

        # Fisher transform correlation values
        corr_data_2D = np.arctanh(corr_data_mat)[0,1:]

        # turn 2D correlation data back into 4D
        corr_data = subcort_masker.inverse_transform(corr_data_2D)

        # save image
        roi_corr_file = os.path.abspath('roi-%02d_correlations.nii.gz'%mx)
        nib.save(corr_data, roi_corr_file)

        # add filename to output list
        all_roi_corr_files.append(roi_corr_file)

    return all_roi_corr_files
# create node that takes single run, all ROIs' mean timeseries:
roi_voxel_correlater = MapNode(Function(input_names=['functional_run',
                                                     'mask',
                                                     'mean_ts',
                                                     'fwhm'],
                                        output_names=['all_roi_corr_files'],
                                        function=correlate_roi_voxels),
                               name='roi_voxel_correlater',
                               iterfield=['functional_run', 'mean_ts'])
roi_voxel_correlater.inputs.fwhm = fwhm


''' datasink '''
ds = Node(DataSink(parameterization=False), name='sinker')
ds.inputs.base_directory = out_dir
ds.inputs.parameterization = False
ds.plugin_args = {'overwrite': True}

''' create workflow '''
wf = Workflow(name='rs_atlas')
wf.config['execution']['crashfile_format'] = 'txt'

wf.connect(infosource, 'subject_id', dg, 'subject_id')

wf.connect(dg, 'atlas', masker, 'atlas')
wf.connect(dg, 'func', img_cleaner, 'filename')

wf.connect(dg, 'anat', region_extracter, 'anat')

wf.connect(region_extracter, 'atlas_masks', mean_ts_extracter, 'atlas_masks')
wf.connect(img_cleaner, 'clean_func', mean_ts_extracter, 'functional_run')
#wf.connect(dg, 'func', mean_ts_extracter, 'functional_run')

wf.connect(img_cleaner, 'clean_func', roi_voxel_correlater, 'functional_run')
#wf.connect(dg, 'func', roi_voxel_correlater, 'functional_run')
wf.connect(masker, 'mask_filename', roi_voxel_correlater, 'mask')
wf.connect(mean_ts_extracter, 'mean_ts', roi_voxel_correlater, 'mean_ts')

wf.connect(infosource, 'subject_id', ds, 'container')
wf.connect(masker, 'mask_filename', ds, 'subcortical_mask')
wf.connect(img_cleaner, 'clean_func', ds, 'temp_filter')
wf.connect(roi_voxel_correlater, 'all_roi_corr_files', ds, 'seed_correlations')
wf.connect(mean_ts_extracter, 'mean_ts', ds, 'seed_timeseries')

wf.base_dir = work_dir

wf.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
