'''
Create nipype workflow using dipy's workflows
https://github.com/nipy/dipy/blob/master/dipy/workflows/
'''
from nipype import config
config.set('execution', 'remove_unnecessary_outputs', 'false')
#config.set('execution', 'crashfile_format', 'txt')

from nipype import Node, Function, Workflow, DataGrabber, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink

import os
from glob import glob

# define inputs
recon = 'csd'
analysis = 'dipy_0.16_workflows_%s'%recon
num_threads = 4
b0_thresh = 80

project_dir = os.path.abspath('/om2/user/ksitek/hcp_7t/')
data_dir = os.path.join(project_dir, 'data')
out_dir = os.path.join(project_dir, 'derivatives', analysis)
work_dir = os.path.join('/om2/scratch/ksitek/hcp/', analysis)

''' define subjects '''
sub_list = [os.path.basename(x) for x in sorted(glob(project_dir+'/data/13*'))]
#sub_list = ['100610'] # test on one subject

''' set up nodes '''
# set up iterables
infosource = Node(IdentityInterface(fields=['subject_id']),
                                    name='infosource')
infosource.iterables = [('subject_id', sub_list)]

# Create DataGrabber node
dg = Node(DataGrabber(infields=[],
                      outfields=['dwi_file',
                                 'bval_file',
                                 'bvec_file',
                                 'atlas',
                                 'mask']),
          name='datagrabber')

# Location of the dataset folder
dg.inputs.base_directory = project_dir

# Necessary default parameters
dg.inputs.template = '*'
dg.inputs.sort_filelist = True

dg.inputs.template_args = {'dwi_file': [['subject_id']],
                           'bval_file' : [['subject_id']],
                           'bvec_file': [['subject_id']],
                           'atlas': [['subject_id','subject_id']],
                           'mask': [['subject_id']]}#,'subject_id']]}
dg.inputs.field_template = {'dwi_file': 'data/%s/T1w/Diffusion_7T/data.nii.gz',
                            'bval_file': 'data/%s/T1w/Diffusion_7T/bvals',
                            'bvec_file': 'data/%s/T1w/Diffusion_7T/bvecs',
                            'atlas': 'data/%s/T1w/%s/mri/aseg.hires.nii.gz',
                            'mask': 'data/%s/T1w/Diffusion_7T/'\
                                    'nodif_brain_mask.nii.gz'}
#                            'mask': 'derivatives/rs_corr_fwhm-3.2/%s/'\
#                                    'subcortical_mask/_subject_id_%s/'\
#                                    'subcortical_mask.nii.gz'}

def make_mask(atlas,mask):
    import nibabel as nib
    import numpy as np
    import os
    from nilearn.image import resample_to_img

    subcort_labels = [16, 28, 60, 10, 49] # brainstem, L/R VentralDC, L/R thalamus

    atlas_img = nib.load(atlas)
    atlas_data = atlas_img.get_data()
    affine = atlas_img.affine

    mask_data = np.isin(atlas_data, subcort_labels).astype('uint8')
    mask_img = nib.Nifti1Image(mask_data, affine=affine)
    mask_filename = os.path.abspath('subcortical_mask.nii.gz')

    nodif_mask_img = nib.load(mask)
    mask_resample = resample_to_img(mask_img, nodif_mask_img)
    nib.save(mask_resample, mask_filename)

    return mask_filename
masker = Node(Function(input_names=['atlas', 'mask'],
                       output_names=['mask_filename'],
                       function=make_mask), name='masker')


# DTI (produces FA map used for tracking)
def dti_recon(dwi_file, bval_file, bvec_file, mask, b0_threshold):
    import os

    # define outputs
    out_tensor = os.path.abspath('tensors.nii.gz')
    out_fa = os.path.abspath('fa.nii.gz')
    out_ga = os.path.abspath('ga.nii.gz')
    out_rgb = os.path.abspath('rgb.nii.gz')
    out_md = os.path.abspath('md.nii.gz')
    out_ad = os.path.abspath('ad.nii.gz')
    out_rd = os.path.abspath('rd.nii.gz')
    out_mode = os.path.abspath('mode.nii.gz')
    out_evec = os.path.abspath('evecs.nii.gz')
    out_eval = os.path.abspath('evals.nii.gz')

    # run DTI
    from dipy.workflows.reconst import ReconstDtiFlow
    dti = ReconstDtiFlow()
    dti.run(dwi_file, bval_file, bvec_file, mask,
        b0_threshold=b0_threshold, bvecs_tol=0.01, save_metrics=[],
        out_dir='', out_tensor=out_tensor, out_fa=out_fa,
        out_ga=out_fa, out_rgb=out_rgb, out_md=out_md,
        out_ad=out_ad, out_rd=out_rd, out_mode=out_mode,
        out_evec=out_evec, out_eval=out_eval)

    return (out_tensor, out_fa, out_ga, out_rgb, out_md,
            out_ad, out_rd, out_mode, out_evec, out_eval)

recon_dti = Node(Function(input_names=['dwi_file', 'bval_file',
                                       'bvec_file', 'mask', 'b0_threshold'],
                          output_names=['out_tensor', 'out_fa',
                                        'out_ga', 'out_rgb',
                                        'out_md', 'out_ad', 'out_rd',
                                        'out_mode', 'out_evec', 'out_eval'],
                          function=dti_recon),
                          name='recon_dti')
recon_dti.inputs.b0_threshold = b0_thresh

# choose ODF model
def dmri_recon(dwi_file, bval_file, bvec_file, mask, b0_threshold,
               recon='csd', num_threads=2):
    # define input parameters
    csd_resp_fa_thresh = 0.5 # dropped from 0.7

    # define output filenames
    import os
    out_pam = os.path.abspath('peaks.pam5')
    out_shm = os.path.abspath('shm.nii.gz')
    out_peaks_dir = os.path.abspath('peaks_dirs.nii.gz')
    out_peaks_values = os.path.abspath('peaks_values.nii.gz')
    out_peaks_indices = os.path.abspath('peaks_indices.nii.gz')
    out_gfa = os.path.abspath('gfa.nii.gz')

    # import and run/fit model
    if recon == 'csd':
        from dipy.workflows.reconst import ReconstCSDFlow
        csd = ReconstCSDFlow()
        csd.run(dwi_file, bval_file, bvec_file, mask,
            b0_threshold=b0_threshold, bvecs_tol=0.01,
            roi_center=None, roi_radius=10,
            fa_thr=csd_resp_fa_thresh, frf=None,
            extract_pam_values=True,
            odf_to_sh_order=8,
            out_dir='',
            out_pam=out_pam, out_shm=out_shm,
            out_peaks_dir=out_peaks_dir,
            out_peaks_values=out_peaks_values,
            out_peaks_indices=out_peaks_indices, out_gfa=out_gfa)
    elif recon == 'csa':
        from dipy.workflows.reconst import ReconstCSAFlow
        csa = ReconstCSAFlow()
        csa.run(dwi_file, bval_file, bvec_file, mask,
            odf_to_sh_order=8,
            b0_threshold=b0_threshold, bvecs_tol=0.01,
            extract_pam_values=True,
            out_dir='',
            out_pam=out_pam, out_shm=out_shm,
            out_peaks_dir=out_peaks_dir,
            out_peaks_values=out_peaks_values,
            out_peaks_indices=out_peaks_indices, out_gfa=out_gfa)
    return (out_pam, out_shm, out_peaks_dir,
            out_peaks_values, out_peaks_indices, out_gfa)
recon_dmri = Node(Function(input_names=['dwi_file', 'bval_file',
                                        'bvec_file', 'mask',
                                        'recon', 'num_threads',
                                        'b0_threshold'],
                          output_names=['out_pam', 'out_shm',
                                        'out_peaks_dir',
                                        'out_peaks_values',
                                        'out_peaks_indices',
                                        'out_gfa'],
                          function=dmri_recon), name='recon_dmri')
recon_dmri.inputs.recon = recon
recon_dmri.inputs.num_threads = num_threads
recon_dmri.inputs.b0_threshold = b0_thresh
# run tractography based on ODFs
def tracking(pam_files, stopping_files, seeding_files):
    # define inputs
    tracking_method = 'eudx' #"deterministic"
    stopping_thr = 0.1 # dropped from 0.2

    # define outputs
    import os
    out_tractogram = os.path.abspath('tractogram.trk')

    # import and run tractography algorithm
    from dipy.workflows.tracking import LocalFiberTrackingPAMFlow
    track_local = LocalFiberTrackingPAMFlow()
    track_local.run(pam_files, stopping_files, seeding_files,
              stopping_thr=stopping_thr,
              seed_density=1,
              tracking_method=tracking_method,
              pmf_threshold=0.1,
              max_angle=30.,
              out_dir='',
              out_tractogram=out_tractogram)
    return out_tractogram
tracker = Node(Function(input_names=['pam_files', 'stopping_files',
                                        'seeding_files'],
                          output_names=['out_tractogram'],
                          function=tracking), name='tracker')

# define output node
ds = Node(DataSink(parameterization=False), name='sinker')
ds.inputs.base_directory = out_dir
ds.plugin_args = {'overwrite': True}

''' create nipype workflow '''
wf = Workflow(name='dipy_tracker')
#wf.config['execution']['crashfile_format'] = 'txt'
wf.base_dir = work_dir

# connect inputs
wf.connect(infosource, 'subject_id', dg, 'subject_id')

wf.connect(dg, 'atlas', masker, 'atlas')
wf.connect(dg, 'mask', masker, 'mask')

wf.connect(dg, 'dwi_file', recon_dti, 'dwi_file')
wf.connect(dg, 'bval_file', recon_dti, 'bval_file')
wf.connect(dg, 'bvec_file', recon_dti, 'bvec_file')
wf.connect(masker, 'mask_filename', recon_dti, 'mask')


wf.connect(dg, 'dwi_file', recon_dmri, 'dwi_file')
wf.connect(dg, 'bval_file', recon_dmri, 'bval_file')
wf.connect(dg, 'bvec_file', recon_dmri, 'bvec_file')
wf.connect(masker, 'mask_filename', recon_dmri, 'mask')

wf.connect(recon_dmri, 'out_pam', tracker, 'pam_files')
wf.connect(recon_dti, 'out_fa', tracker, 'stopping_files')
wf.connect(masker, 'mask_filename', tracker, 'seeding_files')

# send outputs to data sink
wf.connect(infosource, 'subject_id', ds, 'container')
wf.connect([(recon_dti, ds, [('out_tensor', 'dti.@tensor'),
                            ('out_fa', 'dti.@fa'),
                            ('out_ga', 'dti.@ga'),
                            ('out_rgb', 'dti.@rgb'),
                            ('out_md', 'dti.@md'),
                            ('out_ad', 'dti.@ad'),
                            ('out_rd', 'dti.@rd'),
                            ('out_evec', 'dti.@evec'),
                            ('out_eval', 'dti.@eval')])])
wf.connect([(recon_dmri, ds, [('out_pam', 'recon.@pam'),
                            ('out_shm', 'recon.@shm'),
                            ('out_gfa', 'recon.@gfa')])])
wf.connect(masker, 'mask_filename', ds, 'brainstem_mask')
wf.connect(tracker, 'out_tractogram', ds, 'tracking')


wf.run(plugin='MultiProc', plugin_args={'n_procs' : num_threads})
