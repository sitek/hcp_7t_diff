
import os
import numpy as np
import nipype.pipeline.engine as pe


sub = '100610'
datadir = os.path.abspath('/om/user/ksitek/hcp_7t/data/%s/T1w/'%sub)

bvecs = os.path.join(datadir, 'Diffusion_7T', 'bvecs')
bvals = os.path.join(datadir, 'Diffusion_7T', 'bvals')
dwi = os.path.join(datadir, 'Diffusion_7T', 'data.nii.gz')
grad_dev = os.path.join(datadir, 'Diffusion_7T', 'grad_dev.nii.gz')

brain_mask = os.path.join(datadir, 'Diffusion_7T', 'nodif_brain_mask.nii.gz')

seed = os.path.join(datadir, 'Diffusion_7T', 'IC_L_sphere_bin.nii.gz')

from nipype.interfaces import fsl
bedp = pe.Node(fsl.BEDPOSTX5(), name='bedpost')
bedp.inputs.bvecs = bvecs 
bedp.inputs.bvals = bvals
bedp.inputs.dwi = dwi
bedp.inputs.mask = brain_mask
bedp.inputs.grad_dev = grad_dev
bedp.inputs.n_fibres = 1
bedp.inputs.use_gpu = True
bedp.inputs.out_dir = os.path.join(datadir, 'Diffusion_7T.bedpostX')

from nipype.interfaces import fsl
pbx2 = pe.Node(fsl.ProbTrackX2(), name='probtrackx')
pbx2.inputs.seed = seed
pbx2.inputs.mask = brain_mask
pbx2.inputs.out_dir = os.path.join(datadir, 'Diffusion_7T.probtrackx2')

wf = pe.Workflow(name='fdt')
wf.connect([(bedp, pbx2,[('merged_fsamples','fsamples'),
                         ('merged_phsamples','phsamples'),
                         ('merged_thsamples','thsamples')]
                         )])

wf.run(plugin='SLURM', 
       plugin_args='--gres=gpu:1 --time=18:00:00 --qos=gablab --mem=40G -c 4')

