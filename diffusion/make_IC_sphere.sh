#!/bin/bash

dwi_path=/om/user/ksitek/hcp_7t/data/100610/T1w/Diffusion_7T/

fslmaths ${dwi_path}mean_dwi.nii.gz -mul 0 -add 1 -roi 90 1 85 1 62 1 0 1 ${dwi_path}IC_L_point -odt float
fslmaths ${dwi_path}IC_L_point.nii.gz -kernel sphere 2 -fmean ${dwi_path}IC_L_sphere -odt float
fslmaths ${dwi_path}IC_L_sphere.nii.gz -bin ${dwi_path}IC_L_sphere_bin.nii.gz
