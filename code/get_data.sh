#!/bin/bash
# grab HCP data through Amazon S3
# based on http://www.adliska.com/2016/02/04/how-to-download-human-connectome-project-data-from-amazon-web-services-aws.html
# KRS 2017.06.22

#SBATCH --time=1:00:00
#SBATCH --mem=100G
#SBATCH -c 10
#SBATCH -p om_bigmem

data_dir=/om2/user/ksitek/hcp_7t/data/

#subject=102816
for subject_dir in $data_dir/13*; do
  subject=$(basename $subject_dir)

  #image_path=MNINonLinear/Results/
  #file_path=rfMRI_REST1_7T_PA/

  image_path=MNINonLinear
  file_path=xfms

  write_out=${data_dir}/${subject}/${image_path}/${file_path}/
  mkdir -p $write_out

  aws s3 cp s3://hcp-openaccess/HCP_1200/${subject}/${image_path}/$file_path \
    $write_out \
    --exclude "*" --include "*standard*.nii.gz" \
    --recursive \
    --region us-east-1

  #echo "copied ${subject} ${image_dir}"
done
