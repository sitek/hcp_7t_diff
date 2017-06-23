#!/bin/bash
# grab HCP data through Amazon S3
# based on http://www.adliska.com/2016/02/04/how-to-download-human-connectome-project-data-from-amazon-web-services-aws.html
# KRS 2017.06.22

#SBATCH --time=2:00:00
#SBATCH --mem=30G
#SBATCH --qos=gablab

subject=100610
diffdir=data/$subject/T1w/
mkdir -p $diffdir

aws s3 cp s3://hcp-openaccess/HCP_1200/100610/T1w/ \
  $diffdir \
  --recursive \
  --region us-east-1

echo "done"
