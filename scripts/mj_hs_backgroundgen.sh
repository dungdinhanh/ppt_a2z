#!/bin/bash


cmd="accelerate launch batch_inference_fgrm.py --meta /home/ubuntu/data/data_2D/mj_hmsp_67k/record_mj_hmsp_hc_filtered.csv --image_dir /home/ubuntu/data/data_2D/mj_hmsp_67k/images --mask_dir /home/ubuntu/data/data_2D/mj_hmsp_67k/images \
    --output_dir /home/ubuntu/data/data_2D_processed/mj_hmsp_67k/background/ --batch_size 15"
echo ${cmd}
eval ${cmd}