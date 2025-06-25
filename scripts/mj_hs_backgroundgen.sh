#!/bin/bash


cmd="accelerate launch batch_inference_fgrm.py --meta /home/ubuntu/data/data_2D/px_wh_54k/record_blipcap_p5.csv --image_dir /home/ubuntu/data/data_2D/px_wh_54k/images --mask_dir /home/ubuntu/data/data_2D/px_wh_54k/images \
    --output_dir /home/ubuntu/data/data_2D_processed/px_wh_54k/background/ --batch_size 7"
echo ${cmd}
eval ${cmd}