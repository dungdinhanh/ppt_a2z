#!/bin/bash


cmd="accelerate launch batch_inference_fgrm.py --meta /home/ubuntu/data/data_2D/fashionpedia_46k/record_fashionpd_hc.csv --image_dir /home/ubuntu/data/data_2D/fashionpedia_46k/images/ --mask_dir /home/ubuntu/data/data_2D/fashionpedia_46k/images/ \
    --output_dir /home/ubuntu/data/data_2D_processed/fashionpedia_46k/background/ --batch_size 7"
echo ${cmd}
eval ${cmd}