# fire detection project
## This is a fire detection project which focus on solving problems that is difficult to detect for human beings. so i decide to create a model to solve it.
## this project is based on ai-studio and the link is https://aistudio.baidu.com/aistudio/projectdetail/2239594

! git clone https://gitee.com/paddlepaddle/PaddleDetection.git
!mv PaddleDetection/ work/
!pip install -r work/PaddleDetection/requirements.txt


import matplotlib.pyplot as plt 
import matplotlib
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
import PIL.Image as Image
import cv2 
import os

%mv PaddleDetection work

%cd work/PaddleDetection


!unzip -oq /home/aistudio/data/data70782/火焰已标注数据集500张.zip
!gzip -dfq /home/aistudio/data/data43124/dataset.tar.gz

!python tools/train.py -c configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml --eval --use_vdl=True --vdl_log_dir="./output"
%cd /home/aistudio
%cp work/PaddleDetection/output/yolov3_mobilenet_v3_large_ssld_270e_voc/model_final.pdparams best_model
%cp work/PaddleDetection/output/yolov3_mobilenet_v3_large_ssld_270e_voc/model_final.pdopt best_model

!python work/PaddleDetection/tools/infer.py -c work/PaddleDetection/configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml -o weights=work/PaddleDetection/output/yolov3_mobilenet_v3_large_ssld_270e_voc/model_final.pdparams --infer_img=dataset2/JPEGImages/113.jpg

!python work/PaddleDetection/tools/export_model.py -c work/PaddleDetection/configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml \
        --output_dir=./inference_model \
        -o weights=work/PaddleDetection/output/yolov3_mobilenet_v3_large_ssld_270e_voc/model_final.pdparams
