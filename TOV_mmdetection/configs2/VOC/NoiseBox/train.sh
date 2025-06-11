##E-noiseBox
#N=0.4 && CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
# tools/dist_train.sh configs2/VOC/NoiseBox/ENoiseBox_r50_fpn_1x_voc.py 2 \
# --work-dir=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc/ \
# --resume-from=outputs/VOC_noise_${N}/ENoiseBox_r50_fpn_1x_voc/latest.pth

