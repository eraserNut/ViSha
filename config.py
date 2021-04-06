# coding: utf-8
'''
Dataset root form: ($path, $type, $name)
path: dataset path
type: "image" or "video"
name: just for easy mark
'''
### Saliency datasets
# DUT_OMRON_training_root = ('/home/ext/chenzhihao/Datasets/saliency_dataset/DUT-OMRON', 'image', 'DUT-OMRON')
# MSRA10K_training_root = ('/home/ext/chenzhihao/Datasets/saliency_dataset/MSRA10K', 'image', 'MSRA10K')
# DAVIS_training_root = ('/home/ext/chenzhihao/Datasets/saliency_dataset/DAVIS_train', 'video', 'DAVIS_train')
# DAVIS_validation_root = ('/home/ext/chenzhihao/Datasets/saliency_dataset/DAVIS_val', 'video', 'DAVIS_val')

# Shadow datasets
# SBU_training_root = ('/home/ext/chenzhihao/Datasets/SBU-shadow/SBUTrain4KRecoveredSmall', 'image', 'SBU_train')
# SBU_testing_root = ('/home/ext/chenzhihao/Datasets/SBU-shadow/SBU-Test', 'image', 'SBU_test')
ViSha_training_root = ('/home/ext/chenzhihao/Datasets/ViSha/train', 'video', 'ViSD_train')
ViSha_validation_root = ('/home/ext/chenzhihao/Datasets/ViSha/test', 'video', 'ViSD_test')


'''
Pretrained single model path
'''
# PDBM_single_path = '/home/ext/chenzhihao/code/video_shadow/models_saliency/PDBM_single_256/50000.pth'
# DeepLabV3_path = '/home/ext/chenzhihao/code/video_shadow/models/deeplabv3/20.pth'
# FPN_path = '/home/ext/chenzhihao/code/video_shadow/models/FPN/20.pth'