from utils.inference import test
import torch

custom_option = {}

custom_option['save_imgs'] = True
custom_option['scale'] = 4
custom_option['result_folder'] = './experiments/'
custom_option['data_mode'] = 'adobe'
custom_option['test_dataset_folder'] = '/home/ubuntu/Disk/dataset/adobe240fps/test/LR/*'
custom_option['cuda'] = '1'

custom_option['multiple_frames_generation'] = True
custom_option['N_output_for_interpolating_multiple_frames'] = 7
custom_option['N_output_for_interpolating_middle_frame'] = 7

# ===============================================================================================

custom_option['code_name'] = 'inference'
custom_option['model_path'] = './final.pth'
custom_option['opt'] = './configs/final.yml'

test(custom_option)