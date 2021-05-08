from utils.evaluation_multiple_frames import test

custom_option = {}

# DEFAULT =============================================================
custom_option['cuda'] = '0'
custom_option['code_name'] = 'TMNet'
custom_option['use_time'] = True
custom_option['N_ot'] = 7
custom_option['model_path'] = './checkpoints/tmnet_multiple_frames.pth'
custom_option['result_folder'] = './evaluations'

# DATASET =============================================================
custom_option['data_mode'] = 'adobe'
custom_option['dataset_folder'] = './datasets/adobe240fps/test/LR/*'


# custom_option['data_mode'] = 'vid4'
# custom_option['dataset_folder'] = './datasets/vid4/LR/*'
# TEST ================================================================
test(custom_option)