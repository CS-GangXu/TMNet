from utils.evaluation_single_frame import test

test_args = {}

# DEFAULT =============================================================
test_args['cuda'] = '1'
test_args['code_name'] = 'TMNet'
test_args['model_path'] = './checkpoints/tmnet_single_frame.pth'
test_args['result_folder'] = './evaluations'

# DATASET =============================================================
test_args['data_mode'] = 'vid4'
test_args['dataset_folder'] =  './datasets/vid4/LR/*'

# test_args['data_mode'] = 'vimeo_fast'
# test_args['dataset_folder'] =  './datasets/vimeo-90k_septuplet/fast_of_test/LR/*'

# test_args['data_mode'] = 'vimeo_medium'
# test_args['dataset_folder'] =  './datasets/vimeo-90k_septuplet/medium_of_test/LR/*'

# test_args['data_mode'] = 'vimeo_slow'
# test_args['dataset_folder'] =  './datasets/vimeo-90k_septuplet/slow_of_test/LR/*'

# TEST ================================================================
test(test_args)