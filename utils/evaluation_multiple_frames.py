# this code is modified from https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/codes/test.py

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import utils.util as util
import data.util as data_util
import models.modules.STVSR as STVSR
import csv
import time
import skimage
import skimage.metrics as sm
import itertools
import re
import options.options as option

def check_if_folder_exist(folder_path=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if not os.path.isdir(folder_path):
            print('Folder: ' + folder_path + ' exists and is not a folder!')
            exit()

def test(test_args):
    code_name = test_args['code_name']
    model_path = test_args['model_path']
    data_mode = test_args['data_mode']
    dataset_folder = test_args['dataset_folder']
    result_folder = test_args['result_folder']
    os.environ['CUDA_VISIBLE_DEVICES'] = test_args['cuda']
    use_time = test_args['use_time']

    scale = 4
    N_ot = test_args['N_ot']
    N_in = 1+ N_ot // 2
    header_written = False

    model = STVSR.TMNet(64, N_ot, 8, 5, 40)

    #### dataset    
    if data_mode == 'adobe' or data_mode == 'vid4' or data_mode == 'vimeo_fast' or data_mode == 'vimeo_medium' or data_mode == 'vimeo_slow':
        test_dataset_folder = dataset_folder
    if data_mode == 'SPMC':
        test_dataset_folder = '/data/xiang/SR/spmc/*'
    if data_mode == 'Custom':
        test_dataset_folder = '../test_example/*' # TODO: put your own data path here

    folder = os.path.join(result_folder, data_mode)
    save_folder = os.path.join(folder, code_name)
    save_visualization_folder = os.path.join(folder, code_name)
    save_csv = os.path.join(folder, code_name + '.csv')

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'replicate'
    save_imgs = True #True#
    if 'Custom' in data_mode: save_imgs = True
    ############################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')

    util.mkdirs(save_folder)
    
    util.setup_logger(logger_name=code_name + '_with_' + data_mode, root=folder, phase=code_name, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger(name=code_name + '_with_' + data_mode)
    model_params = util.get_model_total_params(model)

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))

    def single_forward(model, imgs_in, use_time=None, N_ot=None):
        with torch.no_grad():
            # imgs_in.size(): [1,n,3,h,w]
            b,n,c,h,w = imgs_in.size()
            h_n = int(4*np.ceil(h/4))
            w_n = int(4*np.ceil(w/4))
            imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
            imgs_temp[:,:,:,0:h,0:w] = imgs_in
            if use_time == True:
                number_side = int((N_ot - 2 - 1) / 2)
                time_number = [(0.5 / (number_side + 1)) * i for i in range(1, number_side + 1)] + [0.5] + [0.5 + (0.5 / (number_side + 1)) * i for i in range(1, number_side + 1)]
                time_Tensors = torch.Tensor(time_number).unsqueeze(0)
                time_Tensors = time_Tensors.to(device)
            else:
                time_Tensors = None
            model_output = model(imgs_temp, time_Tensors)
            model_output = model_output[:, :, :, 0:scale*h, 0:scale*w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
            return output

    sub_folder_l = sorted(glob.glob(dataset_folder))

    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_psnr_y_l = []
    # avg_ssim_l = []
    # avg_ssim_y_l = []
    sub_folder_name_l = []
    psnr_y_by_idx = []
    # ssim_y_by_idx = []
    for sub_folder in sub_folder_l:
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)
        save_visualization_sub_folder = osp.join(save_visualization_folder, sub_folder_name)

        if data_mode == 'SPMC':
            sub_folder = sub_folder + '/LR/'
        img_LR_l = sorted(glob.glob(sub_folder + '/*'))

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = util.read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []
        if data_mode == 'SPMC':
            sub_folder_GT = osp.join(sub_folder.replace('/LR/', '/truth/'))
        else:
            sub_folder_GT = osp.join(sub_folder.replace('/LR/', '/HR/'))

        if 'Custom' not in data_mode:
            img_GT_lists = glob.glob(osp.join(sub_folder_GT,'*'))
            img_GT_lists.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
            for img_GT_path in img_GT_lists:
                img_GT_l.append(util.read_image(img_GT_path))
        cal_n = 0
        avg_psnr, avg_psnr_sum = 0,0
        # avg_ssim, avg_ssim_sum = 0,0
        avg_psnr_y, avg_psnr_sum_y = 0,0
        # avg_ssim_y, avg_ssim_sum_y = 0,0
        
        if len(img_LR_l) == len(img_GT_l):
            skip = True
        else:
            skip = False
        
        if 'Custom' in data_mode:
            select_idx_list = util.test_index_generation(False, N_ot, len(img_LR_l))
        else:
            select_idx_list = util.test_index_generation_multiple_frames(skip, N_ot, len(img_LR_l), use_time)
        # process each image
        psnr_y_dataset = []
        # ssim_y_dataset = []
        for select_idxs in select_idx_list:
            # get input images
            select_idx = select_idxs[0]
            gt_idx = select_idxs[1]
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            output = single_forward(model, imgs_in, use_time=use_time, N_ot=N_ot)

            outputs = output.data.float().cpu().squeeze(0)            

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                outputs = outputs + output

                outputs = outputs / 4

            # save imgs
            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)
                output_f = outputs[idx,:,:,:].squeeze(0)

                output = util.tensor2img(output_f)
                if save_imgs:
                    cv2.imwrite(osp.join(save_sub_folder, str(name_idx) + '.png'), output)

                if 'Custom' not in data_mode:
                    #### calculate PSNR
                    output = output / 255.

                    GT = np.copy(img_GT_l[name_idx])

                    if crop_border == 0:
                        cropped_output = output
                        cropped_GT = GT
                    else:
                        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border, :]
                    crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
                    # crt_ssim = sm.structural_similarity(im1=cropped_output * 255, im2=cropped_GT * 255, data_range=255, multichannel=True)
                    cropped_GT_y = data_util.bgr2ycbcr(cropped_GT, only_y=True)
                    cropped_output_y = data_util.bgr2ycbcr(cropped_output, only_y=True)
                    crt_psnr_y = util.calculate_psnr(cropped_output_y * 255, cropped_GT_y * 255)
                    psnr_y_dataset.append(crt_psnr_y)
                    # crt_ssim_y = sm.structural_similarity(im1=cropped_output_y * 255, im2=cropped_GT_y * 255, data_range=255, multichannel=False)
                    # ssim_y_dataset.append(crt_ssim_y)
                    avg_psnr_sum += crt_psnr
                    avg_psnr_sum_y += crt_psnr_y
                    # avg_ssim_sum += crt_ssim
                    # avg_ssim_sum_y += crt_ssim_y
                    cal_n += 1
                    logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB  PSNR-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_psnr_y))

                    with open(save_csv, "a+", newline="") as wf:
                        writer = csv.DictWriter(wf, fieldnames=['name', 'psnr-y'])
                        if header_written == True:
                            pass
                        else:
                            writer.writeheader()
                            header_written = True
                        writer.writerow({'name': osp.join(sub_folder_name, '{:08d}.png'.format(name_idx+1)), 'psnr-y': crt_psnr_y})

        if 'Custom' not in data_mode:
            avg_psnr = avg_psnr_sum / cal_n
            avg_psnr_y = avg_psnr_sum_y / cal_n
            # avg_ssim = avg_ssim_sum / cal_n
            # avg_ssim_y = avg_ssim_sum_y / cal_n
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_psnr, avg_psnr_y, cal_n))
            # logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} frames; - Average SSIM: {:.6f} dB SSIM-Y: {:.6f} dB for {} frames; '.format(sub_folder_name, avg_psnr, avg_psnr_y, cal_n, avg_ssim, avg_ssim_y, cal_n))
            avg_psnr_l.append(avg_psnr)
            avg_psnr_y_l.append(avg_psnr_y)
            # avg_ssim_l.append(avg_ssim)
            # avg_ssim_y_l.append(avg_ssim_y)

    if 'Custom' not in data_mode:
        logger.info('################ Tidy Outputs ################')
        for name, psnr, psnr_y in zip(sub_folder_name_l, avg_psnr_l, avg_psnr_y_l):
            logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB'.format(name, psnr, psnr_y))
        # for name, psnr, psnr_y, ssim, ssim_y in zip(sub_folder_name_l, avg_psnr_l, avg_psnr_y_l, avg_ssim_l, avg_ssim_y_l):
            # logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB - Average SSIM: {:.6f} dB SSIM-Y: {:.6f} dB. '.format(name, psnr, psnr_y, ssim, ssim_y))
        logger.info('################ Final Results ################')
        logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
        logger.info('Padding mode: {}'.format(padding))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info('Flip Test: {}'.format(flip_test))
        logger.info('Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} clips.'.format(sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_psnr_y_l) / len(avg_psnr_y_l), len(sub_folder_l)))
        # logger.info('Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB for {} clips. Total Average SSIM: {:.6f} dB SSIM-Y: {:.6f} dB for {} clips.'.format(sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_psnr_y_l) / len(avg_psnr_y_l), len(sub_folder_l), sum(avg_ssim_l) / len(avg_ssim_l), sum(avg_ssim_y_l) / len(avg_ssim_y_l), len(sub_folder_l)))

if __name__ == '__main__':
    test()