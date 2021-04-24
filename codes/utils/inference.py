import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.modules.Basic as Basic
import time
import skimage
import skimage.metrics as sm
import itertools
import csv

import re
import options.options as option

def check_if_folder_exist(folder_path='/home/ubuntu/'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if not os.path.isdir(folder_path):
            print('Folder: ' + folder_path + ' exists and is not a folder!')
            exit()

def test(test_args):
    opt = option.parse(test_args['opt'], is_train=True)
    code_name = test_args['code_name']
    scale = test_args['scale']
    
    model_path = test_args['model_path']
    test_dataset_folder = test_args['test_dataset_folder']
    save_imgs = test_args['save_imgs']
    multiple_frames_generation = test_args['multiple_frames_generation']
    result_folder = test_args['result_folder']
    data_mode = test_args['data_mode']
    error_bar_folder = os.path.join(result_folder, data_mode, code_name + '_error_bar')

    if multiple_frames_generation:
        N_ot = test_args['N_output_for_interpolating_multiple_frames']
    else:
        N_ot = test_args['N_output_for_interpolating_middle_frame']
    check_if_folder_exist(error_bar_folder)
    
    cuda = test_args['cuda']


    save_folder = os.path.join(result_folder, data_mode, code_name)
    check_if_folder_exist(save_folder)
    header_written = False
    error_header_written = False
    save_csv = os.path.join(result_folder, data_mode, code_name + '.csv')

    error_bar_csv = os.path.join(result_folder, data_mode, code_name + '_error_bar', code_name + '.csv')

    # plt.figure(figsize=(4,3))

    N_in = 1+ N_ot // 2
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    model = Basic.Network(64, N_ot, 8, 5, 40, opt=opt)

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'replicate'
    if 'Custom' in data_mode: save_imgs = True
    ############################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')
    model_params = util.get_model_total_params(model)

    def single_forward(model, imgs_in, multiple_frames_generation=None, N_ot=None):
        with torch.no_grad():
            # imgs_in.size(): [1,n,3,h,w]
            b,n,c,h,w = imgs_in.size()
            h_n = int(4*np.ceil(h/4))
            w_n = int(4*np.ceil(w/4))
            imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
            imgs_temp[:,:,:,0:h,0:w] = imgs_in

            if multiple_frames_generation == True:
                time_list = []
                number_side = int((N_ot - 2 - 1) / 2)
                time_Tensors = torch.Tensor([(0.5 / (number_side + 1)) * i for i in range(1, number_side + 1)] + [0.5] + [0.5 + (0.5 / (number_side + 1)) * i for i in range(1, number_side + 1)]).unsqueeze(0).cuda()
                # time_Tensors = torch.Tensor([0.1667, 0.3333, 0.5, 0.6667, 0.8333]).unsqueeze(0).cuda()
                # time_Tensors = torch.Tensor([0.25, 0.5, 0.75]).unsqueeze(0).cuda()
                # time_Tensors = torch.Tensor([0.5]).unsqueeze(0).cuda()
            else:
                time_Tensors = None
            model_output = model(imgs_temp, time_Tensors)
            model_output = model_output[:, :, :, 0:scale*h, 0:scale*w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))

    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_psnr_y_l = []
    avg_ssim_l = []
    avg_ssim_y_l = []
    sub_folder_name_l = []
    total_time = []
    psnr_y_by_idx = []
    ssim_y_by_idx = []
    # total_time = []
    for sub_folder in sub_folder_l:
        print('Processing ' + sub_folder)
        gt_tested_list = []
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

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
        avg_ssim, avg_ssim_sum = 0,0
        avg_psnr_y, avg_psnr_sum_y = 0,0
        avg_ssim_y, avg_ssim_sum_y = 0,0
        
        if len(img_LR_l) == len(img_GT_l):
            skip = True
        else:
            skip = False
        
        if 'Custom' in data_mode:
            select_idx_list = util.test_index_generation(False, N_ot, len(img_LR_l))
        else:
            if multiple_frames_generation == True:
                select_idx_list = util.test_multiple_index_generation(skip, N_ot, len(img_LR_l), use_topAndEnd=multiple_frames_generation)
            else:
                select_idx_list = util.test_index_generation(skip, N_ot, len(img_LR_l), use_topAndEnd=multiple_frames_generation)
        # process each image
        psnr_y_dataset = []
        ssim_y_dataset = []
        for select_idxs in select_idx_list:
            
            print(select_idxs)
            # get input images
            select_idx = select_idxs[0]
            gt_idx = select_idxs[1]
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            time_start = time.time()
            output = single_forward(model, imgs_in, multiple_frames_generation=multiple_frames_generation, N_ot=N_ot)
            time_end = time.time()
            running_time_single = time_end - time_start
            total_time.append(running_time_single)

            outputs = output.data.float().cpu().squeeze(0)            

            # save imgs
            for idx, name_idx in enumerate(gt_idx):
                if multiple_frames_generation == False:
                    if name_idx in gt_tested_list:
                        continue
                gt_tested_list.append(name_idx)
                output_f = outputs[idx,:,:,:].squeeze(0)

                output = util.tensor2img(output_f)

                if save_imgs:
                    if multiple_frames_generation == True:
                        path = osp.join(save_sub_folder, name_idx + '.png')
                        # path = osp.join(save_sub_folder, str(select_idx[0]) + '-' + str(idx) + '-' + str(select_idx[1]) + '.png')
                        # path = osp.join(save_sub_folder, '{:08d}.png'.format(name_idx)) 
                    else:
                        path = osp.join(save_sub_folder, '{:08d}.png'.format(name_idx+1))
                    cv2.imwrite(path, output)
                
                if multiple_frames_generation == True:
                    continue

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
                    crt_ssim = sm.structural_similarity(im1=cropped_output * 255, im2=cropped_GT * 255, data_range=255, multichannel=True)

                    cropped_GT_y = data_util.bgr2ycbcr(cropped_GT, only_y=True)
                    cropped_output_y = data_util.bgr2ycbcr(cropped_output, only_y=True)

                    crt_psnr_y = util.calculate_psnr(cropped_output_y * 255, cropped_GT_y * 255)
                    print('PSNR:' + str(name_idx) + ' ' + str(crt_psnr_y))
                    psnr_y_dataset.append(crt_psnr_y)
                    crt_ssim_y = sm.structural_similarity(im1=cropped_output_y * 255, im2=cropped_GT_y * 255, data_range=255, multichannel=False)
                    ssim_y_dataset.append(crt_ssim_y)

                    # logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB  PSNR-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_psnr_y))

                    avg_psnr_sum += crt_psnr
                    avg_psnr_sum_y += crt_psnr_y
                    avg_ssim_sum += crt_ssim
                    avg_ssim_sum_y += crt_ssim_y
                    cal_n += 1

                    with open(save_csv, "a+", newline="") as wf:
                        writer = csv.DictWriter(wf, fieldnames=['name', 'psnr-y'])
                        if header_written == True:
                            pass
                        else:
                            writer.writeheader()
                            header_written = True
                        writer.writerow({'name': osp.join(sub_folder_name, '{:08d}.png'.format(name_idx+1)), 'psnr-y': crt_psnr_y})
        
        if multiple_frames_generation == True:
            continue
        
        ssim_y_dataset = ssim_y_dataset[0:len(ssim_y_dataset)-((len(ssim_y_dataset) - N_ot) % (N_ot - 1))]
        psnr_y_dataset = psnr_y_dataset[0:len(psnr_y_dataset)-((len(psnr_y_dataset) - N_ot) % (N_ot - 1))]

        # plt.cla()
        # plt.title('error bar')
        idxs = []
        psnr_y_means = []
        psnr_y_stds = []
        ssim_y_means = []
        ssim_y_stds =[]
        for idx in range(N_ot):
            idxs.append(idx)
            try:
                temp = psnr_y_by_idx[idx]
            except IndexError:
                psnr_y_by_idx.append([])
            try:
                temp = ssim_y_by_idx[idx]
            except IndexError:
                ssim_y_by_idx.append([])
            psnr_y_by_idx[idx] += psnr_y_dataset[idx: idx + len(psnr_y_dataset) - N_ot + 1:N_ot - 1]
            psnr_y_means.append(np.mean(psnr_y_dataset[idx: idx + len(psnr_y_dataset) - N_ot + 1:N_ot - 1]))
            psnr_y_stds.append(np.std(psnr_y_dataset[idx: idx + len(psnr_y_dataset) - N_ot + 1:N_ot - 1]))
            ssim_y_by_idx[idx] += ssim_y_dataset[idx: idx + len(ssim_y_dataset) - N_ot + 1:N_ot - 1]
            ssim_y_means.append(np.mean(ssim_y_dataset[idx: idx + len(ssim_y_dataset) - N_ot + 1:N_ot - 1]))
            ssim_y_stds.append(np.std(ssim_y_dataset[idx: idx + len(ssim_y_dataset) - N_ot + 1:N_ot - 1]))

        # plt.errorbar(x=idxs, y=means, yerr=stds, fmt='.k')
        # plt.savefig(os.path.join(error_bar_folder, sub_folder.replace("/", "_") + '.png')) 
        
        with open(error_bar_csv, "a+", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=[
                'item', 
                '0-psnr-y', 
                '1-psnr-y', 
                '2-psnr-y', 
                '3-psnr-y', 
                '4-psnr-y', 
                '5-psnr-y', 
                '6-psnr-y', 
                'mean-psnr-y', 
                '0-ssim-y', 
                '1-ssim-y', 
                '2-ssim-y', 
                '3-ssim-y', 
                '4-ssim-y', 
                '5-ssim-y', 
                '6-ssim-y', 
                'mean-ssim-y'
            ])
            if error_header_written == True:
                pass
            else:
                writer.writeheader()
                error_header_written = True
            writer.writerow({
                'item': sub_folder_name,
                '0-psnr-y': psnr_y_means[0],
                '1-psnr-y': psnr_y_means[1],
                '2-psnr-y': psnr_y_means[2],
                '3-psnr-y': psnr_y_means[3],
                '4-psnr-y': psnr_y_means[4],
                '5-psnr-y': psnr_y_means[5],
                '6-psnr-y': psnr_y_means[6],
                'mean-psnr-y': np.mean(psnr_y_means),
                '0-ssim-y': ssim_y_means[0],
                '1-ssim-y': ssim_y_means[1],
                '2-ssim-y': ssim_y_means[2],
                '3-ssim-y': ssim_y_means[3],
                '4-ssim-y': ssim_y_means[4],
                '5-ssim-y': ssim_y_means[5],
                '6-ssim-y': ssim_y_means[6],
                'mean-ssim-y': np.mean(ssim_y_means),
            })


        if 'Custom' not in data_mode:
            avg_psnr = avg_psnr_sum / cal_n
            avg_psnr_y = avg_psnr_sum_y / cal_n
            avg_ssim = avg_ssim_sum / cal_n
            avg_ssim_y = avg_ssim_sum_y / cal_n

            avg_psnr_l.append(avg_psnr)
            avg_psnr_y_l.append(avg_psnr_y)
            avg_ssim_l.append(avg_ssim)
            avg_ssim_y_l.append(avg_ssim_y)
    
    with open(error_bar_csv, "a+", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=[
                'item', 
                '0-psnr-y', 
                '1-psnr-y', 
                '2-psnr-y', 
                '3-psnr-y', 
                '4-psnr-y', 
                '5-psnr-y', 
                '6-psnr-y', 
                'mean-psnr-y', 
                '0-ssim-y', 
                '1-ssim-y', 
                '2-ssim-y', 
                '3-ssim-y', 
                '4-ssim-y', 
                '5-ssim-y', 
                '6-ssim-y', 
                'mean-ssim-y'
            ])
            if error_header_written == True:
                pass
            else:
                writer.writeheader()
                error_header_written = True
            writer.writerow({
                'item': 'total',
                '0-psnr-y': np.mean(psnr_y_by_idx[0]),
                '1-psnr-y': np.mean(psnr_y_by_idx[1]),
                '2-psnr-y': np.mean(psnr_y_by_idx[2]),
                '3-psnr-y': np.mean(psnr_y_by_idx[3]),
                '4-psnr-y': np.mean(psnr_y_by_idx[4]),
                '5-psnr-y': np.mean(psnr_y_by_idx[5]),
                '6-psnr-y': np.mean(psnr_y_by_idx[6]),
                'mean-psnr-y': np.mean(psnr_y_by_idx),
                '0-ssim-y': np.mean(ssim_y_by_idx[0]),
                '1-ssim-y': np.mean(ssim_y_by_idx[1]),
                '2-ssim-y': np.mean(ssim_y_by_idx[2]),
                '3-ssim-y': np.mean(ssim_y_by_idx[3]),
                '4-ssim-y': np.mean(ssim_y_by_idx[4]),
                '5-ssim-y': np.mean(ssim_y_by_idx[5]),
                '6-ssim-y': np.mean(ssim_y_by_idx[6]),
                'mean-ssim-y': np.mean(ssim_y_by_idx),
            })
    
    if multiple_frames_generation == True:
        exit()
    return