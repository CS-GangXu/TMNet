# This code is modified form the Zooming Slow-Mo https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/codes/models/modules/Sakuya_arch.py
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.convlstm import ConvLSTM, ConvLSTMCell
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class TMB(nn.Module):
    def __init__(self):
        super(TMB, self).__init__()
        self.t_process = nn.Sequential(*[
            nn.Conv2d( 1, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        ])
        self.f_process = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ])

    def forward(self, x, t):
        feature = self.f_process(x)
        modulation_vector = self.t_process(t)
        output = feature * modulation_vector
        return output

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8, use_time=False):
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2 
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        if use_time == True:
            self.TMB_A_l1 = TMB()
            self.TMB_B_l1 = TMB()
            self.TMB_A_l2 = TMB()
            self.TMB_B_l2 = TMB()
            self.TMB_A_l3 = TMB()
            self.TMB_B_l3 = TMB()

    def forward(self, fea1, fea2, t=None, t_back=None):
        '''align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        '''
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset)) 
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset)) if t is None else self.lrelu(self.L3_offset_conv2_1(L3_offset)) + self.TMB_A_l3(L3_offset, t)
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        B, C, L2_H, L2_W = fea1[1].size()
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(L3_offset, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset)) if t is None else self.lrelu(self.L2_offset_conv3_1(L2_offset)) + self.TMB_A_l2(L2_offset, t)
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        B, C, L1_H, L1_W = fea1[0].size()
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(L2_offset, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset)) if t is None else self.lrelu(self.L1_offset_conv3_1(L1_offset)) + self.TMB_A_l1(L1_offset, t)
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset)) if t_back is None else self.lrelu(self.L3_offset_conv2_2(L3_offset)) + self.TMB_B_l3(L3_offset, t_back)
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset)) if t_back is None  else self.lrelu(self.L2_offset_conv3_2(L2_offset)) + self.TMB_B_l2(L2_offset, t_back)
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, size=[L2_H, L2_W], mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset)) if t_back is None else self.lrelu(self.L1_offset_conv3_2(L1_offset)) + self.TMB_B_l1(L1_offset, t_back)
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, size=[L1_H, L1_W], mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
        
        y = torch.cat(y, dim=1)
        return y

class Easy_PCD(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # feature size: f1 = f2 = [B, N, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        try:
            L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        except RuntimeError:
            L3_fea = L3_fea.view(B, N, -1, L3_fea.shape[2], L3_fea.shape[3])

        fea1 = [L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(), L3_fea[:, 0, :, :, :].clone()]
        fea2 = [L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :, :].clone(), L3_fea[:, 1, :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea) # [B, N, C, H, W]
        return fusion_fea

class DeformableConvLSTM(ConvLSTM):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        ConvLSTM.__init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, 
              batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        #### extract features (for each frame)
        nf = input_dim

        self.pcd_h = Easy_PCD(nf=nf, groups=groups)
        self.pcd_c = Easy_PCD(nf=nf, groups=groups)        

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input_tensor, hidden_state = None):
        '''        
        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: 
            None. 
            
        Returns
        -------
        last_state_list, layer_output
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3),input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),tensor_size=tensor_size)
        
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                in_tensor = cur_layer_input[:, t, :, :, :] 
                h_temp = self.pcd_h(in_tensor, h)
                c_temp = self.pcd_c(in_tensor, c)
                h, c = self.cell_list[layer_idx](input_tensor=in_tensor,
                                                 cur_state=[h_temp, c_temp])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, tensor_size):
        return super()._init_hidden(batch_size, tensor_size)

class BiDeformableConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        super(BiDeformableConvLSTM, self).__init__()
        self.forward_net = DeformableConvLSTM(input_size=input_size, input_dim=input_dim, hidden_dim=hidden_dim,
                                           kernel_size=kernel_size, num_layers=num_layers, front_RBs=front_RBs,
                                           groups=groups, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        self.conv_1x1 = nn.Conv2d(2*input_dim, input_dim, 1, 1, bias=True)

    def forward(self, x):
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, reversed_idx, ...]
        out_fwd, _ = self.forward_net(x)
        out_rev, _ = self.forward_net(x_rev)
        rev_rev = out_rev[0][:, reversed_idx, ...]
        B, N, C, H, W = out_fwd[0].size()
        result = torch.cat((out_fwd[0], rev_rev), dim=2)
        result = result.view(B*N,-1,H,W)
        result = self.conv_1x1(result)
        return result.view(B, -1, C, H, W) 

class TMNet(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10, opt=None):
        super(TMNet, self).__init__()
        self.opt = opt
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        p_size = 48 # a place holder, not so useful
        patch_size = (p_size, p_size) 
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups, use_time=True)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.ConvBLSTM = BiDeformableConvLSTM(input_size=patch_size, input_dim=nf, hidden_dim=hidden_dim, kernel_size=(3,3), num_layers=1, batch_first=True, front_RBs=front_RBs, groups=groups)
        #### reconstruction
        self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        layersAtBOffset = []
        layersAtBOffset.append(nn.Conv2d(128, 64, 3, 1, 1, bias=True))
        layersAtBOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersAtBOffset.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.layersAtBOffset = nn.Sequential(*layersAtBOffset)
        self.layersAtB = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8)

        layersCtBOffset = []
        layersCtBOffset.append(nn.Conv2d(128, 64, 3, 1, 1, bias=True))
        layersCtBOffset.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersCtBOffset.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        self.layersCtBOffset = nn.Sequential(*layersCtBOffset)
        self.layersCtB = DCN_sep(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8)

        layersFusion = []
        layersFusion.append(nn.Conv2d(192, 192, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(192, 192, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(192, 192, 1, 1, 0, bias=True))
        layersFusion.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        layersFusion.append(nn.Conv2d(192, 64, 1, 1, 0, bias=True))
        self.layersFusion = nn.Sequential(*layersFusion)

    def forward(self, x, t=None):
        try:
            t_B, t_N = t.shape[0], t.shape[1]
            t = t.view(t_B * t_N).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            t_back = 1 - t
            t = ((t / 0.5) - 1).view(t_B, t_N, 1, 1, 1)
            t_back = ((t_back / 0.5) - 1).view(t_B, t_N, 1, 1, 1)
            use_time = True
        except AttributeError:
            use_time = False
            
        B, N, C, H, W = x.size()  # N input video frames
        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        try:
            L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        except RuntimeError:
            L3_fea = L3_fea.view(B, N, -1, L3_fea.shape[2], L3_fea.shape[3])
        #### align using pcd
        to_lstm_fea = []
        '''
        0: + fea1, fusion_fea, fea2
        1: + ...    ...        ...  fusion_fea, fea2
        2: + ...    ...        ...    ...       ...   fusion_fea, fea2
        '''
        for idx in range(N-1):
            fea1 = [
                L1_fea[:, idx, :, :, :].clone(), L2_fea[:, idx, :, :, :].clone(), L3_fea[:, idx, :, :, :].clone()
            ]
            fea2 = [
                L1_fea[:, idx+1, :, :, :].clone(), L2_fea[:, idx+1, :, :, :].clone(), L3_fea[:, idx+1, :, :, :].clone()
            ]
            if idx == 0:
                to_lstm_fea.append(fea1[0])
            if use_time == True:
                for i in range(t_N):
                    aligned_fea = self.pcd_align(
                        fea1, 
                        fea2,
                        t=t[:, i, :, :, :], 
                        t_back=t_back[:, i, :, :, :]
                    )
                    fusion_fea = self.fusion(aligned_fea)
                    to_lstm_fea.append(fusion_fea)
            else:
                aligned_fea = self.pcd_align(fea1, fea2)
                fusion_fea = self.fusion(aligned_fea)
                to_lstm_fea.append(fusion_fea)
            to_lstm_fea.append(fea2[0])
        dnc_feats = torch.stack(to_lstm_fea, dim = 1)

        back_feats = dnc_feats

        B, T, C, H, W = dnc_feats.size()
        dnc_feats = dnc_feats.view(B, T, C, H, W)
        feats_non_linear_comparison = []
        for i in range(T):
            if i == 0:
                idx = [0, 0, 1]
            else:
                if i == T - 1:
                    idx = [T - 2, T - 1, T - 1]
                else:
                    idx = [i - 1, i, i + 1]
            fea0 = dnc_feats[:, idx[0], :, :, :].contiguous()
            fea1 = dnc_feats[:, idx[1], :, :, :].contiguous()
            fea2 = dnc_feats[:, idx[2], :, :, :].contiguous()
            AtBOffset = self.layersAtBOffset(torch.cat([fea0, fea1], dim=1))
            fea0_aligned = self.lrelu(self.layersAtB(fea0, AtBOffset))

            CtBOffset = self.layersCtBOffset(torch.cat([fea2, fea1], dim=1))
            fea2_aligned = self.lrelu(self.layersCtB(fea2, CtBOffset))

            feats_non_linear_comparison.append(self.layersFusion(torch.cat([fea0_aligned, fea1, fea2_aligned], dim=1)))
        feats_after_comparison = torch.stack(feats_non_linear_comparison, dim = 1)
        lstm_feats = dnc_feats + feats_after_comparison.view(B, T, C, H, W)

        feats = self.ConvBLSTM(lstm_feats) 
        B, T, C, H, W = feats.size()

        feats = feats.view(B * T, C, H, W)
        out = self.recon_trunk(feats)
        out = out + back_feats.view(B * T, C, H, W)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, K, G = out.size()
        outs = out.view(B, T, -1, K, G)
        return outs