import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model.decoder import dec_builder, Integrator

from model.content_encoder import content_enc_builder
from model.references_encoder import comp_enc_builder
from model.Component_Attention_Module import ComponentAttentiomModule, Get_style_components
from model.memory import Memory

def position(H, W, is_cuda=True):  
    if is_cuda:   
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride] 

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5) 

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

class GCFM(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(GCFM, self).__init__()
        self.in_planes = in_planes 
        self.out_planes = out_planes  
        self.head = head    
        self.kernel_att = kernel_att  
        self.kernel_conv = kernel_conv   
        self.stride = stride  
        self.dilation = dilation 

        self.head_dim = self.out_planes // self.head   

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)    
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2   
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att) 
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride) 
        self.softmax = torch.nn.Softmax(dim=1)  
        self.fc = nn.Conv2d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)

        self.reset_parameters()


    def reset_parameters(self):
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i//self.kernel_conv, i%self.kernel_conv] = 1.

        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)  
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)


    def forward(self, x, y, z):

        q, k, v = self.conv1(x), self.conv2(y), self.conv3(z)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h//self.stride, w//self.stride

        pe = self.conv_p(position(h, w, x.is_cuda))   

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w) 

        if self.stride > 1:   
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) 
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) 

        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)  
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h*w), k.view(b, self.head, self.head_dim, h*w), v.view(b, self.head, self.head_dim, h*w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.dep_conv(f_conv)

        return 0.5 * out_att + 0.5 * out_conv

class Generator(nn.Module):
    """
    Generator
    """
    def __init__(self, C_in, C, C_out, cfg, comp_enc, dec, content_enc, integrator_args):
        super().__init__()
        self.num_heads = cfg.num_heads 
        self.kshot = cfg.kshot 

        self.component_encoder = comp_enc_builder(C_in, C, **comp_enc)  
        self.mem_shape = self.component_encoder.out_shape 
        assert self.mem_shape[-1] == self.mem_shape[-2]

        self.Get_style_components = Get_style_components()
        self.Get_style_components_1 = Get_style_components()
        self.Get_style_components_2 = Get_style_components()  
        self.cam = ComponentAttentiomModule()  
        self.sk = GCFM(in_planes=256, out_planes=256, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1)
        self.memory = Memory()
        self.shot = cfg.kshot
        C_content = content_enc['C_out']
        C_reference = comp_enc['C_out']
        self.content_encoder = content_enc_builder(C_in, C, **content_enc) 
        self.decoder = dec_builder(
            C, C_out, **dec
        )   

        self.Integrator = Integrator(C * 8, **integrator_args, C_content=C_content, C_reference=C_reference) 
        self.Integrator_local = Integrator(C * 8, **integrator_args, C_content=C_content, C_reference=0)  

    def reset_memory(self): 
        """
        reset memory
        """
        self.memory.reset_memory()

    def read_decode(self, target_style_ids, trg_sample_index, content_imgs, learned_components, trg_unis, ref_unis,
                    chars_sim_dict, reset_memory=True, reduction='mean'):
        reference_feats = self.memory.read_chars(target_style_ids, trg_sample_index, reduction=reduction)
        reference_feats = torch.stack([x for x in reference_feats])
        content_fea = self.content_encoder(content_imgs)  
        content_feats = content_fea[-1]

        B, C, H, W = content_fea[-1].shape

        q = content_fea[-1] 
        k = content_fea[-2]
        v = content_fea[-2]

        atten = self.sk(content_fea[-2], content_fea[-2], content_fea[-1])

        try:
            style_components = self.Get_style_components(learned_components, reference_feats)
            style_components = self.Get_style_components_1(style_components, reference_feats)
            style_components = self.Get_style_components_2(style_components, reference_feats)
        except Exception as e:
            traceback.print_exc()

        sr_features = self.cam(content_feats, learned_components, style_components) 
        global_style_features = self.Get_style_global(trg_unis, ref_unis, reference_feats, content_feats, chars_sim_dict)
        all_features = self.Integrator(sr_features, content_feats, global_style_features) 
        out = self.decoder(all_features, atten) 
        if reset_memory:
            self.reset_memory()

        return out, style_components  

    def encode_write_comb(self, style_ids, style_sample_index, style_imgs, reset_memory=True):

        if reset_memory:
            self.reset_memory()

        feats = self.component_encoder(style_imgs)
        feat_scs = feats["last"]  
        self.memory.write_comb(style_ids, style_sample_index, feat_scs)

        return feat_scs

    def CosineSimilarity(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


    def Get_style_global(self, trg_unis, ref_unis, reference_feats, chars_sim_dict):

        list_trg_chars = list(trg_unis)
        list_ref_unis = list(ref_unis)   

        B, K, C, H, W = reference_feats.shape  
        global_feature = torch.zeros([B, C, H, W]).cuda()  

        for i in range(0, B):
            distance_0 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][0]]   

            distance_1 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][1]]

            distance_2 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][2]]

            distance_3 = chars_sim_dict[list_trg_chars[i]][list_ref_unis[i][3]]

            weight = torch.tensor([distance_0, distance_1, distance_2, distance_3])
            t = 1
            weight = F.softmax(weight / t, dim=0)

            global_feature[i] = reference_feats[i][0] * weight[0] + reference_feats[i][1] * weight[1] \
                                + reference_feats[i][2] * weight[2] + reference_feats[i][3] * weight[3]
        return global_feature


    def infer(self, in_style_ids, in_imgs, style_sample_index, trg_style_ids,
              content_imgs, learned_components, trg_unis=None, ref_unis=None, chars_sim_dict=None,
              reduction="mean", k_shot_tag=False):
        in_style_ids = in_style_ids.cuda() 
        in_imgs = in_imgs.cuda() 

        infer_size = content_imgs.size()[0]
        learned_components = learned_components[:infer_size]  

        content_imgs = content_imgs.cuda()  

        reference_feats = self.encode_write_comb(in_style_ids, style_sample_index, in_imgs) 

        if not k_shot_tag:
            reference_feats = reference_feats.unsqueeze(1) 
        else:
            KB, C, H, W = reference_feats.size()
            reference_feats = torch.reshape(reference_feats, (KB // self.shot, self.shot, C, H, W))

        content_fea = self.content_encoder(content_imgs)  

        content_feats = content_fea[-1]

        B, C, H, W = content_fea[-1].shape

        q = content_fea[-1] 
        k = content_fea[-2]
        v = content_fea[-2]

        atten = self.sk(content_fea[-2], content_fea[-2], content_fea[-1])

        content_feats = content_fea[-1]

        content_feats = content_feats.cuda()

        style_components = self.Get_style_components(learned_components, reference_feats)
        style_components = self.Get_style_components_1(style_components, reference_feats)
        style_components = self.Get_style_components_2(style_components, reference_feats)

        sr_features = self.cam(content_feats, learned_components, style_components)  

        if k_shot_tag:
            global_style_features = self.Get_style_global(trg_unis, ref_unis, reference_feats, content_feats, chars_sim_dict)
            all_features = self.Integrator(sr_features, content_feats, global_style_features)
        else:
            all_features = self.Integrator(sr_features, content_feats, reference_feats.squeeze())

        out = self.decoder(all_features, atten) 

        return out, style_components, sr_features

