import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Get_style_components(nn.Module):
    def __init__(self, num_heads=8, num_channels=256):
        super(Get_style_components, self).__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels

        self.linears_key = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias=False)

        self.multihead_concat_fc = nn.Linear(num_channels, num_channels, bias=False)
        self.layer_norm = nn.LayerNorm(num_channels, eps=1e-6)

    def get_kqv_matrix(self, fm, linears):
        # matmul with style featuremaps and content featuremaps
        ret = linears(fm)
        return ret

    def get_reference_key_vaule(self, reference_map_list):
        B, K, C, H, W = reference_map_list.shape
        m = self.num_heads
        d_channel = self.num_channels // self.num_heads
        reference_sequence = rearrange(reference_map_list, 'b k c h w -> b (k h w) c')

        key_reference_matrix = self.get_kqv_matrix(reference_sequence, self.linears_key)  
        key_reference_matrix = torch.reshape(key_reference_matrix, (B, K * H * W, m, d_channel))
        key_reference_matrix = rearrange(key_reference_matrix, 'b khw m d -> b m khw d')

        value_reference_matrix = self.get_kqv_matrix(reference_sequence, self.linears_value) 
        value_reference_matrix = torch.reshape(value_reference_matrix, (B, K * H * W, m, d_channel))
        value_reference_matrix = rearrange(value_reference_matrix, 'b khw m d -> b m khw d')

        return key_reference_matrix, value_reference_matrix

    def get_component_query(self, component_sequence):
        B, N, C = component_sequence.shape
        d_channel = self.num_channels // self.num_heads
        m = self.num_heads
        query_component_matrix = self.get_kqv_matrix(component_sequence, self.linears_query)
        query_component_matrix = torch.reshape(query_component_matrix, (B, N, m, d_channel))
        query_component_matrix = rearrange(query_component_matrix, 'b n m d -> b m d n')

        return query_component_matrix


    def cross_attention(self, query, key, value, mask=None, dropout=None):
        residual = query
        query = self.get_component_query(query)  
        scores = torch.matmul(key, query) 
        scores = rearrange(scores, 'b m khw n -> b m n khw')
        p_attn = F.softmax(scores, dim=-1)

        out = torch.matmul(p_attn, value)  
        out = rearrange(out, 'b m n d -> b n (m d)')
        out = self.multihead_concat_fc(out)
        out = self.layer_norm(out + residual)

        return out


    def forward(self, component_sequence, reference_map_list):
        key_reference_matrix, value_reference_matrix = self.get_reference_key_vaule(reference_map_list)  # (b m) khw d_channel

        style_components = self.cross_attention(component_sequence, key_reference_matrix,
                                                value_reference_matrix)

        return style_components


class ComponentAttentiomModule(nn.Module):
    def __init__(self, num_heads=8, num_channels=256):
        super(ComponentAttentiomModule, self).__init__()
        self.linears_key = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias=False)
        self.multihead_concat_fc = nn.Linear(num_channels, num_channels, bias=False)
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(num_channels, eps=1e-6)

    def get_kqv_matrix(self, fm, linears):
        # matmul with style featuremaps and content featuremaps
        ret = linears(fm)
        return ret

    def get_content_query(self, content_feature_map):
        B, C, H, W = content_feature_map.shape
        m = self.num_heads
        d_channel = C // m
        query_component_matrix = rearrange(content_feature_map, 'b c h w -> b (h w) c')
        query_component_matrix = self.get_kqv_matrix(query_component_matrix, self.linears_query)
        query_component_matrix = torch.reshape(query_component_matrix, (B, H * W, m, d_channel))
        query_component_matrix = rearrange(query_component_matrix, 'b hw m d_channel -> (b m) hw d_channel')

        return query_component_matrix

    def get_component_key_value(self, component_sequence, keys=False):
        B, N, C = component_sequence.shape
        m = self.num_heads
        d_channel = C // m

        if keys:
            key_component_matrix = self.get_kqv_matrix(component_sequence, self.linears_key)
            key_component_matrix = torch.reshape(key_component_matrix, (B, N, m, d_channel))
            key_component_matrix = rearrange(key_component_matrix, 'b n m d_channel -> (b m) n d_channel')
            return key_component_matrix
        else:
            value_component_matrix = self.get_kqv_matrix(component_sequence, self.linears_value)
            value_component_matrix = torch.reshape(value_component_matrix, (B, N, m, d_channel))
            value_component_matrix = rearrange(value_component_matrix, 'b n m d_channel -> (b m) n d_channel')
            return value_component_matrix

    def cross_attention(self, content_feature, components, style_components, mask=None, dropout=None):

        B, C, H, W = content_feature.shape
        content_query = self.get_content_query(content_feature) 
        components_key = self.get_component_key_value(components, keys=True)
        style_components_value = self.get_component_key_value(style_components) 

        residual = content_query
        d_k = content_query.size(-1)
        scores = torch.matmul(content_query, components_key.transpose(-2, -1)) / math.sqrt(d_k) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        out = torch.matmul(p_attn, style_components_value)
        out = rearrange(out, '(b m) hw c -> b hw (c m)', m=self.num_heads)

        residual = rearrange(residual, '(b m) hw c -> b hw (c m)', m=self.num_heads)

        # add & norm
        out = self.layer_norm(out + residual) 
        out = self.multihead_concat_fc(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H)

        return out 

    def forward(self, content_feature, components, style_components):
        transfer_feature = self.cross_attention(content_feature, components, style_components)
        return transfer_feature

