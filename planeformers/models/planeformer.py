import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PlaneFormer(nn.Module):

    def __init__(self, params, device='cuda'):
        super(PlaneFormer, self).__init__()

        self.transformer_on = params.transformer_on 
        self.project_ft = params.project_ft
        if self.project_ft:
            self.projection_mlp = nn.Linear(params.d_model, 899)
            params.d_model = 899

        if self.transformer_on:
            encoder_layers = TransformerEncoderLayer(params.d_model, params.nhead, params.fc_dim, params.dropout, batch_first=True)
            self.transformer_encoder = TransformerEncoder(encoder_layers, params.nlayers)

        self.device = device

        num_inputs_fc = params.d_model * 2
        self.camera_head = nn.Sequential(nn.Linear(num_inputs_fc * 2, num_inputs_fc), \
            nn.ReLU(), \
            nn.Linear(num_inputs_fc, num_inputs_fc//2), \
            nn.ReLU(), \
            nn.Linear(num_inputs_fc//2, num_inputs_fc//4), \
            nn.ReLU(), \
            nn.Linear(num_inputs_fc//4, num_inputs_fc//8),
            nn.ReLU(), \
            nn.Linear(num_inputs_fc//8, 1))
        if params.freeze_camera_corr_head:
            for model_param in self.camera_head.parameters():
                model_param.requires_grad = False

        self.rot_head = nn.Sequential(nn.Linear(num_inputs_fc * 2, num_inputs_fc), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc, num_inputs_fc//2), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//2, num_inputs_fc//4), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//4, num_inputs_fc//8),
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//8, 4))
        self.trans_head = nn.Sequential(nn.Linear(num_inputs_fc * 2, num_inputs_fc), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc, num_inputs_fc//2), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//2, num_inputs_fc//4), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//4, num_inputs_fc//8),
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//8, 3))
        if params.freeze_camera_residual_head:
            for model_param in self.rot_head.parameters():
                model_param.requires_grad = False
            for model_param in self.trans_head.parameters():
                model_param.requires_grad = False

        self.plane_corr_head = nn.Sequential(nn.Linear(num_inputs_fc * 2, num_inputs_fc), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc, num_inputs_fc//2), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//2, num_inputs_fc//4), \
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//4, num_inputs_fc//8),
                nn.ReLU(), \
                nn.Linear(num_inputs_fc//8, 1))
        if params.freeze_plane_corr_head:
            for model_param in self.plane_corr_head.parameters():
                model_param.requires_grad = False



    def forward(self, input, src_padding_mask):

        if self.project_ft:
            input['emb'] = self.projection_mlp(input['emb'])

        if self.transformer_on:
            transformer_out = self.transformer_encoder(input['emb'], src_key_padding_mask=src_padding_mask)
        else:
            transformer_out = input['emb']
        
        B, T, D = transformer_out.shape

        img1_global_ft_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=self.device)
        img1_global_ft_mask[torch.arange(B), input['num_planes'][:, 0]] = True
        img1_global_ft_mask = torch.cumsum(img1_global_ft_mask, dim=1)
        img1_global_ft_mask = img1_global_ft_mask[:, :-1].unsqueeze(2).repeat(1, 1, D).to(torch.bool)
        img1_global_ft = transformer_out.clone()
        img1_global_ft[img1_global_ft_mask] = 0
        img1_global_ft = torch.sum(img1_global_ft, dim=1)/input['num_planes'][:, 0].unsqueeze(1) # B x D
        img1_global_ft = img1_global_ft.unsqueeze(1).repeat(1, T, 1).unsqueeze(2).repeat(1, 1, T, 1) # B x T x T x D

        img2_global_ft_mask = torch.zeros(B, T + 1, dtype=torch.int8, device=self.device)
        img2_global_ft_mask[torch.arange(B), input['num_planes'][:, 0]] = 1
        img2_global_ft_mask[torch.arange(B), input['num_planes'][:, 0] + input['num_planes'][:, 1]] = 1
        img2_global_ft_mask = torch.cumsum(img2_global_ft_mask, dim=1)
        img2_global_ft_mask[img2_global_ft_mask > 1] = 0
        img2_global_ft_mask = img2_global_ft_mask.to(torch.bool)
        img2_global_ft_mask = (img2_global_ft_mask == False)
        img2_global_ft_mask = img2_global_ft_mask[:, :-1] # B x T
        img2_global_ft = transformer_out.clone()
        img2_global_ft[img2_global_ft_mask.unsqueeze(2).repeat(1, 1, D)] = 0
        img2_global_ft = torch.sum(img2_global_ft, dim=1)/input['num_planes'][:, 1].unsqueeze(1) # B x D
        img2_global_ft = img2_global_ft.unsqueeze(1).repeat(1, T, 1).unsqueeze(2).repeat(1, 1, T, 1) # B x T x T x D

        cat_embs = torch.cat([transformer_out.clone().unsqueeze(2).repeat(1, 1, T, 1), transformer_out.clone().unsqueeze(1).repeat(1, T, 1, 1), \
            img1_global_ft, img2_global_ft], dim=3)
        
        camera_corr = self.camera_head(cat_embs).squeeze(3)
        rot_res = self.rot_head(cat_embs)
        trans_res = self.trans_head(cat_embs)
        plane_corr = torch.sigmoid(self.plane_corr_head(cat_embs)).squeeze(3)

        # mask for output
        padding_mask = img2_global_ft_mask.clone().unsqueeze(1).repeat(1, T, 1) # B x T x T
        row_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=self.device)
        row_mask[torch.arange(B), input['num_planes'][:, 0]] = True
        row_mask = (torch.cumsum(row_mask, dim=1)[:, :-1]).unsqueeze(2).repeat(1, 1, T).to(torch.bool)
        padding_mask[row_mask] = True

        # averaging camera output to compute average score
        camera_corr[padding_mask] = 0
        rot_res[padding_mask.unsqueeze(3).repeat(1, 1, 1, 4)] = 0
        trans_res[padding_mask.unsqueeze(3).repeat(1, 1, 1, 3)] = 0
        pool_factor = input['num_planes'][:, 0] * input['num_planes'][:, 1]
        camera_corr = torch.sum(camera_corr, dim=[1, 2])/pool_factor
        rot_res = torch.sum(rot_res, dim=[1, 2])/pool_factor.unsqueeze(1)
        trans_res = torch.sum(trans_res, dim=[1, 2])/pool_factor.unsqueeze(1)

        output = {}
        output['camera_corr'] = torch.sigmoid(camera_corr)
        output['rot_residual'] = rot_res
        output['trans_residual'] = trans_res
        output['plane_corr'] = plane_corr
        output['plane_mask'] = (padding_mask == False)

        return output