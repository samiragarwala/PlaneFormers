from heapq import merge
import os
from pyexpat import model
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCELoss, MSELoss, L1Loss
from torch.utils.data import Subset, DataLoader, dataset
from tqdm import tqdm
import torchvision
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
from random import choice, sample
from types import SimpleNamespace
import pdb
import pickle
import quaternion
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pycocotools.mask as mask_util
import copy
from detectron2.config import CfgNode, get_cfg

sys.path.append("./SparsePlanes/sparsePlane/tools/")
from inference_sparse_plane import PlaneRCNN_Branch
import KMSolver

from planeformers.utils.dataset import *
from planeformers.models.planeformer import *
from planeformers.models.inference import *
from sparseplane.utils.mesh_utils import *
from sparseplane.config import get_sparseplane_cfg_defaults


# class to collate batch and pad plane input batches
class collate_batch:

    def __init__(self, padding_value=0.0, balance_cam=False, \
        hard_cam=False, params=None):
        self.padding_value = padding_value
        self.balance_cam = balance_cam
        self.hard_cam = hard_cam
        self.params = params
    
    def __call__(self, batch):
        # balancing pos and neg camera in batch before collating samples
        if self.balance_cam or self.hard_cam:
            tot_samples = list(range(self.params.batch_size))
            if self.balance_cam:
                pos_cams = random.sample(tot_samples, self.params.batch_size//2)
                # left 1/2 are random neg cams
            else:
                pos_cams = random.sample(tot_samples, self.params.batch_size//3)
                neg_cams = random.sample(list(set(tot_samples) - set(pos_cams)), self.params.batch_size//3)
                # left 1/3 are hard negative cams
            bal_batch = []

            # loading kmeans centroids for rotation and translation
            kmeans_rot = pickle.load(open(self.params.kmeans_rot, "rb"))
            kmeans_trans = pickle.load(open(self.params.kmeans_trans, "rb"))

            for i, sample in enumerate(batch):
                rot_cluster = kmeans_rot.predict(sample['gt_rot'].reshape(1, 4).numpy())[0]
                trans_cluster = kmeans_trans.predict(sample['gt_trans'].reshape(1,3).numpy())[0]
                camera_corr = 1

                if self.balance_cam:
                    if i not in pos_cams:
                        rot_cluster = random.choice([*range(0, rot_cluster), *range(rot_cluster + 1, 32)])
                        trans_cluster = random.choice([*range(0, trans_cluster), *range(trans_cluster + 1, 32)])
                        camera_corr = 0

                # hard cam case
                else:
                    if i not in pos_cams:

                        if i in neg_cams:
                            err_val = 0
                        else:
                            err_val = 1

                        # rot error
                        temp_rot = kmeans_rot.cluster_centers_
                        rot_err = np.abs(temp_rot @ sample['gt_rot'].reshape(4, 1).numpy())
                        rot_err[rot_err > 1] = 1
                        rot_err = 2 * np.arccos(rot_err) * 180/np.pi
                        rot_err[rot_err <= 45] = err_val
                        rot_err[rot_err > 45] = (1 - err_val)
                        rot_err_idx = np.flatnonzero(rot_err).tolist()
                        if rot_cluster in rot_err_idx:
                            rot_err_idx.remove(rot_cluster)
                        rot_cluster = random.choice(rot_err_idx)

                        # trans error
                        trans_err = np.linalg.norm(kmeans_trans.cluster_centers_ - \
                            sample['gt_trans'].reshape(1, 3).numpy(), axis=1)
                        trans_err[trans_err <= 3] = err_val
                        trans_err[trans_err > 3] = (1 - err_val)
                        trans_err_idx = np.flatnonzero(trans_err).tolist()
                        if trans_cluster in trans_err_idx:
                            trans_err_idx.remove(trans_cluster)
                        # accounting for outliers in dataset
                        if not len(trans_err_idx):
                            trans_cluster = random.choice([*range(0, trans_cluster), *range(trans_cluster + 1, 32)])
                        else:
                            trans_cluster = random.choice(trans_err_idx)

                        # setting neg camera corr
                        camera_corr = 0
                
                cam_rot = kmeans_rot.cluster_centers_[rot_cluster, :]
                cam_trans = kmeans_trans.cluster_centers_[trans_cluster, :]

                camera_info = {}
                camera_info['rotation'] = quaternion.quaternion(cam_rot[0], cam_rot[1], cam_rot[2], cam_rot[3])
                camera_info['position'] = cam_trans.reshape((1, 3))
                planes0_global_params = torch.tensor(get_plane_params_in_global(sample['planes0'].copy(), camera_info)).to(torch.float) * self.params.plane_param_scaling
                planes1 = torch.tensor(sample['planes1'].copy()).to(torch.float) * torch.tensor([1, -1, -1]).reshape(1, 3) * self.params.plane_param_scaling # suncg to habitat 

                if not hasattr(self.params, 'use_plane_params') or self.params.use_appearance_embeddings:
                    concat_emb0 = [sample['emb0']]
                    concat_emb1 = [sample['emb1']]
                else: 
                    concat_emb0 = []
                    concat_emb1 = []

                if hasattr(self.params, 'use_plane_params'):
                    if self.params.use_plane_params:
                        concat_emb0.append(planes0_global_params)
                        concat_emb1.append(planes1)
                else:
                    concat_emb0.append(planes0_global_params)
                    concat_emb1.append(planes1)


                if self.params.use_camera_conf_score:
                    num_p0 = sample['num_planes'][0]
                    num_p1 = sample['num_planes'][1]
                    rot_cluster_conf = torch.tensor(sample['rot_conf'])[rot_cluster].to(torch.float)
                    trans_cluster_conf = torch.tensor(sample['trans_conf'])[trans_cluster].to(torch.float)
                    concat_emb0.extend([rot_cluster_conf.repeat(num_p0, 1), trans_cluster_conf.repeat(num_p0, 1)])
                    concat_emb1.extend([rot_cluster_conf.repeat(num_p1, 1), trans_cluster_conf.repeat(num_p1, 1)])

                
                if self.params.use_plane_mask:
                    masks0_decoded = np.array([mask_util.decode(sample['instances0'][mask_idx]['segmentation']) for mask_idx in range(len(sample['instances0']))])
                    masks1_decoded = np.array([mask_util.decode(sample['instances1'][mask_idx]['segmentation']) for mask_idx in range(len(sample['instances1']))])
                    
                    warped_mask = warp_mask(masks0_decoded, sample['planes0'].copy(), camera_info)
                    
                    seg_masks0 = torch.flatten(
                        torchvision.transforms.Resize((self.params.mask_height, self.params.mask_width))(torch.FloatTensor(warped_mask)),
                        start_dim=1,
                    )
                    seg_masks1 = torch.flatten(
                        torchvision.transforms.Resize((self.params.mask_height, self.params.mask_width))(torch.FloatTensor(masks1_decoded.copy())),
                        start_dim=1,
                    )

                    concat_emb0.extend([seg_masks0])
                    concat_emb1.extend([seg_masks1])

                emb0 = torch.cat(concat_emb0, dim=1)
                emb1 = torch.cat(concat_emb1, dim=1)
                embs = torch.cat([emb0, emb1], dim=0)

                bal_sample = {}
                bal_sample['emb'] = embs
                bal_sample['num_planes'] = sample['num_planes']
                bal_sample['rot_centroid'] = torch.tensor(cam_rot).to(torch.float)
                bal_sample['trans_centroid'] = torch.tensor(cam_trans).to(torch.float)
                bal_sample['rot_conf'] = torch.tensor(sample['rot_conf'][rot_cluster]).to(torch.float)
                bal_sample['trans_conf'] = torch.tensor(sample['trans_conf'][trans_cluster]).to(torch.float)
                bal_sample['gt_plane_corr'] = sample['gt_plane_corr']
                bal_sample['gt_camera_corr'] = torch.tensor(camera_corr).to(torch.float)
                bal_sample['gt_rot'] = sample['gt_rot']
                bal_sample['gt_trans'] = sample['gt_trans']

                bal_batch.append(bal_sample)
            
            # overwriting provided batch samples with balanced batch for collation
            batch = bal_batch

        # collating data from samples into a batch for model to use
        embs = []
        num_planes = []
        gt_rot = []
        gt_trans = []
        gt_camera_corr = []
        rot_centroid = []
        trans_centroid = []
        rot_conf = []
        trans_conf = []

        for sample in batch:
            embs.append(sample['emb'])
            num_planes.append(sample['num_planes'])
            gt_rot.append(sample['gt_rot'])
            gt_trans.append(sample['gt_trans'])
            gt_camera_corr.append(sample['gt_camera_corr'])
            rot_centroid.append(sample['rot_centroid'])
            trans_centroid.append(sample['trans_centroid'])
            rot_conf.append(sample['rot_conf'])
            trans_conf.append(sample['trans_conf'])
    
        embs = pad_sequence(embs, padding_value=self.padding_value, batch_first=True)
        num_planes = torch.stack(num_planes, dim=0)
        gt_rot = torch.stack(gt_rot, dim=0)
        gt_trans = torch.stack(gt_trans, dim=0)
        gt_camera_corr = torch.stack(gt_camera_corr, dim=0)
        rot_centroid = torch.stack(rot_centroid, dim=0)
        trans_centroid = torch.stack(trans_centroid, dim=0)    
        rot_conf = torch.stack(rot_conf, dim=0)
        trans_conf = torch.stack(trans_conf, dim=0)        

        _, T, _ = embs.shape
        gt_corr = torch.zeros(len(batch), T, T)
        gt_plane_mask = torch.zeros(len(batch), T, T, dtype=torch.bool)
        for i, sample in enumerate(batch):
            num_p0, num_p1 = sample['num_planes']
            if not self.params.dataset_inference:
                gt_corr[i, :num_p0, :num_p1] = sample['gt_plane_corr']
            gt_plane_mask[i, :num_p0, :num_p1] = True

        proc_batch = {
            'emb': embs,
            'gt_plane_corr': gt_corr,
            'num_planes': num_planes,
        }
        proc_batch['gt_rot'] = gt_rot
        proc_batch['gt_trans'] = gt_trans
        proc_batch['gt_camera_corr'] = gt_camera_corr
        proc_batch['rot_centroid'] = rot_centroid
        proc_batch['trans_centroid'] = trans_centroid
        proc_batch['gt_plane_mask'] = gt_plane_mask
        proc_batch['rot_conf'] = rot_conf
        proc_batch['trans_conf'] = trans_conf
        
        return proc_batch


def create_src_padding_mask(embs, num_planes, device='cuda'):
    B, T, _ = embs.shape
    mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
    mask[torch.arange(0, B), num_planes] = True
    mask = mask.cumsum(dim=1)[:, :-1].to(torch.bool)
    return mask



def create_data_loader(dataset_type, params):

    dataset = PlaneEmbeddingDataset(dataset_type, params)
    shuffle = True

    if params.subset:
        torch.manual_seed(0)
        random.seed(0)
        idx = torch.randperm(len(dataset))[:params.subset]
        dataset = Subset(dataset, idx)
        shuffle = False

    if params.emb_format == "balance_cam" or params.emb_format == "hard_cam":
        drop_last_batch = True
    else:
        drop_last_batch = False

    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffle, \
        collate_fn=collate_batch(params.padding_value, (params.emb_format == 'balance_cam'), \
            (params.emb_format == "hard_cam"), params), drop_last=drop_last_batch)

    return dataloader


def compute_ipaa(pred_corr, gt_corr):
    '''
        pred_corr: x by y shape numpy association pred matrix
            each row/column should sum to 1
        gt_corr: np array corresponding to gt plane correspondence
    '''
    x, y = pred_corr.shape
    gt_corr = np.nonzero(gt_corr)
    num_instances = x + y - len(gt_corr[0])

    if len(gt_corr) == 0:
        inst_rows = inst_cols = []
    else:
        inst_rows = gt_corr[0]
        inst_cols = gt_corr[1]
    
    # verifying instances in gt_corr
    num_correct = np.sum((pred_corr[inst_rows, inst_cols] == 1))

    # verifying other rows and columns sum to 0
    for i in range(x):
        if i not in inst_rows and np.sum(pred_corr[i, :]) == 0:
            num_correct += 1
    for j in range(y):
        if j not in inst_cols and np.sum(pred_corr[:, j]) == 0:
            num_correct += 1

    ipaa = num_correct/num_instances
    return ipaa


def apply_km(cost_mat, threshold=0.3):
    cost_mat = (cost_mat * 1000).astype(np.int)
    assignments = KMSolver.solve(cost_mat, threshold=int(threshold * 1000))
    return assignments


def compute_loss(input, output, params, reduction='mean', device='cuda'):
    '''
        Computes BCE loss over all non-padded elements of output and target
        Parameters:
            input: Dictionary input into model
            output: Dictionary output from model
            params: parameters of model
            reduction: Mean to average over number of elements  
                or sum to aggregate and return total along with
                number of elements
    '''
    loss_dict = {}
    num_preds = {}

    # creating masks for plane masking
    plane_pred_mask = output['plane_mask']
    plane_gt_mask = input['gt_plane_mask']

    if params.model_name == "plane_corr":
        loss_fn = BCELoss(reduction='sum')
        loss = loss_fn(output['plane_corr'][plane_pred_mask], input['gt_plane_corr'][plane_gt_mask])
        loss_dict['plane_corr_loss'] = loss
        num_preds['plane_corr_loss'] = torch.sum(plane_gt_mask.to(torch.float))

    elif params.model_name == "plane_camera_corr":
        corr_loss_fn = BCELoss(reduction='sum')

        if params.use_l1_res_loss:
            residual_loss_fn = L1Loss(reduction='sum')
        else:
            residual_loss_fn = MSELoss(reduction='sum')

        B =  output['plane_corr'].shape[0]
        if not params.freeze_camera_corr_head:
            loss_dict['camera_corr_loss'] = params.camera_corr_wt * corr_loss_fn(output['camera_corr'], input['gt_camera_corr'])
            num_preds['camera_corr_loss'] = B
        
        camera_pose_mask = (input['gt_camera_corr'] == 1).reshape(-1)
        num_samples_gt_cam = camera_pose_mask.to(torch.float).sum()
        if num_samples_gt_cam > 0:
            
            if not params.freeze_plane_corr_head:
                # plane corr bce loss when correct camera provided
                loss_dict['plane_corr_loss'] = params.plane_corr_wt * corr_loss_fn(output['plane_corr'][camera_pose_mask, :, :][plane_pred_mask[camera_pose_mask, :, :]], \
                    input['gt_plane_corr'][camera_pose_mask, :, :][plane_gt_mask[camera_pose_mask, :, :]])
                num_preds['plane_corr_loss'] = plane_gt_mask[camera_pose_mask, :, :].to(torch.float).sum()

            if not params.freeze_camera_residual_head:
                # calculating loss for rot using mse
                rot_pred = output['rot_residual'][camera_pose_mask, :] + input['rot_centroid'][camera_pose_mask, :] # adding rot centroid to residual
                rot_pred = rot_pred/torch.norm(rot_pred, dim=1, keepdim=True) # normalizing to unit rotation quaternion
                rot_gt = input['gt_rot'][camera_pose_mask, :]
                loss_dict['rot_error_loss'] = params.rot_loss_wt * residual_loss_fn(rot_pred, rot_gt)
                num_preds['rot_error_loss'] = num_samples_gt_cam

                # calculating loss for trans using mse
                trans_pred = output['trans_residual'][camera_pose_mask, :] + input['trans_centroid'][camera_pose_mask, :] # adding trans centroid to residual
                trans_gt = input['gt_trans'][camera_pose_mask, :]
                loss_dict['trans_error_loss'] = params.trans_loss_wt * residual_loss_fn(trans_pred, trans_gt)
                num_preds['trans_error_loss'] = num_samples_gt_cam

    elif params.model_name == "camera_corr":
        corr_loss_fn = BCELoss(reduction='sum')
        loss_dict['camera_corr_loss'] = corr_loss_fn(output['camera_corr'], input['gt_camera_corr'])
        num_preds['camera_corr_loss'] = output['camera_corr'].shape[0]
        
    else:
        raise Exception("loss_fn: Model %s not defined" % (params.model_name))

    if reduction == 'mean':
        tot_loss = 0
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key]/num_preds[key]
            tot_loss = tot_loss + loss_dict[key]
        loss_dict['loss'] = tot_loss
        return loss_dict
    else:
        return loss_dict, num_preds

def compute_auroc(pred, target, padding_mask=None):

    # if padding mask provided mask out pred/target
    if padding_mask != None:
        mask = (padding_mask == False)
        pred = pred[mask].reshape(-1)
        target = target[mask].reshape(-1)

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    return roc_auc_score(target, pred)


# saving planeformer model predictions in format that
# can be run by sparseplanes eval code
@torch.no_grad()
def gen_eval_file(params, camera_search_len=1, device='cuda', no_file=False):

    assert camera_search_len >= 1 and camera_search_len <= 32

    # defining dataset
    dataset = PlaneEmbeddingDataset(params.eval_set, params)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    checkpoint = torch.load(params.eval_ckpt)

    # defining model
    model = PlaneFormer(params)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # restoring model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    if params.write_new_eval_file or no_file:
        output_dict = []
    else:
        path = params.eval_input_path
        f = open(path, 'rb')
        output_dict = pickle.load(f)


    if params.kmeans_rot:
        kmeans_rot = pickle.load(open(params.kmeans_rot, "rb"))
    if params.kmeans_trans:
        kmeans_trans = pickle.load(open(params.kmeans_trans, "rb"))

    for i in tqdm(range(len(dataset))):

        if params.store_camera_outputs:
            first_dim = camera_search_len * camera_search_len
            if i == 0:
                camera_corr_x = np.zeros((first_dim, 3))
                camera_corr_y = np.zeros(first_dim)
            else:
                camera_corr_x = np.concatenate((camera_corr_x, np.zeros((first_dim, 3))), axis=0)
                camera_corr_y = np.concatenate((camera_corr_y, np.zeros(first_dim)), axis=0)

            gt_rot_centroid = None
            gt_trans_centroid = None

        with open(os.path.join(os.path.join(params.path, params.eval_set), str(i) + '.pkl'), 'rb') as f:
            data = pickle.load(f)

        rot_confs = torch.tensor(data['camera']['logits']['rot'])
        trans_confs = torch.tensor(data['camera']['logits']['tran'])

        # top-k rotations and translations
        if camera_search_len == 32:
            top_rot_idx = torch.arange(32)
            top_trans_idx = torch.arange(32)
        else:
            _, top_rot_idx = torch.topk(rot_confs, 32)
            _, top_trans_idx = torch.topk(trans_confs, 32)


        best_camera_corr_score = -1
        best_rot_idx = -1
        best_trans_idx = -1
        for j in range(camera_search_len): 
            rot_idx = top_rot_idx[j].cpu().numpy()
            for k in range(camera_search_len):
                trans_idx = top_trans_idx[k].cpu().numpy()

                camera_centroids = {'rot_cluster': rot_idx, 'trans_cluster': trans_idx}
                sample = dataset.__getitem__(i, provided_camera=True, camera_centroids=camera_centroids)

                if params.store_camera_outputs and not gt_rot_centroid and not gt_trans_centroid:
                    gt_rot_centroid = kmeans_rot.predict(sample['gt_rot'].reshape(1, 4).numpy())[0]
                    gt_trans_centroid = kmeans_trans.predict(sample['gt_trans'].reshape(1, 3).numpy())[0]

                sample['emb'] = sample['emb'].unsqueeze(0).to(device)
                sample['num_planes'] = sample['num_planes'].unsqueeze(0).to(device)
                sample['gt_rot'] = sample['gt_rot'].unsqueeze(0).to(device)
                sample['gt_trans'] = sample['gt_trans'].unsqueeze(0).to(device)
                sample['rot_centroid'] = sample['rot_centroid'].unsqueeze(0).to(device)
                sample['trans_centroid'] = sample['trans_centroid'].unsqueeze(0).to(device)
                sample['rot_conf'] = sample['rot_conf'].unsqueeze(0).to(device)
                sample['trans_conf'] = sample['trans_conf'].unsqueeze(0).to(device)
                sample['gt_plane_corr'] = sample['gt_plane_corr'].unsqueeze(0).to(device)
                sample['gt_camera_corr'] = sample['gt_camera_corr'].unsqueeze(0).to(device)
                sample['rot_conf'] = sample['rot_conf'].unsqueeze(0).to(device)
                sample['trans_conf'] = sample['trans_conf'].unsqueeze(0).to(device)

                # src padding mask is None since only one example
                # processed at a time and thus no padding present
                pred = model(sample, src_padding_mask=None)

                if pred['camera_corr'][0] > best_camera_corr_score:
                    best_camera_corr_score = pred['camera_corr'][0]
                    best_pred = pred
                    best_proc_sample = sample
                    best_rot_idx = rot_idx
                    best_trans_idx = trans_idx

                if params.store_camera_outputs:
                    wt_idx = i * first_dim + j * camera_search_len + k
                    camera_corr_x[wt_idx, 0] = sample['rot_conf'].cpu().numpy()
                    camera_corr_x[wt_idx, 1] = sample['trans_conf'].cpu().numpy()
                    camera_corr_x[wt_idx, 2] = pred['camera_corr'][0].cpu().numpy()

                    if rot_idx == gt_rot_centroid and trans_idx == gt_trans_centroid:
                        camera_corr_y[wt_idx] = 1
                    else:
                        camera_corr_y[wt_idx] = 0


        # finding best camera according to camera corr head
        pred_camera_rot = kmeans_rot.cluster_centers_[best_rot_idx, :]
        pred_camera_trans = kmeans_trans.cluster_centers_[best_trans_idx, :]

        best_camera_centroids = {
            'position': np.array(pred_camera_trans.reshape((3,))),
            'rotation': np.array(pred_camera_rot.reshape((4,)))
        }

        if params.eval_add_cam_residual or params.num_residual_updates >= 1:
            pred_camera_rot += best_pred['rot_residual'][0, :].cpu().numpy()
            pred_camera_trans += best_pred['trans_residual'][0, :].cpu().numpy()

        # ensuring camera rot is a unit quaternion
        pred_camera_rot /= np.linalg.norm(pred_camera_rot)
        best_camera = {
            'position': pred_camera_trans.reshape((3,)),
            'rotation': pred_camera_rot.reshape((4,))
        }


        # finding planar correspondences
        num_p0 = best_proc_sample['num_planes'][0, 0]
        num_p1 = best_proc_sample['num_planes'][0, 1]
        cost_mat = 1 - best_pred['plane_corr'].squeeze(0)[:num_p0, num_p0: num_p0 + num_p1]
        pred_corr = apply_km(cost_mat.cpu().numpy(), threshold=params.plane_corr_threshold)

        if params.write_new_eval_file or no_file:
            temp_pred = {}
            temp_pred['best_camera'] = best_camera
            temp_pred['best_assignment'] = pred_corr
            temp_pred['plane_param_override'] = None
            temp_pred['best_camera_centroids'] = best_camera_centroids
            output_dict.append(temp_pred)
        else:
            output_dict[i]['best_assignment'] = pred_corr
            output_dict[i]['best_camera'] = best_camera
            output_dict[i]['best_camera_centroids'] = best_camera_centroids
            if "plane_param_override" in  output_dict[i].keys():
                output_dict[i]["plane_param_override"] = None

    if params.store_camera_outputs:
        with open(os.path.join(params.camera_output_dir, 'camera_corr_x.npy'), 'wb') as f:
            np.save(f, camera_corr_x)
        with open(os.path.join(params.camera_output_dir, 'camera_corr_y.npy'), 'wb') as f:
            np.save(f, camera_corr_y)

    if no_file:
        return output_dict
    else:
        with open(params.output_file, 'wb') as output_file:
            pickle.dump(output_dict, output_file, protocol=pickle.HIGHEST_PROTOCOL)


# function modifies best_camera in .pkl input file to use centroids 
#   only instead of centroids + residual during eval
def change_output_camera(input_file, output_file):
    f = open(input_file, 'rb')
    output_dict = pickle.load(f)

    for i in tqdm(range(len(output_dict))):
        output_dict[i]['best_camera'] = output_dict[i]['best_camera_centroids']

    with open(output_file, 'wb') as out_f:
        pickle.dump(output_dict, out_f, protocol=pickle.HIGHEST_PROTOCOL)

        

@torch.no_grad()
def validate(model, dataloader, params, device='cuda', writer=None, \
    iter_num=None):

    model.eval()
    stats = {} # tracking val stats
    val_loss_dict = {} # tracking loss-related val stats
    val_numel = {}

    # storing all pred and target for plane auroc computation
    plane_pred_list = []
    plane_target_list = []

    # storing pred and trarget lists for camera corr auroc calculation
    camera_pred_list = []
    camera_target_list = []

    # storing rot_err and trans_err lists
    rot_err_list = []
    trans_err_list = []

    for i, batch in tqdm(enumerate(dataloader)):
        batch['emb'] = batch['emb'].to(device)
        batch['num_planes'] = batch['num_planes'].to(device)
        batch['gt_plane_corr'] = batch['gt_plane_corr'].to(device)
        batch['gt_rot'] = batch['gt_rot'].to(device)
        batch['gt_trans'] = batch['gt_trans'].to(device)
        batch['gt_camera_corr'] = batch['gt_camera_corr'].to(device)
        batch['rot_centroid'] = batch['rot_centroid'].to(device)
        batch['trans_centroid'] = batch['trans_centroid'].to(device)
        batch['gt_plane_mask'] = batch['gt_plane_mask'].to(device)
        batch['rot_conf'] = batch['rot_conf'].to(device)
        batch['trans_conf'] = batch['trans_conf'].to(device)

        src_padding_mask = create_src_padding_mask(batch['emb'], torch.sum(batch['num_planes'], dim=1), \
            device=device)
        output = model(batch, src_padding_mask)

        loss_dict, numel_sample = compute_loss(batch, output, params, reduction='sum', device=device)
        for key in loss_dict.keys():
            if key not in val_loss_dict.keys():
                val_loss_dict[key] = 0
                val_numel[key] = 0
            val_loss_dict[key] += loss_dict[key].cpu().item()
            val_numel[key] += numel_sample[key]

        # computing camera corr auroc
        camera_pred_list.append(output["camera_corr"].reshape(-1))
        camera_target_list.append(batch["gt_camera_corr"].reshape(-1))
        camera_corr_mask = (batch['gt_camera_corr'] == 1).reshape(-1)
        if camera_corr_mask.to(torch.float).sum() == 0:
            continue

        # appending pred and target to lists
        masked_pred = output["plane_corr"][camera_corr_mask, :, :][output['plane_mask'][camera_corr_mask, :, :]].reshape(-1)
        B, T, _ = batch['emb'].shape
        masked_target = (batch["gt_plane_corr"][camera_corr_mask, :, :][batch['gt_plane_mask'][camera_corr_mask, :, :]]).reshape(-1)
        plane_pred_list.append(masked_pred)
        plane_target_list.append(masked_target)

        # evaluating rot/trans error regression (assumes zeros used as padding)
        rot_cor = output['rot_residual'][camera_corr_mask, :] + batch['rot_centroid'][camera_corr_mask, :]
        rot_cor /= torch.norm(rot_cor, dim=1, keepdim=True)
        pi = torch.tensor(np.pi, device=device)
        rot_error = 2 * torch.arccos(torch.clamp(torch.abs(torch.sum(rot_cor * batch['gt_rot'][camera_corr_mask, :], dim=1)), 0, 1)) * \
            180/pi
        rot_err_list.append(rot_error)

        trans_cor = output['trans_residual'][camera_corr_mask, :] + batch['trans_centroid'][camera_corr_mask, :]
        trans_error = torch.norm(trans_cor - batch['gt_trans'][camera_corr_mask, :], dim=1)
        trans_err_list.append(trans_error)


    # computing plane corr auroc
    if len(plane_pred_list) > 0:
        plane_pred_list = torch.cat(plane_pred_list)
        plane_target_list = torch.cat(plane_target_list)
        stats['plane_auroc'] = compute_auroc(plane_pred_list, plane_target_list)
        if writer:
            writer.add_scalar("val/plane_corr/auroc", stats['plane_auroc'], iter_num)

    # computing camera corr auroc
    if len(camera_pred_list) > 0:
        camera_pred_list = torch.cat(camera_pred_list)
        camera_target_list = torch.cat(camera_target_list)
        stats['camera_auroc'] = compute_auroc(camera_pred_list, camera_target_list)
        if writer:
            writer.add_scalar("val/camera_corr/auroc", stats['camera_auroc'], iter_num)

    if len(rot_err_list) > 0 and len(trans_err_list) > 0:
        rot_err_list = torch.cat(rot_err_list)
        trans_err_list = torch.cat(trans_err_list)
        stats['median_rot_error'] = torch.median(rot_err_list).item()
        stats['mean_rot_error'] = torch.mean(rot_err_list).item()
        stats['median_trans_error'] = torch.median(trans_err_list).item()
        stats['mean_trans_error'] = torch.mean(trans_err_list).item()

        if writer:
            writer.add_scalar("val/camera_pose/median_rot_error", stats['median_rot_error'], iter_num)
            writer.add_scalar("val/camera_pose/mean_rot_error", stats['mean_rot_error'], iter_num)
            writer.add_scalar("val/camera_pose/median_trans_error", stats['median_trans_error'], iter_num)
            writer.add_scalar("val/camera_pose/mean_trans_error", stats['mean_trans_error'], iter_num)

    tot_loss = 0
    for key in val_loss_dict.keys():
        val_loss_dict[key] /= val_numel[key]
        tot_loss += val_loss_dict[key]
        if writer:
            writer.add_scalar("val_loss/" + key, val_loss_dict[key], iter_num)
    if writer:
        writer.add_scalar("val_loss/tot_loss", tot_loss.item(), iter_num)

    model.train()

    return stats


def get_default_dataset_config(emb_format):
    params = SimpleNamespace()
    params.path = None # path to embedding dataset
    params.json_path = None # parent directory of json files for training/val/test
    params.kmeans_rot = "./SparsePlanes/sparsePlane/models/kmeans_rots_32.pkl"
    params.kmeans_trans = "./SparsePlanes/sparsePlane/models/kmeans_trans_32.pkl"
    params.emb_format = emb_format
    params.deg_cut_off = 30.0
    params.trans_cut_off = 1.0
    params.plane_corr_wt = 1.0
    params.camera_corr_wt = 1.0
    params.rot_loss_wt = 1.0
    params.trans_loss_wt = 0.5
    params.transform = None
    params.inference_pred_box = False
    params.subset = None
    params.padding_value = 0.0
    params.d_model = 899
    params.model_name = "plane_camera_corr"
    params.train = False
    params.restore_ckpt = None
    params.not_restore_optimizer = False
    params.eval = False
    params.gen_eval_file = False
    params.eval_ckpt = False
    params.eval_set = 'test'
    params.camera_search_len = 3
    params.plane_corr_threshold = 0.7
    params.nhead = 1
    params.fc_dim = 2048
    params.dropout = 0.1
    params.nlayers = 5
    params.freeze_camera_corr_head  = False
    params.freeze_plane_corr_head = False
    params.freeze_camera_residual_head = False
    params.use_camera_conf_score = False
    params.plane_param_scaling = 0.1
    params.use_plane_mask = True
    params.use_appearance_embeddings = True
    params.use_plane_params = True
    params.mask_height = 24
    params.mask_width = 32
    params.dataset_inference = False
    params.batch_size = 40
    params.project_ft = False
    params.use_plane_params = True
    params.use_camera_conf_score = False
    params.sparseplane_config = "./SparsePlanes/sparsePlane/tools/demo/config.yaml"
    params.transformer_on = True
    params.eval_add_cam_residual = True
    return params

# supporting older model configs which may not have all of the new params
def support_legacy_config(params):
    if not hasattr(params, 'use_camera_conf_score'):
        params.use_camera_conf_score = False
    if not hasattr(params, 'plane_param_scaling'):
        params.plane_param_scaling = 1
    if not hasattr(params, 'use_plane_mask'):
        params.use_plane_mask = False
    if not hasattr(params, 'mask_height'):
        params.mask_height = 24
    if not hasattr(params, 'mask_width'):
        params.mask_width = 32
    return params


# taken from sparseplane
# https://github.com/jinlinyi/p-sparse-plane/blob/d8d9514ba3c3b23a0f7de21e972173010fc652be/planeRecon/utils/relpose.py#L25
def get_relative_T_in_cam2_ref(R2, t1, t2):
    new_c2 = - np.dot(R2, t2)
    return np.dot(R2, t1) + new_c2

# taken from sparseplane
def get_relative_pose_from_datapoint(datapoint):
    q1 = datapoint[0]['camera']['rotation']
    q2 = datapoint[1]['camera']['rotation']
    t1 = datapoint[0]['camera']['position']
    t2 = datapoint[1]['camera']['position']
    if type(q1) == list:
        q1 = quaternion.from_float_array(q1)
    if type(q2) == list:
        q2 = quaternion.from_float_array(q2)
    if type(t1) == list:
        t1 = np.array(t1)
    if type(t2) == list:
        t2 = np.array(t2)
    relative_rotation = (q2.inverse() * q1)
    relative_translation = get_relative_T_in_cam2_ref(quaternion.as_rotation_matrix(q2.inverse()), t1, t2)

    rel_pose = {'position': relative_translation, 'rotation': relative_rotation}
    return rel_pose