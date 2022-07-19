from operator import concat
from platform import processor
import torch
import json
import numpy as np
import random
import os
from torch.functional import norm
from torch.utils.data import Dataset
from quaternion import rotate_vectors
from torch.nn.utils.rnn import pad_sequence
import pickle
import quaternion
import pycocotools.mask as mask_util
import sys
import torchvision
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color

from sparseplane.utils.mesh_utils import *
import planeformers.utils.visualization


def get_homography(camera_info, plane_param, h, w, focal_length = 517.97):
    # solve analytically https://en.wikipedia.org/wiki/Homography_(computer_vision)
    R = quaternion.as_rotation_matrix(camera_info['rotation'])
    R = np.diag([1,-1,-1]) @ R @ np.diag([1,-1,-1])
    t = np.array(camera_info['position']) * np.array([1, -1, -1])
    d = np.linalg.norm(plane_param)
    n = plane_param / max(d, 1e-6)
    H = R + t.reshape(3,1)@n.reshape(1, 3) / d
    
    offset_x = w/2
    offset_y = h/2
    K = [[focal_length, 0, offset_x],
        [0, focal_length, offset_y],
        [0, 0, 1]]
    H = K @ H @ np.linalg.inv(K)
    return H


def warp_mask(masks, planes, camera_info):
    warped = []
    planes = np.array(planes)
    assert len(masks) == len(planes)
    for i in range(len(masks)):
        h, w = masks[i].shape
        H = get_homography(camera_info, planes[i], h, w)
        mask_warped = cv2.warpPerspective(
            masks[i].astype(np.uint8), 
            H, 
            (w, h),
        )
        warped.append(mask_warped)
    return np.array(warped)

def debug(json, masks0, masks1, masks0_warped, idx):
    img = cv2.imread(json['0']['file_name'])
    common_instances_color = [random_color(rgb=True, maximum=1) for _ in range(len(masks0))]

    vis = Visualizer(img.copy())
    seg_blended = visualization.get_labeled_seg(masks0, vis, common_instances_color)
    cv2.imwrite(f'debug/{str(idx)}_0.png', seg_blended)
    img = cv2.imread(json['1']['file_name'])
    vis = Visualizer(img.copy())
    seg_blended = visualization.get_labeled_seg(masks0_warped, vis, common_instances_color)
    cv2.imwrite(f'debug/{str(idx)}_warped.png', seg_blended)

# two-view
class PlaneEmbeddingDataset(Dataset):

    def __init__(self, dataset_type, params):
        '''
            dataset_type: train/val/test
            params: parameters for dataset
            transform: Any transforms to apply to the data
        '''

        assert dataset_type in ['train', 'val', 'test']

        self.dataset_type = dataset_type
        self.transform = params.transform
        self.path = os.path.join(params.path, dataset_type)
        self.emb_format = params.emb_format
        self.deg_cut_off = params.deg_cut_off
        self.trans_cut_off = params.trans_cut_off
        self.use_camera_conf_score = params.use_camera_conf_score
        self.plane_param_scaling = params.plane_param_scaling
        self.use_plane_mask = params.use_plane_mask
        self.mask_height = params.mask_height
        self.mask_width = params.mask_width
        self.inference = params.dataset_inference
        self.use_appearance_embeddings = params.use_appearance_embeddings
        self.params = params

        if params.kmeans_rot:
            assert(os.path.exists(params.kmeans_rot))
            self.kmeans_rot = pickle.load(open(params.kmeans_rot, "rb"))
        if params.kmeans_trans:
            assert(os.path.exists(params.kmeans_trans))
            self.kmeans_trans = pickle.load(open(params.kmeans_trans, "rb"))

        with open(os.path.join(params.json_path, 'cached_set_%s.json' % (dataset_type)), 'r') as f:
            self.json_annot = json.load(f)['data']

    def __len__(self):
        return len(self.json_annot)
    
    def __getitem__(self, idx, provided_camera=False, camera_centroids=None, camera_transform=None):

        if torch.is_tensor(idx):
            idx = idx.item()

        with open(os.path.join(self.path, str(idx) + '.pkl'), 'rb') as f:
            data = pickle.load(f)

        if self.use_appearance_embeddings:
            emb0 = torch.squeeze(torch.tensor(data['0']['embedding']), dim=0)
            emb1 = torch.squeeze(torch.tensor(data['1']['embedding']), dim=0)
            # num planes
            num_p0 = emb0.shape[0]
            num_p1 = emb1.shape[0]
        else:
            num_p0 = torch.squeeze(torch.tensor(data['0']['embedding']), dim=0).shape[0]
            num_p1 = torch.squeeze(torch.tensor(data['1']['embedding']), dim=0).shape[0]

        gt_corr = torch.tensor(self.json_annot[idx]['gt_corrs'])

        proc_sample = {}

        # modifying embeddings as per format
        if self.emb_format == "appearance_only":
            pass
        elif self.emb_format == "plane_params" or self.emb_format == "plane_and_camera_params":
            planes0 = np.array(data['0']['pred_plane'])
            planes1 = np.array(data['1']['pred_plane'])

            gt_rot = torch.tensor(self.json_annot[idx]["rel_pose"]["rotation"]).reshape(4,).to(torch.float)
            gt_trans = torch.tensor(self.json_annot[idx]["rel_pose"]["position"]).reshape(3,).to(torch.float)
            gt_trans_cluster = self.kmeans_trans.predict(gt_trans.reshape(1, 3).numpy())[0]
            gt_rot_cluster = self.kmeans_rot.predict(gt_rot.reshape(1, 4).numpy())[0]
            

            if provided_camera and camera_transform is not None:
                cam_trans = camera_transform['trans'].reshape(1, 3).numpy()
                cam_rot = camera_transform['rot'].reshape(1, 4).numpy()

                trans_cluster = self.kmeans_trans.predict(cam_trans)[0]
                rot_cluster = self.kmeans_rot.predict(cam_rot)[0]

                cam_trans = cam_trans.reshape((3,))
                cam_rot = cam_rot.reshape((4,))

                if rot_cluster == gt_rot_cluster and trans_cluster == gt_trans_cluster:
                    camera_corr = 1
                else:
                    camera_corr = 0

            else:
                if not provided_camera:
                    if self.transform == "random_camera_corr":
                        '''In this transform, if camera_corr is 1 use GT rot/trans centroids and if camera_corr is 0
                            then randomly pick a centroid other than the GT rot/trans centroids'''
                        camera_corr = np.random.randint(0, 2)
                        trans_cluster = self.kmeans_trans.predict(gt_trans.reshape(1, 3).numpy())[0]
                        rot_cluster = self.kmeans_rot.predict(gt_rot.reshape(1, 4).numpy())[0]
                        if camera_corr == 0:
                            trans_cluster = random.choice([*range(0, trans_cluster), *range(trans_cluster + 1, 32)])
                            rot_cluster = random.choice([*range(0, rot_cluster), *range(rot_cluster + 1, 32)])
                    elif self.transform == "gt_camera_corr":
                        camera_corr = 1
                        trans_cluster = self.kmeans_trans.predict(gt_trans.reshape(1, 3).numpy())[0]
                        rot_cluster = self.kmeans_rot.predict(gt_rot.reshape(1, 4).numpy())[0]
                    elif self.transform == "rpnet_preds":
                        # using predicted rot/trans from RPNet if no transform specified
                        rot_cluster = torch.argmax(torch.tensor(data['camera']['logits']['rot'])).numpy()
                        trans_cluster = torch.argmax(torch.tensor(data['camera']['logits']['tran'])).numpy()
                    else:
                        raise Exception("PlaneEmbeddingDataset: No transform for calculating camera correspondence provided")

                else:
                    rot_cluster = camera_centroids['rot_cluster']
                    trans_cluster = camera_centroids['trans_cluster']

                    if rot_cluster == gt_rot_cluster and trans_cluster == gt_trans_cluster:
                        camera_corr = 1
                    else:
                        camera_corr = 0

                # extracting camera pose rot/trans centroids
                cam_rot = self.kmeans_rot.cluster_centers_[rot_cluster, :]
                cam_trans = self.kmeans_trans.cluster_centers_[trans_cluster, :]

            # cam 2 (i.e. planes1) is global frame, hence converting planes0 params to planes1 frame
            camera_info = {}
            camera_info['rotation'] = quaternion.quaternion(cam_rot[0], cam_rot[1], cam_rot[2], cam_rot[3])
            camera_info['position'] = cam_trans.reshape((1, 3))
            planes0_global_params = torch.tensor(get_plane_params_in_global(planes0, camera_info)).to(torch.float) * self.plane_param_scaling
            planes1 = torch.tensor(planes1).to(torch.float) * torch.tensor([1, -1, -1]).reshape(1, 3) * self.plane_param_scaling # suncg to habitat 

            # app emb
            if self.use_appearance_embeddings:
                concat_emb0 = [emb0]
                concat_emb1 = [emb1]
            else:
                concat_emb0 = []
                concat_emb1 = []

            if self.params.use_plane_params:
                concat_emb0.append(planes0_global_params)
                concat_emb1.append(planes1)

            # calculating camera pose centroid error compared to gt
            rot_err = np.abs(np.dot(gt_rot.numpy(), cam_rot.reshape(4,)))
            rot_err = min(rot_err, 1)
            rot_err = 2 * np.arccos(rot_err) * 180/np.pi
            trans_err = np.linalg.norm(gt_trans.numpy() - cam_trans.reshape(3,))
            if self.transform == "rpnet_preds" and not provided_camera:
                camera_corr = 0
                if rot_err <= self.deg_cut_off and trans_err <= self.trans_cut_off:
                    camera_corr = 1


            proc_sample['rot_centroid'] = torch.tensor(cam_rot).to(torch.float)
            proc_sample['trans_centroid'] = torch.tensor(cam_trans).to(torch.float)
            proc_sample['gt_rot'] = gt_rot
            proc_sample['gt_trans'] = gt_trans
            proc_sample['trans_conf'] = torch.tensor(np.array(data['camera']['logits_sms']['tran'])[trans_cluster]).to(torch.float)
            proc_sample['rot_conf'] = torch.tensor(np.array(data['camera']['logits_sms']['rot'])[rot_cluster]).to(torch.float)
            proc_sample['gt_camera_corr'] = torch.tensor(camera_corr).to(torch.float)
            proc_sample['gt_rot_dist'] = rot_err
            proc_sample['gt_trans_dist'] = trans_err

            if self.emb_format == "plane_and_camera_params":
                concat_emb0.extend([torch.tensor(cam_rot).to(torch.float).repeat(num_p0, 1), \
                    torch.tensor(cam_trans).to(torch.float).repeat(num_p0, 1)])
                concat_emb1.extend([torch.tensor([1, 0, 0, 0]).repeat(num_p1, 1), torch.tensor([0, 0, 0]).repeat(num_p1, 1)])

            if self.use_camera_conf_score:
                rot_cluster_conf = torch.tensor(data['camera']['logits_sms']['rot'])[rot_cluster].to(torch.float)
                trans_cluster_conf = torch.tensor(data['camera']['logits_sms']['tran'])[trans_cluster].to(torch.float)
                concat_emb0.extend([rot_cluster_conf.repeat(num_p0, 1), trans_cluster_conf.repeat(num_p0, 1)])
                concat_emb1.extend([rot_cluster_conf.repeat(num_p1, 1), trans_cluster_conf.repeat(num_p1, 1)])

            if self.use_plane_mask:
                masks0_decoded = np.array([mask_util.decode(data['0']['instances'][mask_idx]['segmentation']) for mask_idx in range(len(data['0']['instances']))])
                masks1_decoded = np.array([mask_util.decode(data['1']['instances'][mask_idx]['segmentation']) for mask_idx in range(len(data['1']['instances']))])
                
                warped_mask = warp_mask(masks0_decoded, data['0']['pred_plane'], camera_info)
                # debug(self.json_annot[idx], masks0_decoded, masks1_decoded, warped_mask, idx)
                
                seg_masks0 = torch.flatten(
                    torchvision.transforms.Resize((self.mask_height, self.mask_width))(torch.FloatTensor(warped_mask)),
                    start_dim=1,
                )
                seg_masks1 = torch.flatten(
                    torchvision.transforms.Resize((self.mask_height, self.mask_width))(torch.FloatTensor(masks1_decoded.copy())),
                    start_dim=1,
                )

                concat_emb0.extend([seg_masks0])
                concat_emb1.extend([seg_masks1])
            
            
            emb0 = torch.cat(concat_emb0, dim=1)
            emb1 = torch.cat(concat_emb1, dim=1)
            embs = torch.cat([emb0, emb1], dim=0)
        elif self.emb_format == "balance_cam" or self.emb_format == "hard_cam":
            if self.use_appearance_embeddings:
                proc_sample['emb0'] = emb0
                proc_sample['emb1'] = emb1
            else:
                proc_sample['emb0'] = []
                proc_sample['emb1'] = []
            proc_sample['trans_conf'] = np.array(data['camera']['logits_sms']['tran'])
            proc_sample['rot_conf'] = np.array(data['camera']['logits_sms']['rot'])
            proc_sample['planes0'] = np.array(data['0']['pred_plane'])
            proc_sample['planes1'] = np.array(data['1']['pred_plane'])
            proc_sample['trans_center'] = self.kmeans_trans.cluster_centers_
            proc_sample['rot_center'] = self.kmeans_rot.cluster_centers_
            proc_sample['instances0'] = data['0']['instances']
            proc_sample['instances1'] = data['1']['instances']
            proc_sample['gt_rot'] = torch.tensor(self.json_annot[idx]["rel_pose"]["rotation"]).reshape(4,).to(torch.float)
            proc_sample['gt_trans'] = torch.tensor(self.json_annot[idx]["rel_pose"]["position"]).reshape(3,).to(torch.float)

            if self.use_appearance_embeddings:
                embs = torch.cat([emb0, emb1], dim=0)
            else:
                embs = []
        else:
            raise Exception("PlaneEmbeddingDataset: Invalid embedding format %s given" % (self.emb_format))
        
        proc_sample['emb'] = embs
        proc_sample['num_planes'] =  torch.tensor([num_p0, num_p1])

        gt_corr_sparse_tensor = torch.zeros(num_p0, num_p1)
        if not self.inference:
            gt_corr_sparse_tensor[gt_corr[:, 0], gt_corr[:, 1]] = 1
        proc_sample['gt_plane_corr'] = gt_corr_sparse_tensor

        return proc_sample   