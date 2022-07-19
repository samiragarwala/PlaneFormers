import numpy as np
import argparse, os, cv2, torch, pickle, quaternion
import imageio
import random
import shutil
import pycocotools.mask as mask_util
from collections import defaultdict
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.ndimage.measurements import center_of_mass
from scipy.special import softmax
from scipy.optimize import least_squares
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes, BoxMode, Instances, pairwise_iou
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.utils.colormap import random_color
from detectron2.engine import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from pytorch3d.structures import join_meshes_as_batch

import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'SparsePlanes/sparsePlane'))

from sparseplane.utils.metrics import compare_planes_one_to_one
from sparseplane.data import PlaneRCNNMapper

from sparseplane.config import get_sparseplane_cfg_defaults
from sparseplane.utils.mesh_utils import save_obj, get_camera_meshes, transform_meshes, rotate_mesh_for_webview, get_plane_params_in_global, get_plane_params_in_local
from sparseplane.utils.vis import get_single_image_mesh_plane
from sparseplane.visualization import create_instances, get_labeled_seg, draw_match, get_gt_labeled_seg

def scale_data(x):
    # 1D data , scale (0, 1)
    scaler = MinMaxScaler()
    scaler.fit(x)
    return scaler.transform(x)


def FPR_95(labels, scores):
    """
    compute FPR@95
    """
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]   
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)

def mp3d2habitat(planes):
    rotation = np.array([
        [1,0,0],
        [0,0,-1],
        [0,1,0],
    ])
    rotation = np.linalg.inv(rotation)
    return (rotation@np.array(planes).T).T


def eval_affinity(predictions, _filter_iou=0.7):
    """
    Evaluate plane correspondence.
    """
    labels = []
    preds = []
    for pred in predictions:
        labels.extend(pred['labels'])
        preds.extend(pred['preds'])
    if not len(labels):
        return
    auc = roc_auc_score(labels, preds)*100
    ap = average_precision_score(labels, preds)*100

    results = {f"ap@iou={_filter_iou}": ap, f"auc@iou={_filter_iou}": auc}
    print(results)
    return results


def file_name_from_image_id(image_id):
    raise "Deprecated"
    house_name, basename = image_id.split('_', 1)
    return os.path.join('/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v3/rgb', house_name, basename+'.png')



def compute_IPAA(pred_assignment_m, gt_assignment_list, IPAA_dict):
    wrong_count = 0
    gt_assignment_list = np.array(gt_assignment_list)
    if len(gt_assignment_list) != 0:
        common_row_idxs = gt_assignment_list[:,0]
        common_colomn_idxs = gt_assignment_list[:,1]
    else:
        common_row_idxs = []
        common_colomn_idxs = []
    if len(gt_assignment_list) != 0:
        for [row, column] in gt_assignment_list:
            if pred_assignment_m[row, column] != 1:
                wrong_count += 1
    for i in range(pred_assignment_m.shape[0]):
        if i not in common_row_idxs:
            if sum(pred_assignment_m[i, :]) != 0:
                wrong_count += 1
    for j in range(pred_assignment_m.shape[1]):
        if j not in common_colomn_idxs:
            if sum(pred_assignment_m[:, j]) != 0:
                wrong_count += 1
    p = float(wrong_count) / (pred_assignment_m.shape[0] + pred_assignment_m.shape[1] - len(gt_assignment_list))
    for key in IPAA_dict.keys():
        if (1-p) * 100 >= key:
            IPAA_dict[key] += 1


def compute_auc(IPAA_dict):
    try:
        return (np.array(list(IPAA_dict.values())) / IPAA_dict['0']).mean()
    except:
        return (np.array(list(IPAA_dict.values())) / IPAA_dict[0]).mean()


def angle_error_rot_vec(v1, v2):
    return 2*np.arccos(np.clip(np.abs(np.sum(np.multiply(v1, v2))), -1.0, 1.0))*180/np.pi

def angle_error_tran_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def normal_angle_error(pred, gt):
    """
    surface normal angle error
    """
    dot_prod = np.clip(np.dot(pred, gt), -1, 1)
    angle_error = np.arccos(dot_prod) / np.pi * 180.0
    return angle_error

def plot_IPAAs(IPAA_dicts, save_path, names=None):
    matplotlib.rc('xtick', labelsize=20)     
    matplotlib.rc('ytick', labelsize=20)
    fig, ax = plt.subplots(figsize=(30,30))
    for IPAA_dict, name in zip(IPAA_dicts, names):
        xs = IPAA_dict.keys()
        ys = np.array(list(IPAA_dict.values()))/IPAA_dict[0]
        if 'optimize' in name:
            ax.plot(xs, ys, '.-', label=name, linewidth=10, markersize=50)
        else:
            ax.plot(xs, ys, '.--', label=name, linewidth=10, markersize=50)
        
    ax.set_ylim(bottom=0)
    ax.set_xlabel('X',fontsize=30)
    ax.set_ylabel('IPAA-X',fontsize=30)
    ax.set_title('IPAA',fontsize=30)
    ax.legend(loc=3, prop={'size': 20})
    fig.savefig(save_path)
    plt.close()


def save_distance_matrix(mat, pred_assignment, gt_assignment, save_path, vmin=0.0, vmax=1.0, cost_flag=True):
    sz_i, sz_j = pred_assignment.shape
    max_sz = max(sz_i, sz_j)
    # try:
    if max_sz < 5:
        max_sz = 5
    elif max_sz < 10:
        max_sz = 10
    text = np.array([['']*sz_j]*sz_i).astype('<U2')
    for i in range(sz_i):
        for j in range(sz_j):
            if pred_assignment[i][j] != 0:
                text[i][j] = '?'
            if gt_assignment[i][j] != 0:
                text[i][j] += '*'
    if cost_flag:
        mat_vis = np.ones([max_sz, max_sz]) * np.infty
    else:
        mat_vis = np.zeros([max_sz, max_sz])
    mat_vis[:sz_i,:sz_j] = mat
    labels = (np.asarray(["{}\n{:.2f}".format(text,data) for text, data in zip(text.flatten(), mat[:sz_i, :sz_j].flatten())])).reshape(text.shape)
    labels_full = np.array([['']*max_sz]*max_sz).astype('<U6')
    labels_full[:sz_i,:sz_j] = labels
    plt.figure()
    sns.heatmap(mat_vis, annot=labels_full, fmt='s', vmin=vmin, vmax=vmax)#, annot_kws={'fontsize': 18}, cbar=False)
    plt.savefig(save_path)
    plt.close()



def setup(args):
    cfg = get_cfg()
    get_sparseplane_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args) # comment out to avoid random seed
    # Setup logger for "meshrcnn" module
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="planercnn")
    return cfg



class Camera_Branch():
    def __init__(self, d2_cfg):
        self.cfg = d2_cfg
        if self.cfg.MODEL.CAMERA_ON:
            with open(os.path.join('/Pool1/users/jinlinyi/workspace/SparsePlanes/sparsePlane/', self.cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH), 'rb') as f:
                self.kmeans_trans = pickle.load(f)
            with open(os.path.join('/Pool1/users/jinlinyi/workspace/SparsePlanes/sparsePlane/',self.cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH), 'rb') as f:
                self.kmeans_rots = pickle.load(f)

    def xyz2class(self, x, y, z):
        return self.kmeans_trans.predict([[x,y,z]])

    def quat2class(self, w, xi, yi, zi):
        return self.kmeans_rots.predict([[w, xi, yi, zi]])

    def class2xyz(self, cls):
        assert((cls >= 0).all() and (cls < self.kmeans_trans.n_clusters).all())
        return self.kmeans_trans.cluster_centers_[cls]

    def class2quat(self, cls):
        assert((cls >= 0).all() and (cls < self.kmeans_rots.n_clusters).all())
        return self.kmeans_rots.cluster_centers_[cls]

    def get_rel_camera(self, pred_dict, tran_topk=0, rot_topk=0):
        
        sorted_idx_tran = np.argsort(pred_dict['camera']['logits']['tran'].numpy())[::-1]
        sorted_idx_rot = np.argsort(pred_dict['camera']['logits']['rot'].numpy())[::-1]
        tran = self.class2xyz(sorted_idx_tran[tran_topk])
        tran_p = pred_dict['camera']['logits_sms']['tran'][sorted_idx_tran[tran_topk]]
        rot = self.class2quat(sorted_idx_rot[rot_topk])
        rot_p = pred_dict['camera']['logits_sms']['rot'][sorted_idx_rot[rot_topk]]
        camera_info = {'position': tran, 'position_prob': tran_p, 
                        'rotation': rot, 'rotation_prob': rot_p}
        return camera_info

class PlaneFormerResult():
    def __init__(self, args, score_threshold = 0.7, dataset='mp3d_val', calculate_geometric_error=True):
        """
        Cache:
        - Predicted camera poses
        - Predicted plane parameters
        - Predicted embedding distance matrix
        
        """
        cfg = setup(args)
        self.score_threshold = score_threshold

        # rcnn_cached_file = '/Pool1/users/jinlinyi/exp/planercnn_detectron2/e15_siamese_pred_box_embedding/e03_predbox_random/evaluate_0034999_gtbox/instances_predictions.pth'
        # rcnn_cached_file = '/Pool1/users/jinlinyi/exp/planercnn_detectron2/e20_benchmark/e01_baseline_pretrain_freeze/evaluate_14999_test_pycoco20/instances_predictions.pth'
        print(f"Loading rcnn data from {args.rcnn_cached_file}")
        assert('instances_predictions.pth' in args.rcnn_cached_file)
        assert(os.path.exists(args.rcnn_cached_file))
        rcnn_cached_file = args.rcnn_cached_file
        if 'val' in dataset:
            print("Please double check camera branch is reading val set")
        elif 'test' in dataset:
            print("Please double check camera branch is reading test set")
        self.gt_box = cfg.TEST.EVAL_GT_BOX
        with open(rcnn_cached_file, 'rb') as f:
            self.rcnn_data = torch.load(f)
        if not self.gt_box:
            self.filter_box_with_low_score()
        else:
            if 'file_name' not in self.rcnn_data[0]['0'].keys():
                for idx in range(len(self.rcnn_data)):
                    for i in range(2):
                        self.rcnn_data[idx][str(i)]['file_name'] = file_name_from_image_id(self.rcnn_data[idx][str(i)]["image_id"])
        self.camera_branch = Camera_Branch(d2_cfg=cfg)
        self.metadata = MetadataCatalog.get(dataset)
        self.load_input_dataset(dataset)

        if calculate_geometric_error and self.gt_box:
            self.calculate_geometric_error()
        self.optimized_dict = None
        if len(args.sparseplane_optimized_dict_path) != 0 and os.path.exists(args.sparseplane_optimized_dict_path):
            print(f"reading from {args.sparseplane_optimized_dict_path}")
            with open(args.sparseplane_optimized_dict_path, 'rb') as f:
                self.optimized_dict = pickle.load(f)
        if len(args.planeformer_optimized_dict_path) != 0 and os.path.exists(args.planeformer_optimized_dict_path):
            print(f"reading from {args.planeformer_optimized_dict_path}")
            with open(args.planeformer_optimized_dict_path, 'rb') as f:
                self.planeformer_optimized_dict = pickle.load(f)
            assert len(self.planeformer_optimized_dict) == len(self.optimized_dict)
            self.merge_dict()
        self.sanity_check()
    
    def merge_dict(self):
        for i in range(len(self.planeformer_optimized_dict)):
            for k in self.optimized_dict[i].keys():
                if k not in self.planeformer_optimized_dict[i].keys():
                    self.planeformer_optimized_dict[i][k] = self.optimized_dict[i][k]


    def sanity_check(self):
        for idx, key in enumerate(self.dataset_dict.keys()):
            assert(self.rcnnidx2datasetkey(idx) == key)

    def load_input_dataset(self, dataset):
        dataset_dict = {}
        dataset_list = list(DatasetCatalog.get(dataset))
        for dic in dataset_list:
            key0 = dic['0']['image_id']
            key1 = dic['1']['image_id']
            key = key0 + '__' + key1
            for i in range(len(dic['0']['annotations'])):
                dic['0']['annotations'][i]['bbox_mode'] = BoxMode(dic['0']['annotations'][i]['bbox_mode'])
            for i in range(len(dic['1']['annotations'])):
                dic['1']['annotations'][i]['bbox_mode'] = BoxMode(dic['1']['annotations'][i]['bbox_mode'])
            dataset_dict[key] = dic
        self.dataset_dict = dataset_dict

    def calculate_geometric_error(self):
        self.single_view_err = []
        for idx in range(len(self.rcnn_data)):
            err = {'idx':idx}
            err.update(self.calculate_geometric_error_by_idx(idx))
            self.single_view_err.append(err)
        
    def save_geometric_error(self, save_dir):
        for key in ['worst_l2_mean', 'worst_offset_mean', 'worst_norm_mean']:
            sorted_idx_list = sorted(np.arange(len(self.rcnn_data)), key=lambda x:self.single_view_err[x][key])
            xs = [i/len(self.rcnn_data)*100 for i in range(len(self.rcnn_data))]
            ys = [self.single_view_err[sorted_idx_list[i]][key] for i in range(len(self.rcnn_data))]
            matplotlib.rc('xtick', labelsize=20)     
            matplotlib.rc('ytick', labelsize=20)
            fig, ax = plt.subplots(figsize=(20,20))
            ax.plot(xs, ys, '.-', label=key, linewidth=10, markersize=50)                
            ax.set_ylim(bottom=0)
            ax.set_xlabel('top k%',fontsize=30)
            ax.set_ylabel(key,fontsize=30)
            ax.set_title(key+' by top k',fontsize=30)
            ax.legend(loc=3, prop={'size': 20})
            fig.savefig(os.path.join(save_dir, key+'.png'))
            plt.close()
        import pdb; pdb.set_trace()

    def calculate_geometric_error_by_idx(self, idx):
        assert(self.gt_box)
        gt_planes = self.get_gt_plane_by_idx(idx)
        rtn_dict = {}
        rtn_dict['0'] = compare_planes_one_to_one(self.rcnn_data[idx]['0']['pred_plane'], gt_planes['0'])
        rtn_dict['1'] = compare_planes_one_to_one(self.rcnn_data[idx]['1']['pred_plane'], gt_planes['1'])
        rtn_dict['worst_l2_mean'] = max(rtn_dict['0']['l2'], rtn_dict['1']['l2'])
        rtn_dict['worst_norm_mean'] = max(rtn_dict['0']['norm'], rtn_dict['1']['norm'])
        rtn_dict['worst_offset_mean'] = max(rtn_dict['0']['offset'], rtn_dict['1']['offset'])
        return rtn_dict

    def get_gt_plane_by_idx(self, idx):
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        gt_planes = {
            '0': [ann['plane'] for ann in self.dataset_dict[key]['0']['annotations']],
            '1': [ann['plane'] for ann in self.dataset_dict[key]['1']['annotations']],
        }
        return gt_planes


    def filter_box_with_low_score(self):
        for idx in range(len(self.rcnn_data)):
            chosens = {}
            for i in range(2):
                scores = self.rcnn_data[idx]['corrs'][str(i)]["embeddingbox"]['scores']
                chosens[i] = (scores > self.score_threshold).nonzero().flatten()
                if len(chosens[i]) == 0:
                    import pdb; pdb.set_trace()
                    pass
                self.rcnn_data[idx]['corrs'][str(i)]["embeddingbox"]['scores'] = self.rcnn_data[idx]['corrs'][str(i)]["embeddingbox"]['scores'][chosens[i]]
                self.rcnn_data[idx]['corrs'][str(i)]["embeddingbox"]['pred_boxes'] = self.rcnn_data[idx]['corrs'][str(i)]["embeddingbox"]['pred_boxes'][chosens[i]]
                self.rcnn_data[idx][str(i)]['pred_plane'] = self.rcnn_data[idx][str(i)]['pred_plane'][chosens[i]]
            if 'emb_aff' in self.rcnn_data[idx]['corrs'].keys():
                self.rcnn_data[idx]['corrs']['emb_aff'] = self.rcnn_data[idx]['corrs']['emb_aff'][chosens[0], :][:, chosens[1]]
            elif 'pred_aff' in self.rcnn_data[idx]['corrs'].keys():
                self.rcnn_data[idx]['corrs']['pred_aff'] = self.rcnn_data[idx]['corrs']['pred_aff'][chosens[0], :][:, chosens[1]]
            # filtered_gt_corrs = []
            # for original_corr in self.rcnn_data[idx]['corrs']['gt_corrs']:
            #     filtered_gt_corrs.append([chosens[0].tolist().index(original_corr[0]), chosens[1].tolist().index(original_corr[1])])
            
    def replace_pred_plane_by_gt_plane(self):
        assert(self.gt_box)
        for idx in range(len(self.rcnn_data)):
            key0 = self.rcnn_data[idx]['0']['image_id']
            key1 = self.rcnn_data[idx]['1']['image_id']
            key = key0 + '__' + key1
            gt_planes = {
                '0': [ann['plane'] for ann in self.dataset_dict[key]['0']['annotations']],
                '1': [ann['plane'] for ann in self.dataset_dict[key]['1']['annotations']],
            }
            self.rcnn_data[idx]['0']['pred_plane'] = torch.tensor(gt_planes['0'])
            self.rcnn_data[idx]['1']['pred_plane'] = torch.tensor(gt_planes['1'])

    def get_camera_info(self, idx, tran_topk, rot_topk):
        # if topk is -1, then use gt pose (NOT GT BINs!)
        print(tran_topk, rot_topk)
        return self.camera_branch.get_rel_camera([self.rcnn_data[idx]], tran_topk, rot_topk)[0]

    def get_gt_cam_topk(self, idx):
        return self.camera_branch.get_gt_cam_topk([self.rcnn_data[idx]])[0]

    def get_emb_distance_matrix(self, idx):
        if 'emb_aff' in self.rcnn_data[idx]['corrs']:
            return 1 - self.rcnn_data[idx]['corrs']['emb_aff'] * 2
        else: 
            return 1 - self.rcnn_data[idx]['corrs']['pred_aff']
    
    def get_geo_distance_matrix(self, idx, tran_topk, rot_topk):
        distance_matrices = defaultdict(dict)
        # l2
        geo_distance_matrix, numPlanes1, numPlanes2 = self.geo_consistency_loss.inference(
            [self.rcnn_data[idx]['0']], [self.rcnn_data[idx]['1']], [self.get_camera_info(idx, tran_topk, rot_topk)], distance='l2')
        distance_matrices.update(geo_distance_matrix)
        # normal angle
        normal_angle_matrix, numPlanes1, numPlanes2 = self.geo_consistency_loss.inference(
            [self.rcnn_data[idx]['0']], [self.rcnn_data[idx]['1']], [self.get_camera_info(idx, tran_topk, rot_topk)], distance='normal')
        distance_matrices.update(normal_angle_matrix)
        return distance_matrices

    def get_maskiou(self, idx):
        """
        calculate mask iou between predicted mask and gt masks
        """
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        mious = {}
        for i in range(2):
            gt_mask_rles = []
            for ann in self.dataset_dict[key][str(i)]['annotations']:
                if isinstance(ann['segmentation'], list):
                    polygons = [np.array(p, dtype=np.float64) for p in ann['segmentation']]
                    rles = mask_util.frPyObjects(polygons, self.dataset_dict[key][str(i)]["height"], self.dataset_dict[key][str(i)]["width"])
                    rle = mask_util.merge(rles)
                elif isinstance(ann['segmentation'], dict):  # RLE
                    rle = ann['segmentation']
                else:
                    raise TypeError(f"Unknown segmentation type {type(ann['segmentation'])}!")
                gt_mask_rles.append(rle)

            pred_mask_rles = [ins['segmentation'] for ins in self.rcnn_data[idx][str(i)]['instances']]
            miou = mask_util.iou(pred_mask_rles, gt_mask_rles, [0]*len(gt_mask_rles))
            mious[str(i)] = miou
        return mious

    def get_maskiou_merged(self, idx, pred_corr=None, gt_corr=None):
        """
        calculate mask iou between merged pred and merged gt
                gt_1    gt_2    gt_m
        pred_1  miou    0       miou(1)
        pred_2  0       miou    miou(2)
        pred_m  miou(1)  miou(2)  avg_miou(1,2)
        """
        mious = self.get_maskiou(idx)
        single2merge_dict = self.get_single2merge(idx, pred_corr=pred_corr, gt_corr=gt_corr)
        
        entry2gt_single_view = single2merge_dict['entry2gt_single_view']
        gt_single_view2entry = single2merge_dict['gt_single_view2entry']
        entry2pred_single_view = single2merge_dict['entry2pred_single_view']
        pred_single_view2entry = single2merge_dict['pred_single_view2entry']

        num_pred_entry = len(entry2pred_single_view.keys())
        num_gt_entry = len(entry2gt_single_view.keys())
        # pred_gt_merged_mask
        mask_iou = np.zeros((num_pred_entry, num_gt_entry))        
        for r in range(num_pred_entry):
            for c in range(num_gt_entry):
                pred_merged = entry2pred_single_view[r]['merged']
                gt_merged = entry2gt_single_view[c]['merged']
                pair_id_pred = entry2pred_single_view[r]['pair']
                pair_id_gt = entry2gt_single_view[c]['pair']
                ann_id_pred = entry2pred_single_view[r]['ann_id']
                ann_id_gt = entry2gt_single_view[c]['ann_id']
                if not pred_merged and not gt_merged:
                    # pred_single & gt_single
                    # Should be in the same image
                    if pair_id_pred != pair_id_gt:
                        continue
                    else:
                        miou_single = mious[pair_id_pred]
                        mask_iou[r][c] = miou_single[ann_id_pred, ann_id_gt]
                elif pred_merged and not gt_merged:
                    # pred_merged & gt_single
                    miou_single = mious[pair_id_gt]
                    mask_iou[r][c] = miou_single[ann_id_pred[int(pair_id_gt)], ann_id_gt]
                elif not pred_merged and gt_merged:
                    # pred_single & gt_merged
                    miou_single = mious[pair_id_pred]
                    mask_iou[r][c] = miou_single[ann_id_pred, ann_id_gt[int(pair_id_pred)]]
                elif pred_merged and gt_merged:
                    # pred_merge & gt_merged, average both
                    miou_single = mious[str(0)]
                    iou0 = miou_single[ann_id_pred[0], ann_id_gt[0]]
                    miou_single = mious[str(1)]
                    iou1 = miou_single[ann_id_pred[1], ann_id_gt[1]]
                    mask_iou[r][c] = (iou0 + iou1) / 2
                else:
                    raise "BUG"
        return mask_iou

    def assign_predbox_to_gtbox(self, idx, filter_mask_iou=0.5):
        """
        for each gtbox, assign a pred box whose mask iou is greater than 0.5.
        """
        mious = self.get_maskiou(idx)
        # Assign gt box to pred box
        selected_pred_box_id = {}
        for i in range(2):
            miou = mious[str(i)]
            iou_sorted, idx_sorted = torch.sort(torch.FloatTensor(miou), dim=0, descending=True)
            selected_pred_box_id[str(i)] = idx_sorted[0,:]
            selected_pred_box_id[str(i)][iou_sorted[0,:]<filter_mask_iou] = -1
        return selected_pred_box_id

    def get_gt_affinity_from_pred_box(self, idx):
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        selected_pred_box_id = self.assign_predbox_to_gtbox(idx)
        gtbox_corr = self.dataset_dict[key]['gt_corrs']
        predbox_corr = []
        for [gt_idx1, gt_idx2] in gtbox_corr:
            pred_idx1 = selected_pred_box_id['0'][gt_idx1]
            pred_idx2 = selected_pred_box_id['1'][gt_idx2]
            if pred_idx1 != -1 and pred_idx2 != -1:
                predbox_corr.append([pred_idx1,pred_idx2])
        return np.array(predbox_corr)


    def get_gt_affinity(self, idx, rtnformat='matrix', gtbox=True):
        """
        return gt affinity. 
        If gtbox is True, return gt affinity for gt boxes; 
        else return gt affinity for pred boxes.
        """
        if gtbox:
            key0 = self.rcnn_data[idx]['0']['image_id']
            key1 = self.rcnn_data[idx]['1']['image_id']
            key = key0 + '__' + key1
            corrlist = np.array(self.dataset_dict[key]['gt_corrs'])
        else:
            corrlist = self.get_gt_affinity_from_pred_box(idx)
        if rtnformat == 'list':
            return corrlist
        elif rtnformat == 'matrix':
            if gtbox:
                mat = torch.zeros((len(self.dataset_dict[key]['0']['annotations']), 
                                    len(self.dataset_dict[key]['1']['annotations'])))
            else:
                mat = torch.zeros((len(self.rcnn_data[idx]['0']['instances']),
                                    len(self.rcnn_data[idx]['1']['instances'])))
            for i in corrlist:
                mat[i[0], i[1]] = 1
            return mat
        else:
            raise NotImplementedError

    def get_pred_depth(self, idx):
        if 'pred_depth' in self.rcnn_data[idx]['0'].keys():
            pred_depth = {
                '0': self.rcnn_data[idx]['0']['pred_depth'],
                '1': self.rcnn_data[idx]['1']['pred_depth'],
            }
            return pred_depth
        return None

    def get_gt_depth(self, idx):
        gt_depth= {}
        for i in range(2):
            house, img_id = self.rcnn_data[idx][str(i)]['image_id'].split('_',1)
            depth_path = os.path.join('/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20/observations', house, img_id+'.pkl')
            with open(depth_path, 'rb') as f:
                obs = pickle.load(f)
            depth = obs['depth_sensor']
            gt_depth[str(i)] = depth
        return gt_depth

    def get_pcd_normal_in_global(self, idx, tran_topk, rot_topk, 
                gt_plane=False, pred_camera=None, gt_segm=False, 
                plane_param_override=None, pred_corr=None, boxscore=False, 
                gt2pred = False, merge=True,
                ):
        """
        if topk is -1, then use gt pose (NOT GT BINs!)
        if tran_topk == -2 and rot_topk == -2, then pred_camera should not be None, this is used for non-binned camera.
        """
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        output = {}

        if gt2pred:
            selected_pred_box_id = self.assign_predbox_to_gtbox(idx)
        for i in range(2):
            img_file = self.rcnn_data[idx][str(i)]['file_name']
            height, width = self.dataset_dict[key][str(i)]['height'], self.dataset_dict[key][str(i)]['width']
            if i == 0:
                if pred_camera is None:
                    camera_info = self.get_camera_info(idx, tran_topk, rot_topk)
                    camera_info = {'position': np.array(camera_info['position']), 'rotation': quaternion.from_float_array(camera_info['rotation'])}
                else:
                    assert(tran_topk == -2 and rot_topk == -2)
                    camera_info = {'position': np.array(pred_camera['position']), 'rotation': quaternion.from_float_array(pred_camera['rotation'])}
            else:
                camera_info = {'position': np.array([0,0,0]), 'rotation': np.quaternion(1,0,0,0)}
            if not gt_plane or not gt_segm:
                p_instance = create_instances(self.rcnn_data[idx][str(i)]['instances'], [height, width], 
                                                    pred_planes=self.rcnn_data[idx][str(i)]['pred_plane'].numpy(), 
                                                    conf_threshold=self.score_threshold)
            if gt_plane:
                plane_params = [ann['plane'] for ann in self.dataset_dict[key][str(i)]['annotations']]
            else:
                if plane_param_override is None:
                    plane_params = p_instance.pred_planes
                else:
                    plane_params = plane_param_override[str(i)]
            if gt_segm:
                segmentations = [ann['segmentation'] for ann in self.dataset_dict[key][str(i)]['annotations']]
            else:
                segmentations = p_instance.pred_masks
            verts_list = get_single_image_pcd(plane_params, segmentations, height=height, width=width)
            verts_list = transform_verts_list(verts_list, camera_info)
            if boxscore:
                scores = p_instance.scores
            else:
                scores = np.ones(len(verts_list))
            output[str(i)] = {
                'verts_list': verts_list,
                'plane': get_plane_params_in_global(plane_params, camera_info),
                'scores': scores,
            }
            if gt2pred:
                output[str(i)]['gt2pred'] = selected_pred_box_id[str(i)]
        if merge:
            if pred_corr is None:
                output['corrs'] = np.array(self.get_gt_affinity(idx, rtnformat='list', gtbox=True))
            else:
                output['corrs'] = pred_corr
            # merge pcd
            merged_pcd = []
            merged_plane = []
            merged_score = []
            normal = {}
            offset = {}
            offset['0'] = np.maximum(np.linalg.norm(output['0']['plane'], ord=2, axis=1), 1e-5).reshape(-1,1)
            offset['1'] = np.maximum(np.linalg.norm(output['1']['plane'], ord=2, axis=1), 1e-5).reshape(-1,1)
            
            normal['0'] = output['0']['plane'] / offset['0']
            normal['1'] = output['1']['plane'] / offset['1']
            
            for i in range(2):
                for ann_id in range(len(output[str(i)]['verts_list'])):
                    if ann_id not in output['corrs'][:,i]:
                        merged_pcd.append(output[str(i)]['verts_list'][ann_id])
                        merged_plane.append(output[str(i)]['plane'][ann_id])
                        merged_score.append(output[str(i)]['scores'][ann_id])
            for ann_id in output['corrs']:
                # concat pcd
                merged_pcd.append(torch.cat((output['0']['verts_list'][ann_id[0]], output['1']['verts_list'][ann_id[1]])))
                # average normal
                normal_pair = np.vstack((normal['0'][ann_id[0]], normal['1'][ann_id[1]]))
                w, v = eigh(normal_pair.T@normal_pair)
                avg_normals = v[:,np.argmax(w)]
                if (avg_normals@normal_pair.T).sum() < 0:
                    avg_normals = - avg_normals
                # average offset
                avg_offset = (offset['0'][ann_id[0]] + offset['1'][ann_id[1]]) / 2
                merged_plane.append(avg_offset * avg_normals)
                # max score
                merged_score.append(max(output['0']['scores'][ann_id[0]], output['1']['scores'][ann_id[1]]))
            output['merged'] = {
                'verts_list': merged_pcd,
                'plane': np.array(merged_plane),
                'scores': np.array(merged_score)
            }
        return output

    def evaluate_matching(self, return_dict=None, method='IPAA', verbose=True):
        if method == 'IPAA':
            IPAA_dict = {}
            for i in range(11):
                IPAA_dict[i*10] = 0
            for idx in sorted(return_dict.keys()):
                pred_assignment = return_dict[idx]['best_assignment']
                gt_assignment_list = self.get_gt_affinity(idx, rtnformat='list', gtbox=self.gt_box)
                compute_IPAA(pred_assignment, gt_assignment_list, IPAA_dict)
            if verbose:
                print("[IPAA_dict]")
                print(IPAA_dict)
                print(compute_auc(IPAA_dict))
            return IPAA_dict
        elif method == 'IPAA_sorted_by_single_view_prediction':
            topks = [0.25, 0.50, 0.75, 1.00]
            idx_list = list(return_dict.keys())
            sorted_idx_list = sorted(idx_list, key=lambda x:self.single_view_err[x]['worst_l2_mean'])
            IPAA_dicts_by_topk_singleview = {}
            for topk in topks:
                IPAA_dict = {}
                for i in range(11):
                    IPAA_dict[i*10] = 0
                for idx in sorted_idx_list[:int(len(sorted_idx_list)*topk)]:
                    pred_assignment = return_dict[idx]['best_assignment']
                    gt_assignment_list = self.get_gt_affinity(idx, rtnformat='list', gtbox=False)
                    compute_IPAA(pred_assignment, gt_assignment_list, IPAA_dict)
                IPAA_dicts_by_topk_singleview[topk] = IPAA_dict
            return IPAA_dicts_by_topk_singleview
        elif method == 'recall':
            recall_dict = {}
            for i in range(11):
                recall_dict[i*10] = 0
            for idx in sorted(return_dict.keys()):
                pred_assignment = return_dict[idx]['best_assignment']
                gt_assignment = self.get_gt_affinity(idx, gtbox=False).numpy()
                tp = (pred_assignment * gt_assignment).sum()
                fp = (pred_assignment * (1-gt_assignment)).sum()
                fn = ((1-pred_assignment) * gt_assignment).sum()
                recall = tp / (tp+fn)
                for i in recall_dict.keys():
                    if recall >= i / 100:
                        recall_dict[i] += 1
            print("[recall_dict]")
            print(recall_dict)
            print(compute_auc(recall_dict))
            return recall_dict
        elif method == 'AP_FPR_95':
            labels = []
            scores = []
            for idx in sorted(return_dict.keys()):
                distance_m = return_dict[idx]['distance_m']
                gt_assignment = self.get_gt_affinity(idx, gtbox=self.gt_box).numpy()

                # cam_prob = self.get_camera_info(idx, return_dict[idx]['best_tran_topk'], return_dict[idx]['best_rot_topk'])
                # distance_m = return_dict[idx]['distance_m'] * 0.432+ \
                #     -np.log(cam_prob['position_prob']) *  0.166 + \
                #     -np.log(cam_prob['rotation_prob']) * 0.092
                # import pdb; pdb.set_trace()
                scores.extend((1-distance_m).flatten())
                labels.extend(gt_assignment.flatten())
                
            scores = np.array(scores)
            scores_scaled = scale_data(np.array(scores).reshape(-1,1)).flatten()
            labels = np.array(labels)
            auc = roc_auc_score(labels, scores_scaled)*100
            ap = average_precision_score(labels, scores_scaled)*100
            fpr = FPR_95(labels, scores_scaled)
            return {'ap': ap, 'auc': auc, 'fpr': fpr}
        else:
            raise NotImplementedError

    def evaluate_plane(self, return_dict=None):
        assert(self.gt_box)
        if 'plane_param_override' not in return_dict[0].keys():
            print("Warning: No plane_param_override in results")
            
        metrics = {'l2':[], 'norm':[], 'offset':[]}
        for idx in sorted(return_dict.keys()):
            gt_plane = self.get_gt_plane_by_idx(idx)
            gt_planes = torch.cat([torch.FloatTensor(gt_plane['0']), torch.FloatTensor(gt_plane['1'])], dim=0)
            try:
                pred_plane = return_dict[idx]['plane_param_override']
                pred_planes = torch.cat([torch.FloatTensor(pred_plane['0']), torch.FloatTensor(pred_plane['1'])], dim=0)
            except:
                pred_planes = torch.cat([self.rcnn_data[idx]['0']['pred_plane'], self.rcnn_data[idx]['1']['pred_plane']], dim=0)
            metric = compare_planes_one_to_one(pred_planes, gt_planes)
            metric['norm'] *= (180/np.pi)
            for key in metric:
                metrics[key].append(metric[key])
        print("Plane metrics")
        metrics['l2_mean'] = np.mean(metrics['l2'])
        metrics['l2_median'] = np.median(metrics['l2'])
        print(f"l2_mean: {metrics['l2_mean']}, l2_median: {metrics['l2_median']}")
        metrics['norm_mean'] = np.mean(metrics['norm'])
        metrics['norm_median'] = np.median(metrics['norm'])
        print(f"norm_mean: {metrics['norm_mean']}, norm_median: {metrics['norm_median']}")
        metrics['offset_mean'] = np.mean(metrics['offset'])
        metrics['offset_median'] = np.median(metrics['offset'])
        print(f"offset_mean: {metrics['offset_mean']}, offset_median: {metrics['offset_median']}")
        return metrics


    def evaluate_camera(self, return_dict=None, sorted_by_single_view_prediction=False, verbose=True):

        if sorted_by_single_view_prediction:
            topks = [0.25, 0.50, 0.75, 1.00]
            idx_list = list(return_dict.keys())
            sorted_idx_list = sorted(idx_list, key=lambda x:self.single_view_err[x]['worst_l2_mean'])
            camera_eval_dicts_by_topk_singleview = {}
            for topk in topks:
                camera_eval_dict = {}
                tran_errs = []
                rot_errs = []
                for idx in sorted_idx_list[:int(len(sorted_idx_list)*topk)]:
                    gt_cam = self.get_camera_info(idx, -1, -1)
                    if return_dict is None:
                        pred_cam = self.get_camera_info(idx, 0, 0)
                    else:
                        pred_cam = return_dict[idx]['best_camera']
                    # Error - translation
                    tran_errs.append(np.linalg.norm(pred_cam['position'] - gt_cam['position']))
                    # Error - rotation
                    d = np.abs(np.sum(np.multiply(pred_cam['rotation'], gt_cam['rotation'])))
                    d = np.clip(d, -1, 1)
                    rot_errs.append(2 * np.arccos(d) * 180 / np.pi)

                tran_acc = sum(_ < 1 for _ in tran_errs)/len(tran_errs)
                rot_acc = sum(_ < 30 for _ in rot_errs)/len(rot_errs)

                median_tran_err = np.median(np.array(tran_errs))
                mean_tran_err = np.mean(np.array(tran_errs))
                median_rot_err = np.median(np.array(rot_errs))
                mean_rot_err = np.mean(np.array(rot_errs))
                if verbose:
                    print('Mean Error [tran, rot]: ', mean_tran_err, mean_rot_err)
                    print('Median Error [tran, rot]: ', median_tran_err, median_rot_err)
                    print('Accuracy [tran, rot]:', tran_acc, rot_acc)
                camera_eval_dict = {
                    'tran_errs': np.array(tran_errs),
                    'rot_errs': np.array(rot_errs),
                    'mean_tran_err': mean_tran_err, 
                    'mean_rot_err': mean_rot_err,
                    'median_tran_err': median_tran_err,
                    'median_rot_err': median_rot_err,
                    'tran_acc': tran_acc,
                    'rot_acc': rot_acc,
                }
                camera_eval_dicts_by_topk_singleview[topk] = camera_eval_dict
            return camera_eval_dicts_by_topk_singleview
        else:
            tran_errs = []
            tran_angle_errs = []
            rot_errs = []
            for idx in tqdm(range(len(self.rcnn_data))):
                gt_cam = self.get_camera_info(idx, -1, -1)
                if return_dict is None:
                    pred_cam = self.get_camera_info(idx, 0, 0)
                else:
                    pred_cam = return_dict[idx]['best_camera']
                # Error - translation
                tran_errs.append(np.linalg.norm(pred_cam['position'] - gt_cam['position']))
                tran_angle = angle_error_tran_vec(pred_cam['position'], gt_cam['position'])
                tran_angle = np.minimum(tran_angle, 180 - tran_angle)  # ambiguity 
                tran_angle_errs.append(tran_angle)
                # Error - rotation
                if type(pred_cam['rotation']) != np.ndarray:
                    raise "Need to convert quaternion to np array"
                d = np.abs(np.sum(np.multiply(pred_cam['rotation'], gt_cam['rotation'])))
                d = np.clip(d, -1, 1)
                rot_errs.append(2 * np.arccos(d) * 180 / np.pi)

            tran_acc = sum(_ < 1 for _ in tran_errs)/len(tran_errs)
            tran_angle_acc = sum(_ < 30 for _ in tran_angle_errs)/len(tran_angle_errs)
            rot_acc = sum(_ < 30 for _ in rot_errs)/len(rot_errs)

            median_tran_err = np.median(np.array(tran_errs))
            mean_tran_err = np.mean(np.array(tran_errs))
            median_tran_angle_err = np.median(np.array(tran_angle_errs))
            mean_tran_angle_err = np.mean(np.array(tran_angle_errs))
            median_rot_err = np.median(np.array(rot_errs))
            mean_rot_err = np.mean(np.array(rot_errs))

            if verbose:
                print('Mean Error [tran, tran_angle, rot]: ', mean_tran_err, mean_tran_angle_err, mean_rot_err)
                print('Median Error [tran, tran_angle, rot]: ', median_tran_err, median_tran_angle_err, median_rot_err)
                print('Accuracy [tran, tran_angle, rot]:', tran_acc, tran_angle_acc, rot_acc)
                print("& \multicolumn{3}{c}{Translation (meters)} & \multicolumn{3}{c}{Translation (degrees)} & \multicolumn{3}{c}{Rotation  (degrees)} \\\\")
                print("Method & Median & Mean & (Err $\leq$ 1m)\% & Median    & Mean & (Err $\leq$ 30$^{ \circ }$)\% & Median    & Mean & (Err $\leq$ 30$^{ \circ }$)\% \\\\")
                print(f"{median_tran_err:.2f} & {mean_tran_err:.2f} & {tran_acc*100:.2f} & {median_tran_angle_err:.2f} & {mean_tran_angle_err:.2f} & {tran_angle_acc*100:.2f} & {median_rot_err:.2f} & {mean_rot_err:.2f} & {rot_acc*100:.2f} \\\\ ")
            camera_eval_dict = {
                'tran_errs': np.array(tran_errs),
                'rot_errs': np.array(rot_errs),
                'mean_tran_err': mean_tran_err, 
                'mean_rot_err': mean_rot_err,
                'median_tran_err': median_tran_err,
                'median_rot_err': median_rot_err,
                'tran_acc': tran_acc,
                'rot_acc': rot_acc,
            }
            return camera_eval_dict

    def get_matching_list(self, affinity_m, affinity_idx, sz_i):
        matching_list = []
        for i in range(sz_i):
            options = []
            for j in affinity_idx[i]:
                if affinity_m[i][j] <= 0.5:
                    continue
                options.append(j)
            if len(options) == 0:
                options.append(-1)
            matching_list.append(options)
        return matching_list

    def is_valid_matching(self, matching):
        """
        Matching proposal should not contain duplicate values except -1.
        """
        cnt = collections.Counter(matching)
        for k in cnt:
            if k != -1 and cnt[k] > 1:
                return False
        return True

    def get_assignment(self, idx, distance_matrix, method='hungarian', weight=None):
        if method == 'hungarian':
            row_ind, col_ind = linear_sum_assignment(distance_matrix)
            p = np.zeros_like(distance_matrix)
            p[row_ind, col_ind] = 1
            p = torch.from_numpy(p).float()
            return p
        elif method == 'km':
            '''
            km: Hungarian Algo
            if the distance > threshold, even it is smallest, it is also false.
            '''
            cost_matrix = (distance_matrix.numpy() * 1000).astype(np.int)

            prediction_matrix_km = KMSolver.solve(cost_matrix, threshold=int((1-weight['threshold'])*1000))
            return prediction_matrix_km
        elif method == 'sinkhorn':
            if distance_matrix.dim() == 2:
                distance_matrix = distance_matrix.unsqueeze(0)
            p = log_optimal_transport(1-distance_matrix)
            return p

        elif method == 'proposal':
            # calculate affinity matrix and matching
            topk_match = 3
            upperbound_match = 128

            num_plane1, num_plane2 = distance_matrix.shape
            affinity_m = 1 - distance_matrix
            if self.gt_box:
                gt_affinity_m = self.get_gt_affinity(idx)
                gt_matching_idx = np.argsort(gt_affinity_m, axis=1).numpy()[:,::-1]
                gt_matching_list = self.get_matching_list(gt_affinity_m, gt_matching_idx, num_plane1)
            affinity_idx = np.argsort(affinity_m, axis=1).numpy()[:,::-1][:, :topk_match]
            
            af_options = []
            for i in range(num_plane1):
                options = []
                for j in affinity_idx[i]:
                    if affinity_m[i][j] <= weight['threshold']:
                        continue
                    options.append(j)
                af_options.append(options)
            # top k
            matching_proposals = []
            gt_match_in_proposal = False
            for _ in range(upperbound_match):
                matching = []
                num_nomatch = 0
                for i in range(num_plane1):
                    options = []
                    for op in af_options[i]:
                        if op not in matching:
                            options.append(int(op))
                    options.append(-1)
                    m = random.choice(options)
                    if m == -1:
                        num_nomatch += 1
                    matching.append(m)

                # check matching is valid
                if not self.is_valid_matching(matching):
                    raise RuntimeError('invalid matching')

                # compute scores
                scores = []
                for i, j in enumerate(matching):
                    if j == -1:
                        continue
                    scores.append(affinity_m[i, j])
                scores = np.array(scores)

                # check if gt_matching_list in matching_proposals
                if self.gt_box:
                    if (matching == np.array(gt_matching_list).flatten()).all():
                        gt_match_in_proposal = True
                matching_proposals.append([matching, num_nomatch, scores])
            return matching_proposals
        else:
            raise NotImplementedError

    def save_original_img(self, idx, output_dir, prefix='', combined=False):
        if combined:
            blended = {}
            for i in range(2):
                blended[str(i)] = cv2.imread(self.rcnn_data[idx][str(i)]['file_name'], cv2.IMREAD_COLOR)[:,:,::-1]
            gt_matching_fig = draw_match(blended['0'], blended['1'], np.array([]), np.array([]),
                                        [], [], vertical=True, outlier_color=[0,176,240])
            gt_matching_fig.save(os.path.join(output_dir, f'{prefix}.png'))
        else:
            for i in range(2):
                img_file = self.rcnn_data[idx][str(i)]['file_name']
                shutil.copy(img_file, os.path.join(output_dir, f'{prefix}_{str(i)}.png'))
                os.chmod(os.path.join(output_dir, f'{prefix}_{str(i)}.png'), 0o755)

    def save_depth(self, idx, output_dir, prefix=''):
        pred_depth = self.get_pred_depth(idx)
        if pred_depth is None:
            return
        gt_depth = self.get_gt_depth(idx)
        for i in range(2):
            io.imsave(os.path.join(output_dir, f'{prefix}_{str(i)}_pred.png'), pred_depth[str(i)])
            io.imsave(os.path.join(output_dir, f'{prefix}_{str(i)}_gt.png'), gt_depth[str(i)])

    def save_pair_blended_mask(self, idx, gt_segm, output_dir, prefix=''):
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        for i in range(2):
            img_file = self.rcnn_data[idx][str(i)]['file_name']
            house_name, basename = self.rcnn_data[idx][str(i)]["image_id"].split('_', 1)
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)[:,:,::-1]
            height, width, _ = img.shape
            vis = Visualizer(img, self.metadata)
            if gt_segm:
                seg_blended = get_gt_labeled_seg(self.dataset_dict[key][str(i)], vis)

                # seg_blended = vis.draw_dataset_dict(self.dataset_dict[key][str(i)]).get_image()

                segmentations = [ann['segmentation'] for ann in self.dataset_dict[key][str(i)]['annotations']]
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, prefix + f'_{i}_gtseg.png'), seg_blended)
            else:
                p_instance = create_instances(self.rcnn_data[idx][str(i)]['instances'], img.shape[:2], 
                                                    pred_planes=self.rcnn_data[idx][str(i)]['pred_plane'].numpy(), 
                                                    conf_threshold=self.score_threshold)       
                seg_blended = get_labeled_seg(p_instance, self.score_threshold, vis)
                segmentations = p_instance.pred_masks
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, prefix + f'_{i}_predseg.png'), seg_blended)
                

    def save_pair_objects(self, idx, tran_topk, rot_topk, output_dir, prefix='', 
                        gt_plane=False, pred_camera=None, gt_segm=False, 
                        plane_param_override=None, show_camera=False, corr_list=[], reduce_size=True, 
                        exclude=None, webvis=False):
        """
        if tran_topk == -2 and rot_topk == -2, then pred_camera should not be None, this is used for non-binned camera.
        if exclude is not None, exclude some instances to make fig 2.
        idx=7867
        exclude = {
            '0': [2,3,4,5,6,7],
            '1': [0,1,2,4,5,6,7],
        }
        """
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        basenames = {}
        meshes_list = []
        #map_files = []
        uv_maps = []
        cam_list = []
        vis_idx = 0
        # get plane parameters
        plane_locals = {}
        p_instances = {}
        for i in range(2):
            if not gt_plane or not gt_segm:
                p_instances[str(i)] = create_instances(self.rcnn_data[idx][str(i)]['instances'], [self.dataset_dict[key][str(i)]['height'], self.dataset_dict[key][str(i)]['width']], 
                    pred_planes=self.rcnn_data[idx][str(i)]['pred_plane'].numpy(), 
                    conf_threshold=self.score_threshold)
            if gt_plane:
                plane_locals[str(i)] = [ann['plane'] for ann in self.dataset_dict[key][str(i)]['annotations']]
            else:
                if plane_param_override is None:
                    plane_locals[str(i)] = p_instances[str(i)].pred_planes
                else:
                    plane_locals[str(i)] = plane_param_override[str(i)]
        # get camera 1 to 2
        if pred_camera is None:
            if tran_topk == -1 and rot_topk == -1:
                camera1to2 = self.dataset_dict[key]['rel_pose']
                camera1to2['rotation'] = quaternion.from_float_array(camera1to2['rotation'])
                camera1to2['position'] = np.array(camera1to2['position'])
            else:
                camera1to2 = self.get_camera_info(idx, tran_topk, rot_topk)
                camera1to2 = {'position': np.array(camera1to2['position']), 'rotation': quaternion.from_float_array(camera1to2['rotation'])}
        
            
        else:
            assert(tran_topk == -2 and rot_topk == -2)
            camera1to2 = {'position': np.array(pred_camera['position']), 'rotation': quaternion.from_float_array(pred_camera['rotation'])}
        
        # Merge planes if they are in correspondence
        if len(corr_list) != 0:
            plane_locals = self.merge_plane_params_from_local_params(plane_locals, corr_list, camera1to2)

        os.makedirs(output_dir, exist_ok=True)
        for i in range(2):
            img_file = self.rcnn_data[idx][str(i)]['file_name']
            house_name, basename = self.rcnn_data[idx][str(i)]["image_id"].split('_', 1)
            basenames[str(i)] = basename
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)[:,:,::-1]

            # save original images
            # imageio.imwrite(os.path.join(output_dir, prefix + f'_view_{i}.png'), img)

            height, width, _ = img.shape
            vis = Visualizer(img, self.metadata)
            if i == 0:
                camera_info = camera1to2
            else:
                camera_info = {'position': np.array([0,0,0]), 'rotation': np.quaternion(1,0,0,0)}
            if not gt_plane or not gt_segm:
                p_instance = p_instances[str(i)]            
            plane_params = plane_locals[str(i)]
            if gt_segm:
                seg_blended = get_gt_labeled_seg(self.dataset_dict[key][str(i)], vis, paper_img=True)
                segmentations = [ann['segmentation'] for ann in self.dataset_dict[key][str(i)]['annotations']]
                # cv2.imwrite(os.path.join(output_dir, prefix + f'_{i}_gtseg.png'), seg_blended)
            else:
                seg_blended = get_labeled_seg(p_instance, self.score_threshold, vis, paper_img=True)
                segmentations = p_instance.pred_masks
                # cv2.imwrite(os.path.join(output_dir, prefix + f'_{i}_predseg.png'), seg_blended)
            if exclude is not None:
                plane_params_override = []
                segmentations_override = []
                for ann_idx in range(len(plane_params)):
                    if ann_idx not in exclude[str(i)]:
                        plane_params_override.append(plane_params[ann_idx])
                        segmentations_override.append(segmentations[ann_idx])
                plane_params = np.array(plane_params_override)
                segmentations = segmentations_override
            # if False:
                # valid = (np.linalg.norm(plane_params, axis=1))>0.2
                # plane_params = plane_params[valid]
                # segmentations = np.array(segmentations)[valid].tolist()
            meshes, uv_map = get_single_image_mesh_plane(plane_params, segmentations, img_file=img_file, 
                            height=height, width=width, webvis=False, tolerance=0)
            uv_maps.extend(uv_map)
            meshes = transform_meshes(meshes, camera_info)
            meshes_list.append(meshes)
            cam_list.append(camera_info)
            
        #pdb.set_trace()
        joint_mesh = join_meshes_as_batch(meshes_list)
        if webvis:
            joint_mesh = rotate_mesh_for_webview(joint_mesh)

        # add camera into the mesh
        if show_camera:
            cam_meshes = get_camera_meshes(cam_list)
            if webvis:
                cam_meshes = rotate_mesh_for_webview(cam_meshes)
        else:
            cam_meshes = None


        # save obj
        if gt_plane:
            if len(prefix) == 0:
                prefix = key+'_gt_plane'
            save_obj(folder=output_dir, prefix=prefix, meshes=joint_mesh, cam_meshes=cam_meshes, decimal_places=10, map_files=None, uv_maps=uv_maps)
        else:
            if len(prefix) == 0:
                prefix = key+'_pred'
            save_obj(folder=output_dir, prefix=prefix, meshes=joint_mesh, cam_meshes=cam_meshes, decimal_places=10, blend_flag=True, map_files=None, uv_maps=uv_maps)

    def save_matching(self, idx, assignment, output_dir, prefix='', fp=True, gt_box=True, gt_segm=False, paper_img=False, vertical=False):
        """
        fp: whether show fp or fn
        gt_box: whether use gtbox 
        """
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        blended = {}
        basenames = {}
        uniq_idx = 0
        # centroids for matching
        centroids = {'0':[],'1':[]}
        
        gt_corr_list = self.get_gt_affinity(idx, rtnformat='list', gtbox=gt_box)
        gt_corr_list = np.unique(gt_corr_list,axis=0)
        # assign mask color to each instance across views so that corresponding instances have the same color.
        num_instances = sum(assignment.shape) - len(gt_corr_list)
        if len(gt_corr_list) != 0:
            num_instances += (len(gt_corr_list[:,0]) - len(np.unique(gt_corr_list[:,0]))) + (len(gt_corr_list[:,1]) - len(np.unique(gt_corr_list[:,1]))) 
        colors = [random_color(rgb=True, maximum=1) for _ in range(int(num_instances))]
        common_instances_color = colors[:len(gt_corr_list)]
        uniq_instances_color = colors[len(gt_corr_list):]
        for i in range(2):
            img_file = self.rcnn_data[idx][str(i)]['file_name']
            house_name, basename = self.rcnn_data[idx][str(i)]["image_id"].split('_', 1)
            basenames[str(i)] = basename
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)[:,:,::-1]
            height, width, _ = img.shape
            vis = Visualizer(img, self.metadata)
            if gt_box and gt_segm:
                assigned_colors = []
                for ann_idx in range(len(self.dataset_dict[key][str(i)]['annotations'])):
                    if ann_idx in gt_corr_list[:,i]:
                        assigned_colors.append(common_instances_color[np.where(gt_corr_list[:,i]==ann_idx)[0][0]])
                    else:
                        assigned_colors.append(uniq_instances_color[uniq_idx])
                        uniq_idx += 1
                seg_blended = get_gt_labeled_seg(self.dataset_dict[key][str(i)], vis, assigned_colors=assigned_colors, paper_img=paper_img)
                # seg_blended = vis.draw_dataset_dict(self.dataset_dict[key][str(i)]).get_image()
                blended[str(i)] = seg_blended
                for ann in self.dataset_dict[key][str(i)]['annotations']:
                    M=center_of_mass(polygons_to_bitmask(ann['segmentation'], height, width))
                    centroids[str(i)].append(M[::-1]) # reverse for opencv
                centroids[str(i)] = np.array(centroids[str(i)])
            else:
                assigned_colors = []
                for ann_idx in range(len(self.rcnn_data[idx][str(i)]['instances'])):
                    if len(gt_corr_list)!=0 and ann_idx in gt_corr_list[:,i]:
                        assigned_colors.append(common_instances_color[np.where(gt_corr_list[:,i]==ann_idx)[0][0]])
                    else:
                        assigned_colors.append(uniq_instances_color[uniq_idx])
                        uniq_idx += 1
                p_instance = create_instances(self.rcnn_data[idx][str(i)]['instances'], img.shape[:2], 
                                                pred_planes=self.rcnn_data[idx][str(i)]['pred_plane'].numpy(), 
                                                conf_threshold=self.score_threshold)
                seg_blended = get_labeled_seg(p_instance, self.score_threshold, vis, assigned_colors=assigned_colors, paper_img=paper_img)
                blended[str(i)] = seg_blended
                # centroid of mask
                for ann in self.rcnn_data[idx][str(i)]['instances']:
                    M=center_of_mass(mask_util.decode(ann['segmentation']))
                    centroids[str(i)].append(M[::-1]) # reverse for opencv
                centroids[str(i)] = np.array(centroids[str(i)])

        pred_corr_list = np.array(torch.FloatTensor(assignment).nonzero().tolist())
        if gt_box:
            if not self.gt_box:
                # it is not applicable to draw predict correspondence using gtbox if prediction is not generated from gt boxes.
                pred_corr_list = self.get_gt_affinity(idx, rtnformat='list', gtbox=True)
        if len(prefix) == 0:
            tag = ''
            if fp:
                tag += 'fp'
            else:
                tag += 'fn'
            if gt_box:
                tag += '_gtbox'
            else:
                tag += '_predbox'
            if gt_segm:
                tag += '_gtsegm'
            else:
                tag += '_predsegm'
            prefix = f"match_{house_name}__{basenames['0']}__{basenames['1']}_tp{tag}.png"
        if gt_box:
            box1 = Boxes(torch.FloatTensor(self.rcnn_data[idx]['corrs']['0']['gt_bbox']))
            box2 = Boxes(torch.FloatTensor(self.rcnn_data[idx]['corrs']['1']['gt_bbox']))
        else:
            box1 = self.rcnn_data[idx]['corrs']['0']['embeddingbox']['pred_boxes']
            box2 = self.rcnn_data[idx]['corrs']['1']['embeddingbox']['pred_boxes']
        if not fp:
            correct_list_gt = []
            for pair in gt_corr_list:
                correct_list_gt.append(list(pair) in pred_corr_list.tolist())
            gt_matching_fig = draw_match(blended['0'], blended['1'], centroids['0'], centroids['1'],
                                        np.array(gt_corr_list), correct_list_gt, vertical=vertical, outlier_color=[0,176,240])
            os.makedirs(output_dir, exist_ok=True)
            gt_matching_fig.save(os.path.join(output_dir, prefix+'.png'))
        else:
            correct_list_pred = []
            for pair in pred_corr_list:
                correct_list_pred.append(list(pair) in gt_corr_list.tolist())
            pred_matching_fig = draw_match(blended['0'], blended['1'], centroids['0'], centroids['1'], 
                                            np.array(pred_corr_list), correct_list_pred, vertical=vertical)
            os.makedirs(output_dir, exist_ok=True)
            pred_matching_fig.save(os.path.join(output_dir, prefix+'.png'))
        # cv2.imwrite(os.path.join(output_dir, prefix+'_segm_0.png'), blended['0'][:,:,::-1])
        # cv2.imwrite(os.path.join(output_dir, prefix+'_segm_1.png'), blended['1'][:,:,::-1])
        



    def save_distance_matrix_by_different_types(self, distance_type, idx, tran_topk, rot_topk, assignment, output_dir):
        house_name = self.rcnn_data[idx]['0']["image_id"].split('_', 1)[0]
        os.makedirs( os.path.join(output_dir, house_name), exist_ok=True)

        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        gt_assignment = self.get_gt_affinity(idx, gtbox=False)
        if 'embedding' in distance_type:
            dis_mat = self.get_emb_distance_matrix(idx)
            save_distance_matrix(dis_mat, assignment, gt_assignment, os.path.join(output_dir, house_name, key+'_embedding.png'))
        if 'geo' in distance_type:
            dis_mat = self.get_geo_distance_matrix(idx, tran_topk, rot_topk)
            save_distance_matrix(dis_mat['offset'][0], assignment, gt_assignment, os.path.join(output_dir, house_name, key+'_offset.png'))
            save_distance_matrix(dis_mat['normal'][0], assignment, gt_assignment, os.path.join(output_dir, house_name, key+'_normal.png'))
            save_distance_matrix(dis_mat['l2'][0], assignment, gt_assignment, os.path.join(output_dir, house_name, key+'_l2.png'))
        


    def objective(self, assignment, idx, tran_topk, rot_topk, embedding_matrix=None, num_nomatch = 0, weight=None):
        if embedding_matrix is None:
            embedding_matrix = self.get_emb_distance_matrix(idx)
        geo_matrix = self.get_geo_distance_matrix(idx, tran_topk, rot_topk)
        # emb distance
        l_emb = (embedding_matrix * assignment).mean()
        # geo distance
        l_geo_l2 = (geo_matrix['l2'][0] * assignment).mean()
        l_geo_normal = (geo_matrix['normal'][0] * assignment).mean()
        l_geo_offset = (geo_matrix['offset'][0] * assignment).mean()
        cam_info = self.get_camera_info(idx, tran_topk, rot_topk)
        l_cam = - cam_info['position_prob'] - cam_info['rotation_prob']
        cost = weight['lambda_emb'] * l_emb + \
               weight['lambda_geo_l2']* l_geo_l2 + \
               weight['lambda_geo_normal']* l_geo_normal + \
               weight['lambda_geo_offset']* l_geo_offset + \
               weight['lambda_cam_rotation']* (- cam_info['rotation_prob']) + \
               weight['lambda_cam_translation']* (- cam_info['position_prob']) + \
               weight['lambda_nomatch']* num_nomatch
        return cost
        
        # Rotation 5
        # Translation 1
        # affinity 5

        # return l_emb + l_geo_normal + l_cam

    def optimize_by_idx(self, idx, maxk=5, weight=None):
        """
        single matrix
        """
        min_cost = np.Inf
        best_assignment = None
        best_camera = None
        best_distance_m = None
        best_tran_topk = 0
        best_rot_topk = 0
        # get assignment based on embedding matrix
        embedding_matrix = self.get_emb_distance_matrix(idx)
        if 'assignment' not in weight.keys() or weight['assignment'] == 'proposal':
            matching_proposals = self.get_assignment(idx, embedding_matrix, 'proposal', weight)
            score_weight = np.array(weight['score_weight']).reshape(-1,1)            
            maxk = weight['maxk']
            best_score = np.NINF
            for tran_topk in range(maxk):
                for rot_topk in range(maxk):
                    geo_matrix = self.get_geo_distance_matrix(idx, tran_topk, rot_topk) #key: l2, offset, normal
                    l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
                    normal_matrix = geo_matrix['normal'] / np.pi
                    offset_matrix = np.clip(geo_matrix['offset'], 0, weight['offset_clamp']) / weight['offset_clamp']
                    pred_cam = self.get_camera_info(idx, tran_topk, rot_topk)
                    for matching, num_nomatch, scores in matching_proposals:
                        assignment = self.assignment_list_to_matrix(matching, embedding_matrix.shape)
                        # x = np.array([
                        #             assignment.sum(), 
                        #             np.log(pred_cam['position_prob']), 
                        #             np.log(pred_cam['rotation_prob']), 
                        #             (embedding_matrix*assignment).numpy().mean(),
                        #             (l2_matrix*assignment).numpy().mean(), 
                        #             (normal_matrix*assignment).numpy().mean(), 
                        #             (offset_matrix*assignment).numpy().mean(),])
                        distance_matrix = weight['lambda_emb'] * embedding_matrix + \
                            weight['lambda_geo_l2'] * l2_matrix + \
                            weight['lambda_geo_normal'] * normal_matrix + \
                            weight['lambda_geo_offset'] * offset_matrix
                        x = np.array([
                            assignment.sum(), 
                            np.log(pred_cam['position_prob']), 
                            np.log(pred_cam['rotation_prob']), 
                            (distance_matrix*assignment).numpy().mean(),
                        ])
                        score = x@score_weight
                        if score > best_score:
                            best_score = score
                            best_assignment = assignment
                            best_camera = self.get_camera_info(idx, tran_topk, rot_topk)
                            best_tran_topk = tran_topk
                            best_rot_topk = rot_topk
        elif weight['assignment'] == 'km':
            assignment = self.get_assignment(idx, embedding_matrix, method='km', weight=weight)
            for tran_topk in range(weight['max_tran_topk']):
                for rot_topk in range(weight['max_rot_topk']):
                    score = self.objective(assignment, idx, tran_topk, rot_topk, num_nomatch=0, weight=weight)
                    if score < min_cost:
                        min_cost = score
                        best_assignment = assignment
                        best_camera = self.get_camera_info(idx, tran_topk, rot_topk)
                        best_tran_topk = tran_topk
                        best_rot_topk = rot_topk
            if 'debug_dir' in weight.keys():
                os.makedirs(weight['debug_dir'], exist_ok=True)
                self.save_distance_matrix_by_different_types(['geo', 'embedding'], idx, 
                best_tran_topk, best_rot_topk, best_assignment, weight['debug_dir'])
                self.save_pair_objects(idx, best_tran_topk, best_rot_topk, weight['debug_dir'])
                print("best_score", min_cost)

                os.makedirs(os.path.join(weight['debug_dir'], 'gt_cam'), exist_ok=True)
                score = self.objective(best_assignment, idx, -1, -1, num_nomatch=0, weight=weight)
                print("gt cam score", score)
                self.save_distance_matrix_by_different_types(['geo', 'embedding'], idx, 
                -1, -1, best_assignment, os.path.join(weight['debug_dir'], 'gt_cam'))
                self.save_pair_objects(idx, -1, -1, os.path.join(weight['debug_dir'], 'gt_cam'))
        elif weight['assignment'] == 'km_gt_cam' or weight['assignment'] == 'km_gt_cam_gt_plane' or weight['assignment'] == 'km_gt_cambin':
            gt_cam = self.get_gt_cam_topk(idx)
            if 'km_gt_cambin' in weight['assignment']:
                geo_matrix = self.get_geo_distance_matrix(idx, gt_cam['gt_tran_topk'], gt_cam['gt_rot_topk']) #key: l2, offset, normal
            else:
                geo_matrix = self.get_geo_distance_matrix(idx, -1, -1) #key: l2, offset, normal
            # l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
            normal_matrix = geo_matrix['normal'] / np.pi
            offset_matrix = np.clip(geo_matrix['offset'], 0, weight['offset_clamp']) / weight['offset_clamp']
            
            distance_matrix = weight['lambda_emb'] * embedding_matrix + \
                weight['lambda_geo_normal'] * normal_matrix + \
                weight['lambda_geo_offset'] * offset_matrix
                # weight['lambda_geo_l2'] * l2_matrix + \

            assignment = self.get_assignment(idx, distance_matrix[0], method='km', weight=weight)
            best_assignment = assignment
            best_camera = self.get_camera_info(idx, gt_cam['gt_tran_topk'], gt_cam['gt_rot_topk'])
            best_tran_topk= gt_cam['gt_tran_topk']
            best_rot_topk= gt_cam['gt_rot_topk']

            if 'debug_dir' in weight.keys():
                os.makedirs(weight['debug_dir'], exist_ok=True)
                self.save_distance_matrix_by_different_types(['geo', 'embedding'], idx, 
                best_tran_topk, best_rot_topk, best_assignment, weight['debug_dir'])
                self.save_pair_objects(idx, best_tran_topk, best_rot_topk, weight['debug_dir'])
                print("best_score", min_cost)

                os.makedirs(os.path.join(weight['debug_dir'], 'gt_cam'), exist_ok=True)
                score = self.objective(best_assignment, idx, -1, -1, num_nomatch=0, weight=weight)
                print("gt cam score", score)
                self.save_distance_matrix_by_different_types(['geo', 'embedding'], idx, 
                -1, -1, best_assignment, os.path.join(weight['debug_dir'], 'gt_cam'))
                self.save_pair_objects(idx, -1, -1, os.path.join(weight['debug_dir'], 'gt_cam'))
        elif weight['assignment'] == 'km_gt_cam_separate':
            gt_cam = self.get_gt_cam_topk(idx)
            if 'km_gt_cambin' in weight['assignment']:
                geo_matrix = self.get_geo_distance_matrix(idx, gt_cam['gt_tran_topk'], gt_cam['gt_rot_topk']) #key: l2, offset, normal
            else:
                geo_matrix = self.get_geo_distance_matrix(idx, -1, -1) #key: l2, offset, normal
            l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
            assignment_emb = self.get_assignment(idx, embedding_matrix, method='km', weight={'threshold': weight['threshold_emb']})
            assignment_l2 = self.get_assignment(idx, l2_matrix[0], method='km', weight={'threshold': weight['threshold_l2']})
            assignment = np.logical_and(assignment_emb, assignment_l2)
            best_assignment = assignment
            best_camera = self.get_camera_info(idx, gt_cam['gt_tran_topk'], gt_cam['gt_rot_topk'])
            best_tran_topk= gt_cam['gt_tran_topk']
            best_rot_topk= gt_cam['gt_rot_topk']
        elif weight['assignment'] == 'search_camera':
            best_score = 0
            best_camera = self.get_camera_info(idx, 0, 0)
            best_assignment = None
            best_tran_topk= 0
            best_rot_topk= 0
            for k_tran in range(weight['topk_tran']):
                for k_rot in range(weight['topk_rot']):
                    pred_cam = self.get_camera_info(idx, k_tran, k_rot)
                    
                    geo_matrix = self.get_geo_distance_matrix(idx, k_tran, k_rot) #key: l2, offset, normal
                    l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
                    normal_matrix = geo_matrix['normal'] / np.pi
                    offset_matrix = np.clip(geo_matrix['offset'], 0, weight['offset_clamp']) / weight['offset_clamp']
            
                    distance_matrix = weight['lambda_emb'] * embedding_matrix + \
                        weight['lambda_geo_l2'] * l2_matrix + \
                        weight['lambda_geo_normal'] * normal_matrix + \
                        weight['lambda_geo_offset'] * offset_matrix

                    assignment = self.get_assignment(idx, distance_matrix[0], method='km', weight=weight)
                    score = assignment.sum()
                    if best_assignment is None:
                        best_assignment = assignment
                    if score > best_score:
                        best_score = score
                        best_camera = pred_cam
                        best_assignment = assignment
                        best_tran_topk= k_tran
                        best_rot_topk= k_rot
        elif weight['assignment'] == 'save_all':
            rtns = {}
            gt_cam_topk = self.get_gt_cam_topk(idx)
            xs = []
            ys = []
            misc = []
            for k_tran in range(weight['topk_tran']):
                for k_rot in range(weight['topk_rot']):
                    pred_cam = self.get_camera_info(idx, k_tran, k_rot)
                    
                    geo_matrix = self.get_geo_distance_matrix(idx, k_tran, k_rot) #key: l2, offset, normal
                    l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
                    normal_matrix = geo_matrix['normal'] / np.pi
                    offset_matrix = np.clip(geo_matrix['offset'], 0, weight['offset_clamp']) / weight['offset_clamp']
            
                    distance_matrix = weight['lambda_emb'] * embedding_matrix + \
                        weight['lambda_geo_l2'] * l2_matrix + \
                        weight['lambda_geo_normal'] * normal_matrix + \
                        weight['lambda_geo_offset'] * offset_matrix

                    assignment = self.get_assignment(idx, distance_matrix[0], method='km', weight=weight)
                    xs.append([assignment.sum(), pred_cam['position_prob'], pred_cam['rotation_prob'], (embedding_matrix*assignment).numpy().mean(),
                         (l2_matrix*assignment).numpy().mean(), (normal_matrix*assignment).numpy().mean(), (offset_matrix*assignment).numpy().mean(),])
                    if k_tran == gt_cam_topk['gt_tran_topk'] and k_rot == gt_cam_topk['gt_rot_topk']:
                        ys.append(1)
                    else:
                        ys.append(0)
                    misc.append([k_tran, k_rot])
            rtn = {'idx': idx, 'xs': np.array(xs), 'ys': np.array(ys), 'misc': np.array(misc)}
            return rtn
        elif weight['assignment'] == 'km_search_cam':
            best_score = np.NINF
            best_assignment = None
            best_camera = None
            best_tran_topk = None
            best_rot_topk = None
            best_distance_m = None
            score_weight = np.array(weight['score_weight']).reshape(-1,1)
            for k_tran in range(weight['topk_tran']):
                for k_rot in range(weight['topk_rot']):
                    pred_cam = self.get_camera_info(idx, k_tran, k_rot)
                    
                    geo_matrix = self.get_geo_distance_matrix(idx, k_tran, k_rot) #key: l2, offset, normal
                    l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
                    normal_matrix = geo_matrix['normal'] / np.pi
                    offset_matrix = np.clip(geo_matrix['offset'], 0, weight['offset_clamp']) / weight['offset_clamp']
            
                    distance_matrix = weight['lambda_emb'] * embedding_matrix + \
                        weight['lambda_geo_l2'] * l2_matrix + \
                        weight['lambda_geo_normal'] * normal_matrix + \
                        weight['lambda_geo_offset'] * offset_matrix

                    assignment = self.get_assignment(idx, distance_matrix[0], method='km', weight=weight)
                    # x = np.array([
                    #         assignment.sum(), 
                    #         np.log(pred_cam['position_prob']), 
                    #         np.log(pred_cam['rotation_prob']), 
                    #         (embedding_matrix*assignment).numpy().mean(),
                    #         (l2_matrix*assignment).numpy().mean(), 
                    #         (normal_matrix*assignment).numpy().mean(), 
                    #         (offset_matrix*assignment).numpy().mean(),])
                    x = np.array([
                        assignment.sum(), 
                        np.log(pred_cam['position_prob']), 
                        np.log(pred_cam['rotation_prob']), 
                        (distance_matrix*assignment).numpy().mean(),
                    ])
                    score = x@score_weight
                    if score > best_score:
                        best_score = score
                        best_assignment = assignment
                        best_distance_m = distance_matrix
                        best_camera = pred_cam
                        best_tran_topk = k_tran
                        best_rot_topk = k_rot
        elif weight['assignment'] == 'gtcorr_gtplane_solve_cam' or weight['assignment'] == 'gtcorr_predplane_solve_cam':
            assert(self.gt_box)
            if 'gtcorr' in weight['assignment']:
                assignment = np.array(self.get_gt_affinity(idx, rtnformat='list'))
            else:
                raise NotImplementedError

            
            if 'gtplane' in weight['assignment']:
                assert(self.gt_box)
                gt_planes = self.get_gt_plane_by_idx(idx)
                x1 = np.array(gt_planes['0'])[assignment[:,0]] * np.array([1,-1,-1])
                x2 = np.array(gt_planes['1'])[assignment[:,1]] * np.array([1,-1,-1])
            elif 'predplane' in weight['assignment']:
                x1 = np.array(self.rcnn_data[idx]['0']['pred_plane'])[assignment[:,0]] * np.array([1,-1,-1])
                x2 = np.array(self.rcnn_data[idx]['1']['pred_plane'])[assignment[:,1]] * np.array([1,-1,-1])
            else:
                raise NotImplementedError
            x1 = x1.T
            x2 = x2.T

            # Solve for R
            S = x1@x2.T
            u, s, vh = np.linalg.svd(S)
            R = vh.T@u.T
            if np.linalg.det(R) < 0:
                R[-1,-1] = -R[-1,-1]
            pred_R = quaternion.as_float_array(quaternion.from_rotation_matrix(R))
            gt_R = quaternion.from_float_array(self.get_camera_info(idx, -1,-1)['rotation'])
            angle_err = angle_error_rot_vec(pred_R, quaternion.as_float_array(gt_R))
            
            # Solve for T
            Rx = (R@x1).T           
            Rx_norm = np.linalg.norm(Rx, axis=1)
            x2_norm = np.linalg.norm(x2.T, axis=1)
            TdotRx = (x2_norm / Rx_norm - 1)*(Rx_norm**2)
            pred_T = np.linalg.inv(Rx.T@Rx)@Rx.T@TdotRx
            gt_T = np.array(self.get_camera_info(idx, -1,-1)['position'])
            tran_err = np.linalg.norm(pred_T - gt_T)
            rtn = {'idx': idx, 'n_corr': len(assignment), 'angle_err': angle_err, 'tran_err': tran_err, 
                'rank': np.linalg.matrix_rank(Rx), 'best_camera': {'position': pred_T, 'rotation': pred_R}}
            return rtn
        elif weight['assignment'] == 'non_linear_optimize':
            def so3ToVec6d(so3):
                return np.array(so3).T.flatten()[:6]

            def vec6dToSo3(vec6d):
                assert(len(vec6d)==6)
                a1 = np.array(vec6d[:3])
                a2 = np.array(vec6d[3:])
                b1 = a1 / np.max([np.linalg.norm(a1), 1e-8])
                b2 = a2 - np.dot(b1, a2)*b1
                b2 = b2 / np.max([np.linalg.norm(b2), 1e-8])
                b3 = np.cross(b1, b2)
                return np.vstack((b1, b2, b3)).T
            
            def quaternion_from_array(float_array):
                assert(len(float_array) == 4)
                float_array = np.array(float_array)
                q = float_array / (np.linalg.norm(float_array) + 1e-5)
                return quaternion.from_float_array(q)
            def rotation_matrix_from_array(float_array):
                q = quaternion_from_array(float_array)
                R = quaternion.as_rotation_matrix(q)
                return R
            def project(R, T, x):
                Rx = R@x
                Rx_norm = np.linalg.norm(Rx, axis=0)
                return (np.dot(T, Rx) / (Rx_norm**2) + 1) * Rx

            def angle_error_mat(R1, R2):
                cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
                cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
                return np.abs(np.arccos(cos))

            def fun(x0, x1, x2):
                assert(len(x0)==6+3)
                R = vec6dToSo3(x0[:6]) 
                # R = rotation_matrix_from_array(x0[:4])
                T = np.array(x0[6:9])
                err = huber(weight['huber_delta'], np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum()
                # err = (np.linalg.norm(project(R, T, x1) - x2, axis=0)**2).sum()
                return err
            
            def fun_with_init_cam(x0, x1, x2, x0_init):
                R = rotation_matrix_from_array(x0[:4])
                T = np.array(x0[4:])
                change_R = angle_error_rot_vec(np.array(x0_init[:4]), np.array(x0[:4])) / 180 * np.pi
                change_T = np.linalg.norm(np.array(x0_init[4:]) - T)
                lambda_R = 100
                # lambda_T = 1
                err = huber(0.01, np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum() + lambda_R * change_R
                # err = huber(0.01, np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum() + lambda_R * huber(1, change_R) #+ huber(1, change_T)
                return err

            def fun_with_online_sift(x0, numPlane, img1, boxes1, segms1, normals1, img2, boxes2, segms2, normals2):
                R = rotation_matrix_from_array(x0[:4])
                T = np.array(x0[4:7])
                offsets1 = x0[7:7+numPlane].reshape(-1,1)
                offsets2 = x0[7+numPlane:7+2*numPlane].reshape(-1,1)
                planes1_suncg = offsets1 * normals1
                planes2_suncg = offsets2 * normals2
                planes1_habitat = (planes1_suncg * np.array([1., -1., -1.])).T
                planes2_habitat = (planes2_suncg * np.array([1., -1., -1.])).T
                
                err_plane = huber(weight['huber_delta'], np.linalg.norm(project(R, T, planes1_habitat) - planes2_habitat, axis=0)).sum()
                plane_corr = np.vstack((np.arange(len(boxes1)), np.arange(len(boxes1)))).T # Already matched
                plot_pixel_matches(img1, boxes1, segms1, planes1_suncg,
                                    img2, boxes2, segms2, planes2_suncg, 
                                    plane_corr)
                err_pixel = get_pixel_error(img1, boxes1, segms1, planes1_suncg,
                                            img2, boxes2, segms2, planes2_suncg,
                                            R, T, plane_corr)
                err = err_plane + err_pixel
                # err = huber(0.01, np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum() + lambda_R * huber(1, change_R) #+ huber(1, change_T)
                return err

            def fun_with_precalculated_sift(x0, numPlane, img1, xys1, normals1, img2, xys2, normals2):
                assert(numPlane == len(xys1))
                assert(numPlane == len(xys2))
                assert(numPlane == len(normals1))
                assert(numPlane == len(normals2))
                R = vec6dToSo3(x0[:6]) 
                T = np.array(x0[6:9])
                offsets1 = x0[9:9+numPlane].reshape(-1,1)
                offsets2 = x0[9+numPlane:9+2*numPlane].reshape(-1,1)
                planes1_suncg = offsets1 * normals1
                planes2_suncg = offsets2 * normals2
                planes1_habitat = (planes1_suncg * np.array([1., -1., -1.])).T
                planes2_habitat = (planes2_suncg * np.array([1., -1., -1.])).T
                
                err_plane = huber(weight['huber_delta'], np.linalg.norm(project(R, T, planes1_habitat) - planes2_habitat, axis=0)).sum()
                err_pixel = get_pixel_error_precalculated_sift(
                    img1, xys1, planes1_suncg,
                    img2, xys2, planes2_suncg,
                    R, T
                )
                err = err_plane + err_pixel
                # err = huber(0.01, np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum() + lambda_R * huber(1, change_R) #+ huber(1, change_T)
                return err

            def fun_with_precalculated_sift_reduce_rot(x0, numPlane, img1, xys1, normals1, img2, xys2, normals2, init_R):
                assert(numPlane == len(xys1))
                assert(numPlane == len(xys2))
                assert(numPlane == len(normals1))
                assert(numPlane == len(normals2))
                R = vec6dToSo3(x0[:6]) 
                T = np.array(x0[6:9])
                offsets1 = x0[9:9+numPlane].reshape(-1,1)
                offsets2 = x0[9+numPlane:9+2*numPlane].reshape(-1,1)
                planes1_suncg = offsets1 * normals1
                planes2_suncg = offsets2 * normals2
                planes1_habitat = (planes1_suncg * np.array([1., -1., -1.])).T
                planes2_habitat = (planes2_suncg * np.array([1., -1., -1.])).T
                
                err_plane = huber(weight['huber_delta'], np.linalg.norm(project(R, T, planes1_habitat) - planes2_habitat, axis=0)).sum()
                err_pixel = get_pixel_error_precalculated_sift(
                    img1, xys1, planes1_suncg,
                    img2, xys2, planes2_suncg,
                    R, T
                )
                change_R = angle_error_mat(R, init_R)
                err = err_plane + err_pixel + change_R*weight['lambda_R']
                # err = huber(0.01, np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum() + lambda_R * huber(1, change_R) #+ huber(1, change_T)
                return err
            """
            def fun_with_precalculated_sift_constrain_rot(x0, numPlane, img1, xys1, normals1, img2, xys2, normals2, rot):
                assert(numPlane == len(xys1))
                assert(numPlane == len(xys2))
                assert(numPlane == len(normals1))
                assert(numPlane == len(normals2))
                R = rotation_matrix_from_array(x0[0]*np.array([0,5.144554016,1]))@rot
                T = np.array(x0[1:4])
                offsets1 = x0[4:4+numPlane].reshape(-1,1)
                offsets2 = x0[4+numPlane:4+2*numPlane].reshape(-1,1)
                planes1_suncg = offsets1 * normals1
                planes2_suncg = offsets2 * normals2
                planes1_habitat = (planes1_suncg * np.array([1., -1., -1.])).T
                planes2_habitat = (planes2_suncg * np.array([1., -1., -1.])).T
                
                err_plane = huber(weight['huber_delta'], np.linalg.norm(project(R, T, planes1_habitat) - planes2_habitat, axis=0)).sum()
                plane_corr = np.vstack((np.arange(len(boxes1)), np.arange(len(boxes1)))).T # Already matched
                err_pixel = get_pixel_error_precalculated_sift(
                    img1, xys1, planes1_suncg,
                    img2, xys2, planes2_suncg,
                    R, T
                )
                err = err_plane + err_pixel
                # err = huber(0.01, np.linalg.norm(project(R, T, x1) - x2, axis=0)).sum() + lambda_R * huber(1, change_R) #+ huber(1, change_T)
                return err
            """
            """
            Load optimized dict
            """
            if self.optimized_dict is None:
                with open(weight['optimized_dict_path'], 'rb') as f:
                    self.optimized_dict = pickle.load(f)
            """
            Initialize camera pose
            """
            if weight['init_cam'] == 'gt':
                # gt camera
                init_R = self.get_camera_info(idx, -1,-1)['rotation']
                init_T = np.array(self.get_camera_info(idx, -1,-1)['position'])
            elif weight['init_cam'] == 'top1':
                init_R = self.get_camera_info(idx, 0, 0)['rotation']
                init_T = np.array(self.get_camera_info(idx, 0, 0)['position'])
            elif weight['init_cam'] == 'best_optimized':
                init_R = self.optimized_dict[idx]['best_camera']['rotation']
                init_T = self.optimized_dict[idx]['best_camera']['position']
            else:
                raise NotImplementedError
            x0 = np.concatenate((so3ToVec6d(rotation_matrix_from_array(init_R)), init_T))
            """
            Select correspondence assignment
            """
            if weight['corr_selection'] == 'gt':
                # gt assignment
                assignment = np.array(self.get_gt_affinity(idx, rtnformat='list', gtbox=False))
            elif weight['corr_selection'] == 'km':
                # pred assignment
                assignment_m = self.get_assignment(idx, embedding_matrix, method='km', weight={'threshold':0.7})
                assignment = np.argwhere(assignment_m)
            elif weight['corr_selection'] == 'best_optimized':
                assignment_m = self.optimized_dict[idx]['best_assignment']
                assignment = np.argwhere(assignment_m)
            else:
                raise NotImplementedError
            if len(assignment) == 0:
                rtn = {'idx': idx, 'n_corr': len(assignment), 'cost':0,  
                   'best_camera': {'position': init_T, 'rotation': init_R},
                   'best_assignment': assignment_m}
                return rtn
            """
            Select plane params
            """
            if weight['plane_selection'] == 'gt':
                assert(self.gt_box)
                gt_planes = self.get_gt_plane_by_idx(idx)
                x1_full = np.array(gt_planes['0'])
                x2_full = np.array(gt_planes['1'])
            elif weight['plane_selection'] == 'pred':
                x1_full = np.array(self.rcnn_data[idx]['0']['pred_plane'])
                x2_full = np.array(self.rcnn_data[idx]['1']['pred_plane'])
            else:
                raise NotImplementedError

            x1 = x1_full[assignment[:,0]]
            x2 = x2_full[assignment[:,1]]

            """
            Select optimized function
            """
            if weight['target_fn'] == 'fun':
                x1 = (x1 * np.array([1,-1,-1])).T # suncg2habitat
                x2 = (x2 * np.array([1,-1,-1])).T
                rst = least_squares(fun, x0, args=(x1, x2))
            elif weight['target_fn'] == 'fun_with_init_cam':
                x1 = (x1 * np.array([1,-1,-1])).T # suncg2habitat
                x2 = (x2 * np.array([1,-1,-1])).T
                rst = least_squares(fun_with_init_cam, x0, args=(x1, x2, x0.copy()))
            elif weight['target_fn'] == 'fun_with_sift':
                boxes1 = np.array([inst['bbox'] for inst in self.rcnn_data[idx]['0']['instances']])[assignment[:,0]]
                boxes2 = np.array([inst['bbox'] for inst in self.rcnn_data[idx]['1']['instances']])[assignment[:,1]]
                segms1 = np.array([inst['segmentation'] for inst in self.rcnn_data[idx]['0']['instances']])[assignment[:,0]]
                segms2 = np.array([inst['segmentation'] for inst in self.rcnn_data[idx]['1']['instances']])[assignment[:,1]]
                offsets1 = np.linalg.norm(x1, axis=1)
                normals1 = x1 / (offsets1.reshape(-1,1)+1e-5)
                offsets2 = np.linalg.norm(x2, axis=1)
                normals2 = x2 / (offsets2.reshape(-1,1)+1e-5)
                
                x0 = np.concatenate((x0, offsets1, offsets2))
                
                img_file1 = self.rcnn_data[idx]['0']['file_name']
                img1 = cv2.imread(img_file1, cv2.IMREAD_COLOR)[:,:,::-1]
                img_file2 = self.rcnn_data[idx]['1']['file_name']
                img2 = cv2.imread(img_file2, cv2.IMREAD_COLOR)[:,:,::-1]
                xys1, xys2 = [], []
                for i in range(len(boxes1)):
                    try:
                        xy1, xy2 = get_pixel_matching(img1, boxes1[i], segms1[i], x1[i], 
                                                    img2, boxes2[i], segms2[i], x2[i])
                    except:
                        print(idx)
                        xy1 = []
                        xy2 = []
                    xys1.append(np.array(xy1))
                    xys2.append(np.array(xy2))
                """
                draw matching
                """
                # xys1_draw = [xy for xy in xys1 if len(xy) != 0]
                # xys2_draw = [xy for xy in xys2 if len(xy) != 0]
                # if len(xys1_draw) != 0 and len(xys2_draw) != 0:
                #     draw_matches_xy(img1, np.vstack(xys1_draw), 
                #                     img2, np.vstack(xys2_draw),
                #                     save_path='/Pool1/users/jinlinyi/public_html/planeRCNN_detectron2/e17_manual_optimize/v13_nonlinear_optimize_sift/pixel_match',
                #                     prefix=str(idx))
                # else:
                #     draw_matches_xy(img1, [], 
                #                     img2, [],
                #                     save_path='/Pool1/users/jinlinyi/public_html/planeRCNN_detectron2/e17_manual_optimize/v13_nonlinear_optimize_sift/pixel_match',
                #                     prefix=str(idx))
                    
     
                # import pdb; pdb.set_trace()
                rst = least_squares(fun_with_precalculated_sift_reduce_rot, x0, args=(len(boxes1),  
                    img1, xys1, normals1, 
                    img2, xys2, normals2, 
                    rotation_matrix_from_array(init_R)))
                
                # rst = least_squares(fun_with_precalculated_sift_constrain_rot, x0, args=(len(boxes1),  
                #     img1, xys1, normals1, 
                #     img2, xys2, normals2))

                offsets1 = rst.x[9:9+len(boxes1)]
                offsets2 = rst.x[9+len(boxes1):9+len(boxes1)*2]
                x1_full[assignment[:,0]] = offsets1.reshape(-1,1) * normals1
                x2_full[assignment[:,1]] = offsets2.reshape(-1,1) * normals2

            else:
                raise NotImplementedError
            # pred_R = quaternion.as_float_array(quaternion_from_array(rst.x[:4]))
            pred_R = quaternion.as_float_array(quaternion.from_rotation_matrix(vec6dToSo3(rst.x[:6])))
            pred_T = rst.x[6:9]
            rtn = {'idx': idx, 'n_corr': len(assignment), 'cost':rst.cost, 
                   'best_camera': {'position': pred_T, 'rotation': pred_R},
                   'best_assignment': assignment_m, 
                   'plane_param_override': {'0':x1_full, '1': x2_full},
            }
            return rtn

        else:
            raise NotImplementedError
        rtn = {'idx': idx, 'best_camera': best_camera, 'best_assignment': best_assignment, 'distance_m': best_distance_m,
                'best_tran_topk': best_tran_topk, 'best_rot_topk': best_rot_topk}
        return rtn

    def assignment_list_to_matrix(self, matching, size):
        rtn = np.zeros(size)
        for i, j in enumerate(matching):
            if j != -1:
                rtn[i, j] = 1
        return rtn

    def optimize_by_list(self, idxs, return_dict=None, maxk=5, weight=None):
        for idx in idxs:
            rtn = self.optimize_by_idx(idx, maxk, weight)
            return_dict[idx] = rtn

    def get_baseline(self, idx_list=None, method='hungarian', threshold=0.5):
        if idx_list is None:
            idx_list = range(len(self.rcnn_data))
        return_dict = {}
        for idx in idx_list:
            distance_matrix = self.get_emb_distance_matrix(idx)
            rtn = { 'idx': idx, 
                    'best_camera': self.get_camera_info(idx, 0, 0), 
                    'best_assignment': self.get_assignment(idx, distance_matrix, method=method, weight={'threshold': threshold}), 
                    'distance_m': distance_matrix,
                    'best_tran_topk': 0, 
                    'best_rot_topk': 0}
            return_dict[idx] = rtn
        return return_dict


    def save_dict(self, return_dict, folder, prefix=None):
        os.makedirs(folder, exist_ok=True)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if prefix is None:
            save_path = os.path.join(folder, f'optimized_{timestr}.pkl')
        else:
            save_path = os.path.join(folder, prefix + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(return_dict.copy(), f)

    def save_json(self, return_dict, folder, prefix=None):
        os.makedirs(folder, exist_ok=True)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if prefix is None:
            save_path = os.path.join(folder, f'optimized_{timestr}.json')
        else:
            save_path = os.path.join(folder, prefix + '.json')
        with open(save_path, 'w') as f:
            json.dump(return_dict.copy(), f)

    def rcnnidx2datasetkey(self, idx):
        key0 = self.rcnn_data[idx]['0']['image_id']
        key1 = self.rcnn_data[idx]['1']['image_id']
        key = key0 + '__' + key1
        return key


    def get_single2merge(self, idx, pred_corr=None, gt_corr=None):
        key = self.rcnnidx2datasetkey(idx)
        # GT merged mapping
        entry2gt_single_view = {}
        gt_single_view2entry = {'0':{}, '1':{}}
        if gt_corr is not None:
            gt_entry_id = 0
            for i in range(2):
                single_gt_idx = len(self.dataset_dict[key][str(i)]['annotations'])
                for s_i in range(single_gt_idx):
                    if s_i not in gt_corr[:,i]:
                        entry2gt_single_view[gt_entry_id] = {'pair':str(i), 'ann_id': s_i, 'merged': False}
                        gt_single_view2entry[str(i)][s_i] = gt_entry_id
                        gt_entry_id += 1
            for pair in gt_corr:
                entry2gt_single_view[gt_entry_id] = {'pair':['0', '1'], 'ann_id': pair, 'merged': True}
                gt_single_view2entry['0'][pair[0]] = gt_entry_id
                gt_single_view2entry['1'][pair[1]] = gt_entry_id
                gt_entry_id += 1

        # Pred merged mapping
        entry2pred_single_view = {}
        pred_single_view2entry = {'0':{}, '1':{}}
        if pred_corr is not None:
            pred_entry_id = 0
            for i in range(2):
                single_idx = len(self.rcnn_data[idx][str(i)]['pred_plane'])
                for s_i in range(single_idx):
                    if len(pred_corr)==0 or s_i not in pred_corr[:,i]:
                        entry2pred_single_view[pred_entry_id] = {'pair':str(i), 'ann_id': s_i, 'merged':False}
                        pred_single_view2entry[str(i)][s_i] = pred_entry_id
                        pred_entry_id += 1
            for pair in pred_corr:
                entry2pred_single_view[pred_entry_id] = {'pair':['0', '1'], 'ann_id': pair, 'merged': True}
                pred_single_view2entry['0'][pair[0]] = pred_entry_id
                pred_single_view2entry['1'][pair[1]] = pred_entry_id
                pred_entry_id += 1
        return {
            'entry2gt_single_view': entry2gt_single_view,
            'gt_single_view2entry': gt_single_view2entry,
            'entry2pred_single_view': entry2pred_single_view,
            'pred_single_view2entry': pred_single_view2entry,
        }

    def evaluate_ap_by_idx(self, idx, options=None):
        """
        get plane errors and mask errors
        """
        key = self.rcnnidx2datasetkey(idx)
        if options['selection'] == 'top1_nomerge':
            pred_corr=[]
            tran_topk = 0; rot_topk = 0; pred_camera = None 
            plane_param_override=None
        elif options['selection'] == 'top1':
            distance_matrix = self.get_emb_distance_matrix(idx)
            pred_corr=np.argwhere(self.get_assignment(idx, distance_matrix, method='km', weight={'threshold': options['threshold']}))
            tran_topk = 0; rot_topk = 0; pred_camera = None 
            plane_param_override=None
        elif options['selection'] == 'km_search_cam':
            pred_corr = np.argwhere(self.optimized_dict[idx]['best_assignment'])
            tran_topk = -2; rot_topk=-2; pred_camera = self.optimized_dict[idx]['best_camera']
            plane_param_override = None
        elif options['selection'] == 'non_linear_optimize':
            """
            if not hasattr(self, 'km_dict'):
                with open('/Pool1/users/jinlinyi/exp/planercnn_detectron2/e22_depth_cam/e04_debug_newbox_p3/evaluate_31999_test_predbox/optimize/km_search_cam/optimized_dict.pkl', 'rb') as f:
                    self.km_dict = pickle.load(f)
            """
            if 'plane_param_override' not in self.optimized_dict[idx].keys():
                self.optimized_dict[idx]['plane_param_override'] = None
            pred_corr = np.argwhere(self.optimized_dict[idx]['best_assignment'])
            tran_topk = -2; rot_topk=-2; pred_camera = self.optimized_dict[idx]['best_camera']
            # pred_camera['rotation'] = self.km_dict[idx]['best_camera']['rotation']
            plane_param_override = self.optimized_dict[idx]['plane_param_override']
            # plane_param_override = None
        elif options['selection'] == 'gtcam_nomerge':
            pred_corr=[]
            tran_topk = -1; rot_topk=-1; pred_camera = None
            plane_param_override=None
        elif options['selection'] == 'planeOdometry':
            pred_corr=[]
            tran_topk = -2; rot_topk=-2; pred_camera = self.optimized_dict[idx]['best_camera']
            plane_param_override=None
        elif options['selection'] == 'superglue':
            pred_corr=[]
            tran_topk = -2; rot_topk=-2; pred_camera = self.optimized_dict[idx]['best_camera']
            plane_param_override=None
        else:
            raise NotImplementedError
        """
        PRED
        """
        # Load predict camera
        if pred_camera is None:
            pred_camera = self.get_camera_info(idx, tran_topk, rot_topk)
            pred_camera = {'position': np.array(pred_camera['position']), 'rotation': quaternion.from_float_array(pred_camera['rotation'])}
        else:
            assert(tran_topk == -2 and rot_topk == -2)
            pred_camera = {'position': np.array(pred_camera['position']), 'rotation': quaternion.from_float_array(pred_camera['rotation'])}

        # Load single view prediction
        pred = {'0':{}, '1':{}, 'merged':{}, 'corrs': pred_corr, 'camera': pred_camera, '0_local':{}, '1_local':{}}
        for i in range(2):
            if i == 0:
                camera_info = pred_camera
            else:
                camera_info = {'position': np.array([0,0,0]), 'rotation': np.quaternion(1,0,0,0)}
            p_instance = create_instances(self.rcnn_data[idx][str(i)]['instances'], [self.dataset_dict[key][str(i)]["height"], self.dataset_dict[key][str(i)]["width"]], 
                                                    pred_planes=self.rcnn_data[idx][str(i)]['pred_plane'].numpy(), 
                                                    conf_threshold=self.score_threshold)
            if plane_param_override is None:
                pred_plane_single = p_instance.pred_planes
            else:
                pred_plane_single = plane_param_override[str(i)]
            # Local frame
            offset = np.maximum(np.linalg.norm(pred_plane_single, ord=2, axis=1), 1e-5).reshape(-1,1)
            normal = pred_plane_single / offset
            pred[str(i)+'_local']['offset'] = offset
            pred[str(i)+'_local']['normal'] = normal
            pred[str(i)+'_local']['scores'] = p_instance.scores

            # Global frame
            plane_global = get_plane_params_in_global(pred_plane_single, camera_info)
            offset = np.maximum(np.linalg.norm(plane_global, ord=2, axis=1), 1e-5).reshape(-1,1)
            normal = plane_global / offset
            pred[str(i)]['offset'] = offset
            pred[str(i)]['normal'] = normal
            pred[str(i)]['scores'] = p_instance.scores
        # Merge prediction
        merged_offset = []
        merged_normal = []
        merged_score = []
        for i in range(2):
            for ann_id in range(len(pred[str(i)]['scores'])):
                if len(pred['corrs'])==0 or ann_id not in pred['corrs'][:,i]:
                    merged_offset.append(pred[str(i)]['offset'][ann_id])
                    merged_normal.append(pred[str(i)]['normal'][ann_id])
                    merged_score.append(pred[str(i)]['scores'][ann_id])
        for ann_id in pred['corrs']:
            # average normal
            normal_pair = np.vstack((pred['0']['normal'][ann_id[0]], pred['1']['normal'][ann_id[1]]))
            w, v = eigh(normal_pair.T@normal_pair)
            avg_normals = v[:,np.argmax(w)]
            if (avg_normals@normal_pair.T).sum() < 0:
                avg_normals = - avg_normals
            # average offset
            avg_offset = (pred['0']['offset'][ann_id[0]] + pred['1']['offset'][ann_id[1]]) / 2
            merged_offset.append(avg_offset)
            merged_normal.append(avg_normals)
            # max score
            merged_score.append(max(pred['0']['scores'][ann_id[0]], pred['1']['scores'][ann_id[1]]))
        pred['merged'] = {
            'merged_offset': np.array(merged_offset),
            'merged_normal': np.array(merged_normal),
            'merged_score': np.array(merged_score)[:, np.newaxis],
        }
        """
        GT
        """
        gt_camera = self.get_camera_info(idx, -1, -1)
        gt_camera = {'position': np.array(gt_camera['position']), 'rotation': quaternion.from_float_array(gt_camera['rotation'])}
        gt_corr = self.get_gt_affinity(idx, rtnformat='list', gtbox=True)
        
        # Load single view gt
        gt = {'0':{}, '1':{}, 'merged':{}, 'corrs': gt_corr, 'camera': gt_camera, '0_local':{}, '1_local':{}}
        for i in range(2):
            if i == 0:
                camera_info = gt_camera
            else:
                camera_info = {'position': np.array([0,0,0]), 'rotation': np.quaternion(1,0,0,0)}
            plane_params = np.array([ann['plane'] for ann in self.dataset_dict[key][str(i)]['annotations']])

            # Local frame
            offset = np.maximum(np.linalg.norm(plane_params, ord=2, axis=1), 1e-5).reshape(-1,1)
            normal = plane_params / offset
            gt[str(i)+'_local']['offset'] = offset
            gt[str(i)+'_local']['normal'] = normal

            # Global frame
            plane_global = get_plane_params_in_global(plane_params, camera_info)
            offset = np.maximum(np.linalg.norm(plane_global, ord=2, axis=1), 1e-5).reshape(-1,1)
            normal = plane_global / offset
            gt[str(i)]['offset'] = offset
            gt[str(i)]['normal'] = normal
        # Merge gt
        merged_offset = []
        merged_normal = []
        for i in range(2):
            for ann_id in range(len(gt[str(i)]['offset'])):
                if len(gt['corrs'])==0 or ann_id not in gt['corrs'][:,i]:
                    merged_offset.append(gt[str(i)]['offset'][ann_id])
                    merged_normal.append(gt[str(i)]['normal'][ann_id])
        for ann_id in gt['corrs']:
            # average normal
            assert(np.linalg.norm(gt['0']['normal'][ann_id[0]] - gt['1']['normal'][ann_id[1]]) < 1e-5)
            assert(np.abs(gt['0']['offset'][ann_id[0]] - gt['1']['offset'][ann_id[1]]) < 1e-5)
            merged_offset.append(gt['0']['offset'][ann_id[0]])
            merged_normal.append(gt['0']['normal'][ann_id[0]])
        gt['merged'] = {
            'merged_offset': np.array(merged_offset),
            'merged_normal': np.array(merged_normal),
        }
        """
        ERRORs
        """
        # compute individual error in its own frame
        individual_error_offset = {}
        individual_error_normal = {}
        for i in range(2):
            individual_error_offset[str(i)] = np.abs(pred[str(i)+'_local']['offset'] - gt[str(i)+'_local']['offset'].T)
            individual_error_normal[str(i)] = np.arccos(np.clip(np.abs(pred[str(i)+'_local']['normal']@gt[str(i)+'_local']['normal'].T), -1,1)) / np.pi * 180
        individual_miou = self.get_maskiou(idx)

        # compute merged error
        err_offsets = np.abs(pred['merged']['merged_offset'] - gt['merged']['merged_offset'].T)
        err_normals = np.arccos(np.clip(np.abs(pred['merged']['merged_normal']@gt['merged']['merged_normal'].T), -1,1)) / np.pi * 180
        mask_iou = self.get_maskiou_merged(idx, pred_corr=pred['corrs'], gt_corr=gt['corrs'])
        output = {
            'err_offsets': err_offsets,
            'err_normals': err_normals,
            'mask_iou': mask_iou,
            'scores': pred['merged']['merged_score'],
            'individual_error_offset': individual_error_offset,
            'individual_error_normal': individual_error_normal,
            'individual_miou': individual_miou,
            'individual_score': {'0': pred['0']['scores'].reshape(-1,1), '1':pred['1']['scores'].reshape(-1,1)},
        }
        return output


    def evaluate_by_idx_pcd(self, idx, options=None):
        """
        Deprecated, evaluate using chamfer distance
        """
        if options['selection'] == 'top1':
            distance_matrix = self.get_emb_distance_matrix(idx)
            pred_corr=np.argwhere(self.get_assignment(idx, distance_matrix, method='km', weight={'threshold': options['threshold']}))
            pred = self.get_pcd_normal_in_global(idx=idx, tran_topk=0, rot_topk=0, gt_plane=False, gt_segm=False, pred_corr=pred_corr, boxscore=True,
            gt2pred=False)
            gt = self.get_pcd_normal_in_global(idx=idx, tran_topk=-1, rot_topk=-1, gt_plane=True, gt_segm=True)
        elif options['selection'] == 'km_search_cam':
            pred = self.get_pcd_normal_in_global(idx=idx, tran_topk=-2, rot_topk=-2, pred_camera= self.optimized_dict[idx]['best_camera'],
                gt_plane=False, gt_segm=False, pred_corr=np.argwhere(self.optimized_dict[idx]['best_assignment']), boxscore=True,
                gt2pred=False)
            gt = self.get_pcd_normal_in_global(idx=idx, tran_topk=-1, rot_topk=-1, gt_plane=True, gt_segm=True)
        elif options['selection'] == 'non_linear_optimize':
            if 'plane_param_override' not in self.optimized_dict[idx].keys():
                self.optimized_dict[idx]['plane_param_override'] = None
            pred = self.get_pcd_normal_in_global(idx=idx, tran_topk=-2, rot_topk=-2, pred_camera= self.optimized_dict[idx]['best_camera'],
                gt_plane=False, gt_segm=False, pred_corr=np.argwhere(self.optimized_dict[idx]['best_assignment']), boxscore=True,
                gt2pred=False, plane_param_override=self.optimized_dict[idx]['plane_param_override'])
            gt = self.get_pcd_normal_in_global(idx=idx, tran_topk=-1, rot_topk=-1, gt_plane=True, gt_segm=True)
        elif options['selection'] == 'single_view':
            pred = self.get_pcd_normal_in_global(idx=idx, tran_topk=-1, rot_topk=-1, 
                gt_plane=False, gt_segm=False, pred_corr=None, boxscore=True,
                gt2pred=False, merge=False)
            gt = self.get_pcd_normal_in_global(idx=idx, tran_topk=-1, rot_topk=-1, gt_plane=True, gt_segm=True, merge=False)
            return_dict = {}
            for i in range(2):
                pcds = []
                verts_list = pred[str(i)]['verts_list']
                for verts in verts_list:
                    assert(torch.cuda.is_available())
                    pcd = verts.unsqueeze(0).cuda()
                    pcds.append(pcd)
                
                # normal
                planes = pred[str(i)]['plane']
                offset = np.maximum(np.linalg.norm(planes, ord=2, axis=1), 1e-5).reshape(-1,1)
                normals = planes / offset
                
                # number of predicted planes
                ndt = len(normals)
                scores = pred[str(i)]['scores']
                scores = scores[:, np.newaxis]

                # Get Ground Truth.
                # point cloud 
                gt_pcds = []
                verts_list = gt[str(i)]['verts_list']
                for verts in verts_list:
                    pcd = verts.unsqueeze(0).cuda()
                    gt_pcds.append(pcd)
                
                # normal
                gt_planes = gt[str(i)]['plane']
                gt_offset = np.maximum(np.linalg.norm(gt_planes, ord=2, axis=1), 1e-5).reshape(-1,1)
                gt_normals = gt_planes / gt_offset
                
                # number of gt planes
                ngt = len(gt_normals)

                # no detected objects
                if ndt == 0:
                    return_dict[str(i)] = {
                        'err_l2': [], 'err_offsets': [], 'err_normals': [],
                        'err_pcds': [], 'scores': [],
                    }

                # compute error
                err_l2 = torch.cdist(torch.FloatTensor(planes), torch.FloatTensor(gt_planes), p=2).numpy()
                err_offsets = np.abs(offset - gt_offset.T)
                err_normals = np.arccos(np.clip((normals@gt_normals.T), -1,1)) / np.pi * 180
                err_pcds = np.zeros((ndt, ngt))
                for r in range(ndt):
                    for c in range(ngt):
                        err_pcds[r, c] = chamfer_distance(pcds[r], gt_pcds[c])[0].cpu().item()
                return_dict[str(i)] = {
                    'err_l2': err_l2, 'err_offsets': err_offsets, 'err_normals': err_normals,
                    'err_pcds': err_pcds, 'scores': scores,
                }
            return return_dict
        else:
            raise NotImplementedError
        # Get Predictions.
        # point cloud 
        pcds = []
        verts_list = pred['merged']['verts_list']
        for verts in verts_list:
            assert(torch.cuda.is_available())
            pcd = verts.unsqueeze(0).cuda()
            pcds.append(pcd)
        
        # normal
        planes = pred['merged']['plane']
        offset = np.maximum(np.linalg.norm(planes, ord=2, axis=1), 1e-5).reshape(-1,1)
        normals = planes / offset
        
        # number of predicted planes
        ndt = len(normals)
        scores = pred['merged']['scores']
        scores = scores[:, np.newaxis]

        # Get Ground Truth.
        # point cloud 
        gt_pcds = []
        verts_list = gt['merged']['verts_list']
        for verts in verts_list:
            pcd = verts.unsqueeze(0).cuda()
            gt_pcds.append(pcd)
        
        # normal
        gt_planes = gt['merged']['plane']
        gt_offset = np.maximum(np.linalg.norm(gt_planes, ord=2, axis=1), 1e-5).reshape(-1,1)
        gt_normals = gt_planes / gt_offset
        
        # number of gt planes
        ngt = len(gt_normals)

        # no detected objects
        if ndt == 0:
            return {'err_l2': [], 'err_offsets': [], 'err_normals': [],
                'err_pcds': [], 'scores': [],}

        # compute error
        err_l2 = torch.cdist(torch.FloatTensor(planes), torch.FloatTensor(gt_planes), p=2).numpy()
        err_offsets = np.abs(offset - gt_offset.T)
        err_normals = np.arccos(np.clip((normals@gt_normals.T), -1,1)) / np.pi * 180
        err_pcds = np.zeros((ndt, ngt))
        for i in range(ndt):
            for j in range(ngt):
                err_pcds[i, j] = chamfer_distance(pcds[i], gt_pcds[j])[0].cpu().item()
        return {
                'err_l2': err_l2, 'err_offsets': err_offsets, 'err_normals': err_normals,
                'err_pcds': err_pcds, 'scores': scores,
            }

    def evaluate_by_list_pcd(self, idxs, options=None):
        return_dict = {}
        for idx in tqdm(idxs, desc='base'+options['prefix']):
            rtn = self.evaluate_by_idx_pcd(idx, options)
            return_dict[idx] = rtn
        with open(os.path.join(options['save_folder'], options['prefix']+'.pkl'), 'wb') as f:
            pickle.dump(return_dict, f)

    def evaluate_by_list(self, idxs, return_dict, options=None):
        for idx in idxs:
            rtn = self.evaluate_ap_by_idx(idx, options)
            return_dict[idx] = rtn

    def merge_plane_params_from_global_params(self, param1, param2, corr_list):
        """
        input: plane parameters in global frame
        output: merged plane parameters using corr_list
        """
        pred = {'0':{}, '1':{}}
        pred['0']['offset'] = np.maximum(np.linalg.norm(param1, ord=2, axis=1), 1e-5).reshape(-1,1)
        pred['0']['normal'] = param1 / pred['0']['offset']
        pred['1']['offset'] = np.maximum(np.linalg.norm(param2, ord=2, axis=1), 1e-5).reshape(-1,1)
        pred['1']['normal'] = param2 / pred['1']['offset']
        for ann_id in corr_list:
            # average normal
            normal_pair = np.vstack((pred['0']['normal'][ann_id[0]], pred['1']['normal'][ann_id[1]]))
            w, v = eigh(normal_pair.T@normal_pair)
            avg_normals = v[:,np.argmax(w)]
            if (avg_normals@normal_pair.T).sum() < 0:
                avg_normals = - avg_normals
            # average offset
            avg_offset = (pred['0']['offset'][ann_id[0]] + pred['1']['offset'][ann_id[1]]) / 2
            avg_plane = avg_normals * avg_offset
            param1[ann_id[0]] = avg_plane
            param2[ann_id[1]] = avg_plane
        return param1, param2

    def merge_plane_params_from_local_params(self, plane_locals, corr_list, camera_pose):
        """
        input: plane parameters in camera frame
        output: merged plane parameters using corr_list
        """
        param1, param2 = plane_locals['0'], plane_locals['1']
        param1_global = get_plane_params_in_global(param1, camera_pose)
        param2_global = get_plane_params_in_global(param2, {'position': np.array([0,0,0]), 'rotation': np.quaternion(1,0,0,0)})
        param1_global, param2_global = self.merge_plane_params_from_global_params(param1_global, param2_global, corr_list)
        param1 = get_plane_params_in_local(param1_global, camera_pose)
        param2 = get_plane_params_in_local(param2_global, {'position': np.array([0,0,0]), 'rotation': np.quaternion(1,0,0,0)})
        return {'0': param1, '1': param2}

        



def optimize_by_list_multiprocess(mo, num_process, idx_list, weight):
    max_iter = len(idx_list)
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    
    per_thread = int(np.ceil(max_iter / num_process))
    split_by_thread = [idx_list[i*per_thread:(i+1)*per_thread] for i in range(num_process)]
    for i in range(num_process):
        p = Process(target=mo.optimize_by_list, args=(split_by_thread[i], return_dict, 5, weight))
        p.start()
        jobs.append(p)

    
    prev = 0
    with tqdm(total=max_iter) as pbar:
        while True:
            time.sleep(0.1)
            curr = len(return_dict.keys())
            pbar.update(curr - prev)
            prev = curr
            if curr == max_iter:
                break
    
    for job in jobs:
        job.join()

    return return_dict




def main(args):
    random.seed(4)
    np.random.seed(4)
    mo = PlaneFormerResult(args, dataset=args.dataset_phase)
    save_folder = args.save_folder
    
    IPAA_list = []
    exp_name = []
    if args.num_data == -1:
        idx_list = range(len(mo.rcnn_data))
    else:
        idx_list = np.random.choice(len(mo.rcnn_data), args.num_data, replace=False)
    
    #idx_list = [5536]
    
    if args.task == 'threshold':
        best_auc = 0
        best_th = 0
        for threshold in np.arange(0.0, 1.0, 0.1):
            print(threshold)
            baseline = mo.get_baseline(idx_list, method='km', threshold=threshold)
            AP_FPR = mo.evaluate_matching(baseline, method='AP_FPR_95')
            IPAA_baseline_km = mo.evaluate_matching(baseline)
            auc = compute_auc(IPAA_baseline_km)
            if auc > best_auc:
                best_auc = auc
                best_th = threshold
        print(f"Best threshold: {best_th}, auc {best_auc:.4f}")
        import pdb; pdb.set_trace()
    elif args.task == 'associative3D':
        weight = {
            # 'threshold': random.randint(0, 10) / 10,
            'threshold': 0.7,
            'lambda_emb': 0.47,
            'lambda_geo_l2': 0.00,
            'l2_clamp': 5,
            'lambda_geo_normal': 0.25,
            'lambda_geo_offset': 0.28,
            'offset_clamp': 4,
            'topk_tran': 32,
            'topk_rot': 32,
            'score_weight': [0.311, 0.166, 0.092, -0.432],
            'assignment': 'proposal',
            'maxk': 5,
        }
        print(weight)
        return_dict = optimize_by_list_multiprocess(mo, args.num_process, idx_list, weight)
        import pdb; pdb.set_trace()
        try:
            IPAA_optimized = mo.evaluate_matching(return_dict)
            camera_eval_dict = mo.evaluate_camera(return_dict)
        except:
            IPAA_optimized = None
            camera_eval_dict = None
        mo.save_dict(return_dict, folder=save_folder, prefix=f'optimized_dict')
        param_result = {
            'weight': weight,
            'IPAA': IPAA_optimized,
            'camera': camera_eval_dict,
        }
        mo.save_json(param_result, folder=save_folder, prefix=f'log')
        import pdb; pdb.set_trace()
        IPAA_list.append(IPAA_optimized)
        exp_name.append(timestamp)
        plot_IPAAs(IPAA_list, os.path.join(save_folder, 'ipaa.png'), exp_name)
    elif args.task == 'collect':
        logs = sorted(list(glob(os.path.join(save_folder, '*', 'log.json'))))
        results = []
        for log in logs:
            with open(log, 'r') as f:
                param_result = json.load(f)
            param_result['path'] = log
            param_result['AUC_IPAA'] = compute_auc(param_result['IPAA'])
            results.append(param_result)
    elif args.task == 'baseline':
        baseline = mo.get_baseline(idx_list, method='km', threshold=0.7)
        IPAA_baseline_km = mo.evaluate_matching(baseline, 'IPAA_sorted_by_single_view_prediction')
        print(IPAA_baseline_km)
        for key in IPAA_baseline_km.keys():
            print(f"top {100*key}%: roc_ipaa_km: {compute_auc(IPAA_baseline_km[key])}")
        baseline = mo.get_baseline(idx_list, method='hungarian')
        IPAA_baseline_hung = mo.evaluate_matching(baseline)
        print(f"roc_ipaa_hungarian: {compute_auc(IPAA_baseline_hung)}")
        print(mo.evaluate_camera(baseline, sorted_by_single_view_prediction=True))

    elif args.task == 'corr_wall':
        np.random.seed(0)
        vis_list = np.random.choice(idx_list, args.num_data, replace=False)
        for idx in tqdm(vis_list):
            tqdm.write(str(idx))
            save_folder_tmp = os.path.join(save_folder, str(idx).zfill(4))
            os.makedirs(save_folder_tmp, exist_ok=True)
            mo.save_original_img(idx, save_folder_tmp, prefix='00_original', combined=True)

            mo.save_matching(idx, mo.optimized_dict[idx]['best_assignment'], save_folder_tmp, 
            prefix='01_corr_sparseplane', fp=True, gt_box=mo.gt_box, gt_segm=False, paper_img=True, vertical=True)

            mo.save_matching(idx, mo.planeformer_optimized_dict[idx]['best_assignment'], save_folder_tmp, 
            prefix='02_corr_ours', fp=True, gt_box=mo.gt_box, gt_segm=False, paper_img=True, vertical=True)

            mo.save_matching(idx, mo.get_gt_affinity(idx, gtbox=True), save_folder_tmp, 
            prefix='03_gt', fp=True, gt_box=True, gt_segm=True, paper_img=True, vertical=True)


    elif args.task == 'visualize':
        np.random.seed(0)
        # vis_list = np.random.choice(idx_list, 20, replace=False)
        # f = open('/Pool1/users/jinlinyi/workspace/p-voxels-real/planeRCNN/sorted_by_singleAP.txt', 'r')
        # lines = f.readlines()
        # f.close()
        # #import pdb; pdb.set_trace()
        # vis_list = lines[0].split(',')
        # vis_list = [int(idx.strip()) for idx in vis_list]
        # # subsample
        # vis_list = vis_list[7:len(vis_list):800]

        # perfect_list = [49, 63, 92, 158, 171, 176, 187, 192, 218, 224, 226, 349, 
        #     364, 421, 486, 561, 597, 605, 618, 620, 621, 692, 718, 765, 802, 966, 
        #     975, 983, 1033, 1038, 1041, 1042, 1046, 1068, 1071, 1074, 1079, 1123, 
        #     1160, 1211, 1227, 1230, 1232, 1274, 1275, 1284, 1291, 1293, 1303, 1304, 
        #     1315, 1334, 1386, 1424, 1433, 1434, 1478, 1479, 1498, 1508, 1512, 1523, 
        #     1532, 1585, 1586, 1637, 1732, 1767, 1788, 1790, 1839, 1926, 2041, 2042, 
        #     2048, 2051, 2159, 2169, 2183, 2217, 2224, 2241, 2248, 2282, 2397, 2422, 
        #     2452, 2539, 2556, 2563, 2564, 2592, 2610, 2721, 2757, 2795, 2839, 2862, 
        #     2866, 2872, 2880, 2886, 2935, 2956, 2958, 2961, 2963, 2973, 2985, 2990, 
        #     3019, 3031, 3060, 3068, 3104, 3128, 3136, 3169, 3236, 3238, 3240, 3241, 
        #     3245, 3262, 3265, 3273, 3279, 3284, 3292, 3304, 3316, 3399, 3401, 3402, 
        #     3403, 3404, 3408, 3416, 3422, 3635, 3637, 3685, 3710, 3721, 3777, 3924, 
        #     3949, 3952, 3963, 3965, 3971, 3982, 3985, 3988, 3989, 3996, 4055, 4062, 
        #     4192, 4194, 4203, 4376, 4393, 4395, 4411, 4414, 4430, 4465, 4478, 4479, 
        #     4488, 4502, 4503, 4507, 4508, 4513, 4537, 4540, 4543, 4544, 4546, 4548, 
        #     4549, 4550, 4554, 4558, 4559, 4560, 4561, 4564, 4569, 4571, 4573, 4668,
        #     4743, 4757, 4764, 4786, 4807, 4818, 4842, 4860, 4866, 4878, 4896, 4898, 
        #     4910, 4963, 5068, 5070, 5075, 5079, 5090, 5094, 5118, 5120, 5143, 5150,
        #     5161, 5208, 5215, 5218, 5242, 5262, 5270, 5287, 5296, 5297, 5301, 5312,
        #     5317, 5320, 5330, 5369, 5386, 5415, 5417, 5418, 5419, 5436, 5437, 5438,
        #     5439, 5441, 5444, 5458, 5473, 5483, 5487, 5495, 5564, 5578, 5608, 5614,
        #     5615, 5619, 5626, 5651, 5656, 5667, 5679, 5693, 5722, 5737, 5739, 5746, 
        #     5747, 5749, 5750, 5754, 5773, 5774, 5777, 5783, 5787, 5788, 5794, 5799, 
        #     5801, 5803, 5804, 5805, 5806, 5807, 5809, 5817, 5818, 5819, 5820, 5821, 
        #     5828, 5838, 5861, 5892, 5899, 5910, 5916, 5923, 5926, 5955, 5972, 5978, 
        #     5984, 6004, 6005, 6011, 6019, 6025, 6030, 6034, 6067, 6073, 6092, 6190, 
        #     6202, 6206, 6208, 6218, 6222, 6224, 6225, 6239, 6277, 6301, 6416, 6420, 
        #     6423, 6424, 6433, 6434, 6443, 6448, 6449, 6453, 6460, 6485, 6491, 6522, 
        #     6526, 6531, 6595, 6618, 6646, 6657, 6666, 6669, 6677, 6687, 6707, 6708, 
        #     6716, 6717, 6723, 6739, 6749, 6775, 6784, 6788, 6798, 6807, 6814, 6820, 
        #     6821, 6852, 6853, 6856, 6860, 6876, 6889, 6905, 6910, 6911, 6939, 6944, 
        #     6945, 6952, 6960, 6966, 6983, 6988, 6989, 7002, 7011, 7016, 7034, 7040, 
        #     7053, 7063, 7084, 7095, 7104, 7124, 7125, 7224, 7248, 7405, 7409, 7410, 
        #     7412, 7420, 7423, 7424, 7425, 7426, 7437, 7446, 7456, 7459, 7470, 7507, 
        #     7557, 7560, 7593, 7598, 7599, 7629, 7697, 7701, 7713, 7736, 7782, 7783, 
        #     7826, 7830, 7841, 7848, 7865, 7876, 7888, 7898, 7917, 7918, 7920, 7952]
        # comparison_list = [ 110,  497,  764,  799,  892, 1086, 1281, 1309, 1319, 1345, 1350,
        # 1387, 1588, 1595, 1715, 1849, 1894, 1999, 2013, 2370, 2478, 2522,
        # 2581, 2624, 3030, 3118, 3132, 3239, 3533, 3558, 3764, 3926, 4394,
        # 4884, 4914, 4982, 5302, 5696, 5708, 5911, 5988, 6002, 6131, 6222,
        # 6437, 6761, 6769, 6811, 6815, 6826, 6913, 7250, 7298, 7372, 7464,
        # 7521, 7624, 7732, 7830, 7871]
        # vis_list = comparison_list
        # vis_list = np.random.choice(perfect_list, 200, replace=False)
        # example_wall = [14, 1074, 1293, 1419, 4058, 4540, 5195, 5312, 5806, 6876, 7470,
        # 1079, 2612, 4194, 4843, 5215, 5617, 6788, 7082, 7890]
        # # example_wall = [5806,4194,4564]
        # vis_list = example_wall
        vis_list = idx_list
        webvis=False
        #vis_list = [3253]
        # sorted_idx_list = sorted(list(range(len(mo.rcnn_data))), key=lambda x:mo.single_view_err[x]['worst_l2_mean'])
        # vis_list = np.random.choice(sorted_idx_list[:int(len(sorted_idx_list)*.25)], 40, replace=False)
        # vis_list = np.random.choice(sorted_idx_list[int(len(sorted_idx_list)*.25):int(len(sorted_idx_list)*.50)], 20, replace=False)
        # vis_list = np.random.choice(sorted_idx_list[int(len(sorted_idx_list)*.50):int(len(sorted_idx_list)*.75)], 20, replace=False)
        # vis_list = np.random.choice(sorted_idx_list[int(len(sorted_idx_list)*.75):int(len(sorted_idx_list)*1)], 40, replace=False)

        #import pdb; pdb.set_trace()
        # cnt = 0
        for idx in tqdm(vis_list):
            tqdm.write(str(idx))
            # if 'plane_param_override' not in optimized_dict[idx].keys():
            #     optimized_dict[idx]['plane_param_override'] = None
            # save_folder_tmp = os.path.join(save_folder, str(idx).zfill(4))
            # mo.save_matching(idx, optimized_dict[idx]['best_assignment'], save_folder_tmp, 
            # prefix='02_corr', fp=True, gt_box=mo.gt_box, gt_segm=False, paper_img=True)
            # continue
            
            # Note: I add prefix 00 and 01 here to make sure the order is correct in webvis.
            # exps except e05

           
            save_folder_tmp = os.path.join(save_folder, str(idx).zfill(4))
            # save_folder_tmp = os.path.join(save_folder, str(cnt).zfill(2)+str(idx).zfill(4))
            # cnt+=1
            # mo.save_pair_objects(idx, 0, 0, save_folder_tmp, 
            #                     prefix="00_Top1Cam_GTDepth_Manhattan", gt_plane=False,  show_camera=True,
            #                     gt_segm=False, 
            #                     corr_list=[], reduce_size=True,
            #                     webvis=webvis)
            # mo.save_pair_objects(idx, -2, -2, save_folder_tmp, 
            #                     prefix="01_Odometry_MidasDepth_Manhattan", gt_plane=False,  show_camera=True,
            #                     gt_segm=False, 
            #                     pred_camera=optimized_dict[idx]['best_camera'],
            #                     corr_list=[], reduce_size=True,
            #                     webvis=webvis)
            # mo.save_matching(idx, mo.get_gt_affinity(idx, gtbox=True), save_folder_tmp, 
            # prefix='02_corr', fp=True, gt_box=True, gt_segm=True, paper_img=True)
            os.makedirs(save_folder_tmp, exist_ok=True)
            mo.save_original_img(idx, save_folder_tmp, prefix='00_original')
            cv2.imwrite(os.path.join(save_folder_tmp, '00_placeholder1.png'), np.zeros((10,10,3))) 
            cv2.imwrite(os.path.join(save_folder_tmp, '00_placeholder2.png'), np.zeros((10,10,3))) 
            cv2.imwrite(os.path.join(save_folder_tmp, '00_placeholder3.png'), np.zeros((10,10,3))) 
            
            mo.optimized_dict[idx]['plane_param_override'] = None
            """
            mo.save_matching(idx, mo.planeformer_optimized_dict[idx]['best_assignment'], save_folder_tmp, 
            prefix='00_corr_ours', fp=True, gt_box=mo.gt_box, gt_segm=False, paper_img=True)
            mo.save_pair_objects(idx, 0, 0, save_folder_tmp, 
                                prefix="01_top1_nomerge", gt_plane=False,  show_camera=True,
                                gt_segm=False, 
                                corr_list=[], reduce_size=True,
                                webvis=webvis)
            """
            """
            mo.save_pair_objects(idx, optimized_dict[idx]['best_tran_topk'], 
                                optimized_dict[idx]['best_rot_topk'], save_folder_tmp, 
                                prefix='02_discrete', gt_plane=False, show_camera=True, gt_segm=False,
                                corr_list=np.argwhere(optimized_dict[idx]['best_assignment']), reduce_size=True,
                                webvis=webvis)
            """
            mo.save_pair_objects(idx, -2, -2, save_folder_tmp, 
                                prefix="01_sparseplane", gt_plane=False,  show_camera=True,
                                gt_segm=False, 
                                pred_camera=mo.optimized_dict[idx]['best_camera'],
                                plane_param_override=mo.optimized_dict[idx]['plane_param_override'], 
                                corr_list=np.argwhere(mo.optimized_dict[idx]['best_assignment']), reduce_size=True,
                                webvis=webvis)
            mo.save_pair_objects(idx, -2, -2, save_folder_tmp, 
                                prefix="02_planeformer", gt_plane=False,  show_camera=True,
                                gt_segm=False, 
                                pred_camera=mo.planeformer_optimized_dict[idx]['best_camera'],
                                plane_param_override=mo.planeformer_optimized_dict[idx]['plane_param_override'], 
                                corr_list=np.argwhere(mo.planeformer_optimized_dict[idx]['best_assignment']), reduce_size=True,
                                webvis=webvis)
            mo.save_pair_objects(idx, -1, -1, save_folder_tmp, 
                                prefix="03_gt", gt_plane=True,  show_camera=True,
                                gt_segm=True, 
                                pred_camera=None,
                                corr_list=[], reduce_size=True,
                                webvis=webvis)
            # e05 top1 camera no merge
            #mo.save_pair_objects(idx, -2, -2, save_folder_tmp, 
            #                    prefix="00_05_refinecamsift_predseg_merged", gt_plane=False, 
            #                    pred_camera=optimized_dict[idx]['best_camera'], show_camera=True,
            #                    gt_segm=False, plane_param_override=optimized_dict[idx]['plane_param_override'], corr_list=[], reduce_size=True)

            # ground truth
            # mo.save_pair_objects(idx, -1, -1, save_folder_tmp, prefix='04_10_gt_all', gt_plane=True, gt_segm=True, show_camera=True, webvis=webvis)
            continue
            os.makedirs(save_folder_tmp, exist_ok=True)
            mo.save_original_img(idx, save_folder_tmp, prefix='01_original')
            mo.save_matching(idx, optimized_dict[idx]['best_assignment'], save_folder_tmp, prefix='02_corr', fp=True, gt_box=mo.gt_box, gt_segm=False)
            mo.save_matching(idx, optimized_dict[idx]['best_assignment'], save_folder_tmp, prefix='03_corr', fp=False, gt_box=mo.gt_box, gt_segm=True)
            mo.save_depth(idx, save_folder_tmp, prefix='04_depth')
            # mo.save_pair_objects(idx, -2, -2, save_folder_tmp, prefix="05_kmsearchcam_predseg", gt_plane=False, pred_camera=optimized_dict[idx]['best_camera'], gt_segm=False)
            mo.save_pair_objects(idx, optimized_dict[idx]['best_tran_topk'], optimized_dict[idx]['best_rot_topk'], save_folder_tmp, prefix='06_kmsearchcam_predseg', gt_plane=False)
            mo.save_pair_objects(idx, 0, 0, save_folder_tmp, prefix='07_top1cam_predseg', gt_plane=False)

            gt_cam_bin = mo.get_gt_cam_topk(idx)
            gt_k_tran, gt_k_rot = gt_cam_bin['gt_tran_topk'], gt_cam_bin['gt_rot_topk']
            mo.save_pair_objects(idx, gt_k_tran, gt_k_rot, save_folder_tmp, prefix='08_gtcambin_gtseg', gt_plane=False, gt_segm=True)
            mo.save_pair_objects(idx, -1, -1, save_folder_tmp, prefix='09_gtcam_gtseg', gt_plane=False, gt_segm=True)
            mo.save_pair_objects(idx, -1, -1, save_folder_tmp, prefix='10_gt_all', gt_plane=True, gt_segm=True)

            if False:
                embedding_matrix = mo.get_emb_distance_matrix(idx)

                k_tran, k_rot = [optimized_dict[idx]['best_tran_topk'], optimized_dict[idx]['best_rot_topk']]
                pred_cam = mo.get_camera_info(idx, k_tran, k_rot)

                geo_matrix = mo.get_geo_distance_matrix(idx, k_tran, k_rot) #key: l2, offset, normal
                l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
                normal_matrix = geo_matrix['normal'] / np.pi
                offset_matrix = np.clip(geo_matrix['offset'], 0, weight['offset_clamp']) / weight['offset_clamp']
        
                distance_matrix = weight['lambda_emb'] * embedding_matrix + \
                        weight['lambda_geo_l2'] * l2_matrix + \
                        weight['lambda_geo_normal'] * normal_matrix + \
                        weight['lambda_geo_offset'] * offset_matrix

                assignment = mo.get_assignment(idx, distance_matrix[0], method='km', weight=weight)
                x = np.array([
                            assignment.sum(), 
                            np.log(pred_cam['position_prob']), 
                            np.log(pred_cam['rotation_prob']), 
                            (embedding_matrix*assignment).numpy().mean(),
                            (l2_matrix*assignment).numpy().mean(), 
                            (normal_matrix*assignment).numpy().mean(), 
                            (offset_matrix*assignment).numpy().mean(),])
                score = x@score_weight
                print(f"[Pred]\nk_tran: {k_tran}; k_rot: {k_rot}\nx: {x}\n score: {score}")
                gt_cam_bin = mo.get_gt_cam_topk(idx)
                k_tran, k_rot = gt_cam_bin['gt_tran_topk'], gt_cam_bin['gt_rot_topk']
                pred_cam = mo.get_camera_info(idx, k_tran, k_rot)
                geo_matrix = mo.get_geo_distance_matrix(idx, k_tran, k_rot) #key: l2, offset, normal
                l2_matrix = np.clip(geo_matrix['l2'], 0, weight['l2_clamp']) / weight['l2_clamp']
                normal_matrix = geo_matrix['normal'] / np.pi
                offset_matrix = np.clip(geo_matrix['offset'], 0, weight['offset_clamp']) / weight['offset_clamp']
        
                distance_matrix = weight['lambda_emb'] * embedding_matrix + \
                        weight['lambda_geo_l2'] * l2_matrix + \
                        weight['lambda_geo_normal'] * normal_matrix + \
                        weight['lambda_geo_offset'] * offset_matrix

                assignment = mo.get_assignment(idx, distance_matrix[0], method='km', weight=weight)
                x = np.array([
                            assignment.sum(), 
                            np.log(pred_cam['position_prob']), 
                            np.log(pred_cam['rotation_prob']), 
                            (embedding_matrix*assignment).numpy().mean(),
                            (l2_matrix*assignment).numpy().mean(), 
                            (normal_matrix*assignment).numpy().mean(), 
                            (offset_matrix*assignment).numpy().mean(),])
                score = x@score_weight
                print(f"[GT]\nk_tran: {k_tran}; k_rot: {k_rot}\nx: {x}\n score: {score}")
        #import pdb; pdb.set_trace()


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="A script that optimize objective functions from top k proposals."
    )
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--save-folder", required=True, help="output directory")
    parser.add_argument("--rcnn-cached-file", required=True, help="path to instances_predictions.pth")
    parser.add_argument("--camera-cached-file", required=True, help="path to summary.pkl")
    parser.add_argument("--task", default='', help="random search for weight, can be 'threshold', 'weight', 'collect'")
    parser.add_argument("--num-process", default=50, type=int, help="number of process for multiprocessing")
    parser.add_argument("--num-data", default=-1, type=int, help="number of data to process, if -1 then all.")
    parser.add_argument("--dataset-phase", default='mp3d_val', type=str, help="dataset and phase")
    parser.add_argument("--sparseplane-optimized-dict-path", default='', type=str, help="path to optimized_dict.pkl")
    parser.add_argument("--planeformer-optimized-dict-path", default='', type=str, help="path to optimized_dict.pkl")
    parser.add_argument("--opts", default=[])
    args = parser.parse_args()
    print(args)
    # args = default_argument_parser().parse_args()
    # import pdb; pdb.set_trace()
    main(args)
