import numpy as np
import argparse, os, cv2, torch, pickle, quaternion
import imageio
import random
import shutil
import json
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
from itertools import combinations

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



def merge_plane_params_from_global_params(plane_globals, corr_list):
    """
    input: plane parameters in global frame
    output: merged plane parameters using corr_list
    """
    pred = defaultdict(dict)
    for key in plane_globals.keys():
        pred[key]['offset'] = np.maximum(np.linalg.norm(plane_globals[key], ord=2, axis=1), 1e-5).reshape(-1,1)
        pred[key]['normal'] = plane_globals[key] / pred[key]['offset']
    for ann_id in corr_list:
        assert len(ann_id) == len(plane_globals.keys())
        # average normal
        normal_list = []
        offset_list = []
        for view_id, plane_id in enumerate(ann_id):
            if plane_id != -1:
                normal_list.append(pred[str(view_id)]['normal'][plane_id])
                offset_list.append(pred[str(view_id)]['offset'][plane_id])
        normal_list = np.vstack(normal_list)
        w, v = eigh(normal_list.T@normal_list)
        avg_normals = v[:,np.argmax(w)]
        if (avg_normals@normal_list.T).sum() < 0:
            avg_normals = - avg_normals
        
        # average offset
        avg_offset = np.average(offset_list)
        avg_plane = avg_normals * avg_offset
        for view_id, plane_id in enumerate(ann_id):
            if plane_id != -1:
                plane_globals[str(view_id)][plane_id] = avg_plane
    return plane_globals

def merge_plane_params_from_local_params(plane_locals, corr_list, camera_pose):
    """
    input: plane parameters in camera frame
    output: merged plane parameters using corr_list
    """
    plane_globals = {}
    for i in range(len(camera_pose)):
        plane_globals[str(i)] = get_plane_params_in_global(plane_locals[str(i)], camera_pose[i])
    
    plane_globals = merge_plane_params_from_global_params(plane_globals, corr_list)

    plane_locals_merged = {}
    for i in range(len(camera_pose)):
        plane_locals_merged[str(i)] = get_plane_params_in_local(plane_globals[str(i)], camera_pose[i])
    return plane_locals_merged



def get_relative_T_in_cam2_ref(R2, t1, t2):
    new_c2 = - np.dot(R2, t2)
    return np.dot(R2, t1) + new_c2

def get_relative_pose_from_datapoint(pose, anchor):
    q1 = pose['rotation']
    q2 = anchor['rotation']
    t1 = pose['position']
    t2 = anchor['position']
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
    relative_rotation = quaternion.as_float_array(relative_rotation).tolist()
    relative_translation = relative_translation.tolist()
    rel_pose = {'position': relative_translation, 'rotation': relative_rotation}
    return rel_pose


class PlaneFormerMVResult():
    def __init__(self, score_threshold, dataset_json_f, num_view):
        """
        Cache:
        - Predicted camera poses
        - Predicted plane parameters
        - Predicted embedding distance matrix
        
        """
        self.score_threshold = score_threshold
        # with open(rcnn_cached_file, 'rb') as f:
        #     self.rcnn_data = [pickle.load(rcnn_cached_file)]
        with open(dataset_json_f, 'r') as f:
            self.dataset_dict = json.load(f)['data']
        self.num_view = num_view

    def save_matching(self, idx, rcnn_output, assignment, output_dir, prefix='', fp=True, gt_box=True, gt_segm=False, paper_img=False, vertical=False):
        """
        fp: whether show fp or fn
        gt_box: whether use gtbox 
        """
        blended = {}
        basenames = {}
        uniq_idx = 0
        # centroids for matching
        centroids = defaultdict(list)
        for i in range(self.num_view):
            img_file = self.dataset_dict[idx][str(i)]['file_name']
            house_name, basename = self.dataset_dict[idx][str(i)]["image_id"].split('_', 1)
            basenames[str(i)] = basename
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)[:,:,::-1]
            height, width, _ = img.shape
            vis = Visualizer(img)
            p_instance = create_instances(rcnn_output[str(i)]['instances'], img.shape[:2], 
                                            pred_planes=rcnn_output[str(i)]['pred_plane'].numpy(), 
                                            conf_threshold=self.score_threshold)
            seg_blended = get_labeled_seg(p_instance, self.score_threshold, vis, paper_img=paper_img)
            blended[str(i)] = seg_blended
            # centroid of mask
            for ann in rcnn_output[str(i)]['instances']:
                M=center_of_mass(mask_util.decode(ann['segmentation']))
                centroids[str(i)].append(M[::-1]) # reverse for opencv
            centroids[str(i)] = np.array(centroids[str(i)])

        for pair in combinations(np.arange(self.num_view), 2):
            pred_corr_list =  np.array(assignment)[:, [pair[0], pair[1]]]
            pred_corr_list_filtered = []
            for corr in pred_corr_list:
                if -1 in corr:
                    continue
                pred_corr_list_filtered.append(corr)
            pred_matching_fig = draw_match(blended[str(pair[0])], blended[str(pair[1])], centroids[str(pair[0])], centroids[str(pair[1])], 
                                            np.array(pred_corr_list_filtered), np.ones(len(pred_corr_list_filtered)), vertical=vertical)
            os.makedirs(output_dir, exist_ok=True)
            pred_matching_fig.save(os.path.join(output_dir, prefix+f'_{pair[0]}_{pair[1]}.png'))

    def save_input(self, idx, output_dir, prefix):
        for i in range(self.num_view):
            img_file = self.dataset_dict[idx][str(i)]['file_name']
            shutil.copy(img_file, os.path.join(output_dir, f'{prefix}_{str(i)}.png'))
            os.chmod(os.path.join(output_dir, f'{prefix}_{str(i)}.png'), 0o755)


    def save_multiple_objects(self, 
        idx, tran_topk, rot_topk, output_dir, rcnn_output, prefix='', 
        gt_plane=False, pred_camera=None, gt_segm=False, 
        plane_param_override=None, show_camera=False, corr_list=[], 
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
        
        file_names = [self.dataset_dict[idx][str(i)]['file_name'] for i in range(self.num_view)]
        key = 'debug'
        basenames = {}
        meshes_list = []
        #map_files = []
        uv_maps = []
        cam_list = []
        vis_idx = 0
        # get plane parameters
        plane_locals = {}
        p_instances = {}
        for i in range(self.num_view):
            if not gt_plane or not gt_segm:
                p_instances[str(i)] = create_instances(rcnn_output[str(i)]['instances'], [480, 640], 
                    pred_planes=rcnn_output[str(i)]['pred_plane'].numpy(), 
                    conf_threshold=0.7)
            if gt_plane:
                plane_locals[str(i)] = [ann['plane'] for ann in self.dataset_dict[idx][str(i)]['annotations']]
            else:
                if plane_param_override is None:
                    plane_locals[str(i)] = p_instances[str(i)].pred_planes
                else:
                    plane_locals[str(i)] = plane_param_override[str(i)]
                    
        # Merge planes if they are in correspondence
        if len(corr_list) != 0:
            plane_locals = merge_plane_params_from_local_params(plane_locals, corr_list, pred_camera)

        os.makedirs(output_dir, exist_ok=True)
        for i in range(self.num_view):
            img_file = file_names[i]
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)[:,:,::-1]

            # save original images
            # imageio.imwrite(os.path.join(output_dir, prefix + f'_view_{i}.png'), img)

            height, width, _ = img.shape
            vis = Visualizer(img)
            if tran_topk == -2 and rot_topk == -2:
                camera_info = pred_camera[i]
            elif tran_topk == -1 and rot_topk == -1:
                camera_info = self.dataset_dict[idx][str(i)]['camera']
                camera_info = get_relative_pose_from_datapoint(camera_info, self.dataset_dict[idx]['0']['camera'])
                camera_info['rotation'] = quaternion.from_float_array(camera_info['rotation'])
                camera_info['position'] = np.array(camera_info['position'])
            else:
                raise NotImplementedError
            if not gt_plane or not gt_segm:
                p_instance = p_instances[str(i)]            
            plane_params = plane_locals[str(i)]
            if gt_segm:
                seg_blended = get_gt_labeled_seg(self.dataset_dict[idx][str(i)], vis, paper_img=True)
                segmentations = [ann['segmentation'] for ann in self.dataset_dict[idx][str(i)]['annotations']]
                #cv2.imwrite(os.path.join(output_dir, prefix + f'_{i}_gtseg.png'), seg_blended)
            else:
                seg_blended = get_labeled_seg(p_instance, 0.7, vis, paper_img=True)
                segmentations = p_instance.pred_masks
                #cv2.imwrite(os.path.join(output_dir, prefix + f'_{i}_predseg.png'), seg_blended)
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



class PlaneFormerInferenceVisualization():
    def __init__(self, img_list):
        """
        Cache:
        - Predicted camera poses
        - Predicted plane parameters
        - Predicted embedding distance matrix
        
        """
        self.score_threshold = 0.7
        self.img_list = img_list
        self.num_view = len(img_list)

    def save_matching(self, rcnn_output, assignment, output_dir, prefix='', fp=True, paper_img=False, vertical=False):
        """
        fp: whether show fp or fn
        """
        blended = {}
        basenames = {}
        uniq_idx = 0
        # centroids for matching
        centroids = defaultdict(list)
        for i in range(self.num_view):
            img_file = self.img_list[i]
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)[:,:,::-1]
            img = cv2.resize(img, (640, 480))
            height, width, _ = img.shape
            vis = Visualizer(img)
            p_instance = create_instances(rcnn_output[str(i)]['instances'], img.shape[:2], 
                                            pred_planes=rcnn_output[str(i)]['pred_plane'].numpy(), 
                                            conf_threshold=self.score_threshold)
            seg_blended = get_labeled_seg(p_instance, self.score_threshold, vis, paper_img=paper_img)
            blended[str(i)] = seg_blended
            # centroid of mask
            for ann in rcnn_output[str(i)]['instances']:
                M=center_of_mass(mask_util.decode(ann['segmentation']))
                centroids[str(i)].append(M[::-1]) # reverse for opencv
            centroids[str(i)] = np.array(centroids[str(i)])

        for pair in combinations(np.arange(self.num_view), 2):
            pred_corr_list =  np.array(assignment)[:, [pair[0], pair[1]]]
            pred_corr_list_filtered = []
            for corr in pred_corr_list:
                if -1 in corr:
                    continue
                pred_corr_list_filtered.append(corr)
            pred_matching_fig = draw_match(blended[str(pair[0])], blended[str(pair[1])], centroids[str(pair[0])], centroids[str(pair[1])], 
                                            np.array(pred_corr_list_filtered), np.ones(len(pred_corr_list_filtered)), vertical=vertical)
            os.makedirs(output_dir, exist_ok=True)
            pred_matching_fig.save(os.path.join(output_dir, prefix+f'_{pair[0]}_{pair[1]}.png'))

    def save_input(self, output_dir, prefix):
        for i in range(self.num_view):
            img_file = self.img_list[i]
            shutil.copy(img_file, os.path.join(output_dir, f'{prefix}_{str(i)}.png'))
            os.chmod(os.path.join(output_dir, f'{prefix}_{str(i)}.png'), 0o755)

    def save_multiple_objects(self, tran_topk, rot_topk, output_dir, rcnn_output, prefix='', \
        pred_camera=None, show_camera=False, corr_list=[], webvis=False):
        """
        if tran_topk == -2 and rot_topk == -2, then pred_camera should not be None, this is used for non-binned camera.
        if exclude is not None, exclude some instances to make fig 2.
        idx=7867
        exclude = {
            '0': [2,3,4,5,6,7],
            '1': [0,1,2,4,5,6,7],
        }
        """
        
        file_names = [self.img_list[i] for i in range(self.num_view)]
        basenames = {}
        meshes_list = []
        #map_files = []
        uv_maps = []
        cam_list = []
        vis_idx = 0
        # get plane parameters
        plane_locals = {}
        p_instances = {}
        for i in range(self.num_view):
            p_instances[str(i)] = create_instances(rcnn_output[str(i)]['instances'], [480, 640], 
                pred_planes=rcnn_output[str(i)]['pred_plane'].numpy(), 
                conf_threshold=0.7)
            plane_locals[str(i)] = p_instances[str(i)].pred_planes
                    
        # Merge planes if they are in correspondence
        if len(corr_list) != 0:
            plane_locals = merge_plane_params_from_local_params(plane_locals, corr_list, pred_camera)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'resized'), exist_ok=True)
        for i in range(self.num_view):
            img_file = file_names[i]
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (640, 480))

            # saving resized images
            img_file = f'{os.path.join(output_dir, "resized", str(i))}.png'
            cv2.imwrite(img_file, img)

            img = img[:,:,::-1]

            # save original images
            # imageio.imwrite(os.path.join(output_dir, prefix + f'_view_{i}.png'), img)

            height, width, _ = img.shape
            vis = Visualizer(img)
            if tran_topk == -2 and rot_topk == -2:
                camera_info = pred_camera[i]

            p_instance = p_instances[str(i)]            
            plane_params = plane_locals[str(i)]
            
            seg_blended = get_labeled_seg(p_instance, 0.7, vis, paper_img=True)
            segmentations = p_instance.pred_masks
            #cv2.imwrite(os.path.join(output_dir, prefix + f'_{i}_predseg.png'), seg_blended)

            meshes, uv_map = get_single_image_mesh_plane(plane_params, segmentations, img_file=img_file, 
                            height=height, width=width, webvis=False, tolerance=0)
            uv_maps.extend(uv_map)
            meshes = transform_meshes(meshes, camera_info)
            meshes_list.append(meshes)
            cam_list.append(camera_info)
            
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

        if len(prefix) == 0:
            prefix = key+'_pred'
        save_obj(folder=output_dir, prefix=prefix, meshes=joint_mesh, cam_meshes=cam_meshes, decimal_places=10, blend_flag=True, map_files=None, uv_maps=uv_maps)

        shutil.rmtree(os.path.join(output_dir, 'resized'), ignore_errors=True)


def main(args):
    num_view = args.num_view
    phase = args.phase
    pkl_file = args.pkl
    dataset_json = f'/Pool1/users/jinlinyi/workspace/multi-surface-recon/multiview_dataset/multiview_set{num_view}_v2/{phase}.json'
    with open(pkl_file, 'rb') as f:
        data= pickle.load(f)
    np.random.seed(2022)
    if args.num_data != -1:
        idx_list = np.random.choice(sorted(list(data.keys())), args.num_data, replace=False)
    else:
        idx_list = sorted(list(data.keys()))
    visualizer = PlaneFormerMVResult(0.7, dataset_json, num_view)

    for idx in idx_list:
        save_folder = os.path.join(args.save_folder, f'set_{num_view}', str(idx).zfill(4))
        os.makedirs(save_folder, exist_ok=True)
        
        prefix = '00_original'
        visualizer.save_input(idx, save_folder, prefix)
        cv2.imwrite(os.path.join(save_folder, '00_placeholder2.png'), np.zeros((10,10,3))) 
        cv2.imwrite(os.path.join(save_folder, '00_placeholder3.png'), np.zeros((10,10,3))) 
        # Save our predictions
        visualizer.save_multiple_objects(idx=idx, tran_topk=-2, rot_topk=-2, output_dir=save_folder,
            rcnn_output=data[idx][0],
            prefix='ours', 
            gt_plane=False, pred_camera=data[idx][1], gt_segm=False, 
            plane_param_override=None, show_camera=True, corr_list=data[idx][3], 
            exclude=None, webvis=False)
        # save predicted plane matching
        visualizer.save_matching(idx=idx, rcnn_output=data[idx][0], assignment=data[idx][3], output_dir=os.path.join(save_folder, 'corr'), prefix='ours', 
        paper_img=True, vertical=False)
        # save GT.
        visualizer.save_multiple_objects(idx=idx, tran_topk=-1, rot_topk=-1, output_dir=save_folder, 
            rcnn_output=None,
            prefix='gt', 
            gt_plane=True, pred_camera=None, gt_segm=True, 
            plane_param_override=None, show_camera=True, corr_list=[], 
            exclude=None, webvis=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="A script that optimize objective functions from top k proposals."
    )
    parser.add_argument("--save-folder", required=True, help="output directory")
    parser.add_argument("--num-data", default=-1, type=int, help="number of data to process, if -1 then all.")
    parser.add_argument("--phase", type=str, help="test/val")
    parser.add_argument("--num-view", type=int, help="3/5")
    parser.add_argument("--pkl", type=str, help="result pkl")
    args = parser.parse_args()
    print(args)
    main(args)