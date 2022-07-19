import os
import cv2
import git
import random
import sys
import numpy as np
import quaternion
import scipy.ndimage as ndimage
from glob import glob
from PIL import Image
from distutils.version import LooseVersion
from itertools import combinations
from datetime import datetime

from tqdm import tqdm
from sacred import Experiment
from easydict import EasyDict as edict

import pickle
import json
import torch
from detectron2.structures import BoxMode
from pytorch3d.structures import join_meshes_as_batch

from utils.disp import colors_256 as colors
import utils.draw_matching as dm
from PIL import Image

from pycococreatortools.pycococreatortools import create_annotation_info

ex = Experiment()
# PHASE = 'val'
# MAX_SET_PER_LEVEL = 1
# SET_SIZE = 3
# MAX_N_NCR = 100
# DEBUG = True

# PHASE = 'test'
# MAX_SET_PER_LEVEL = 10
# SET_SIZE = 3
# MAX_N_NCR = 100
# DEBUG = True


# PHASE = 'val'
# MAX_SET_PER_LEVEL = 1
# SET_SIZE = 5
# MAX_N_NCR = 20
# DEBUG = True

PHASE = 'test'
MAX_SET_PER_LEVEL = 10
SET_SIZE = 5
MAX_N_NCR = 30
DEBUG = True

# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass

class BrokenSemanticInfo(Error):
   """Raised when the input value is too small"""
   pass



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
    relative_rotation = quaternion.as_float_array(relative_rotation).tolist()
    relative_translation = relative_translation.tolist()
    rel_pose = {'position': relative_translation, 'rotation': relative_rotation}
    return rel_pose

def get_inv_intrinsics():
    # define K for PlaneNet dataset
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))
    return K_inv


def get_nms_aff(offsets_list, normals_list, instance_list):
    total_idx = len(offsets_list)
    assert(total_idx==2)
    idx1 = 0
    idx2 = 1
    offset_list1 = offsets_list[idx1]
    offset_list2 = offsets_list[idx2]
    normal_list1 = normals_list[idx1]
    normal_list2 = normals_list[idx2]

    offset_aff = offset_list1[None].T-offset_list2[None]
    normal_aff = normal_list1@normal_list2.T
    assert(offset_aff.shape == normal_aff.shape)

    instance_aff = instance_list[idx1][None].T == instance_list[idx2][None]
    return offset_aff, normal_aff, instance_aff



def find_match(semantic_list):
    assert(len(semantic_list) == 2)
    instance_0 = np.array([semantic_list[0][key]['plane_instance_id'] for key in sorted(semantic_list[0].keys())])[None]
    instance_1 = np.array([semantic_list[1][key]['plane_instance_id'] for key in sorted(semantic_list[1].keys())])[None]
    affinity = instance_0.T == instance_1
    matching_proposals = np.argwhere(affinity)
    # print(matching_proposals)
    return matching_proposals


def get_stiching_dict(images):
    stitch_dict = {}
    for image in images:
        with open(image, 'rb') as f:
            data = pickle.load(f)
        if not data['good_quality']:
            continue
        # filter large non plane image:
        # planeSegmentation = data['planeSegmentation']
        # bkgd_idx = data['backgroundidx']
        # if np.sum(planeSegmentation == bkgd_idx) > 0.5*len(planeSegmentation.flatten()):
        #     continue
        floor, room, view = image.split('/')[-1].split('.')[0].split('_')
        if floor in stitch_dict.keys():
            stitch_dict[floor].append(image)
        else:
            stitch_dict[floor] = [image]
    return stitch_dict


def plot_category(blend_pred, planeSegmentation, semantic_info, bkgd=0):
    
    def get_centers(segmentation,bkgd):
        """
        input: segmentation map, 20 is background
        output: center of mass of each segment
        """
        centers = []
        for i in sorted(np.unique(segmentation)):
            if i == bkgd:
                continue
            elif i not in segmentation:
                centers.append(np.array([0,0]))
            else:
                mask = segmentation == i
                centers.append(np.array(ndimage.measurements.center_of_mass(mask))[::-1])
        return centers
    centers = get_centers(planeSegmentation, bkgd)
    for sem, center in zip(semantic_info, centers):
        cv2.putText(blend_pred, sem['category_name']+ sem['full_id'], tuple(center.astype(np.int32)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    return blend_pred 


def get_camera_dicts(camera_pickle):
    """
    key: image id. 0_0_0
    value: {save_path': '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v3/rgb/17DRP5sb8fy/0_0_0.png', 'img_name': '0_0_0.png', 'house_idx': '17DRP5sb8fy', 'position': array([-9.56018401,  1.577794  , -2.63851021]), 'rotation': quaternion(0.984984462909517, -0.0948432164229211, 0.143593870170991, 0.0138265170857656)}
    """
    with open(camera_pickle, 'rb') as f:
        cameras_list = pickle.load(f)
    cameras_dict = {}
    for c in cameras_list:
        cameras_dict[c['img_name'].split('.')[0]] = c
    return cameras_dict


def erode_seg(segmentation, size=5, non_plane=20):
    for i in np.unique(segmentation):
        if i == non_plane:
            continue
        mask = segmentation == i
        mask_erode = cv2.erode(mask.astype(np.float32), np.ones((size, size))) > 0.5
        segmentation[np.logical_xor(mask_erode, mask)] = non_plane
    return segmentation


def segmap2instmap(planeSegmentation, semantic_info, background=20):
    """
    input:  segmentation map
            semantic_info: [{'category_id': 4, 'category_name': 'door', 'full_id': '0_0_3'}]
    output: switch segmentation to instance color id, 0 is background
    """
    idMasks = np.zeros(planeSegmentation.shape).astype(np.int32)
    # planeSegmentation = erode_seg(planeSegmentation)
    bkgd_mask = planeSegmentation==background
    for idx, info in semantic_info.items():
        instance_id = info['plane_instance_id']
        idMasks[planeSegmentation==idx] = instance_id%len(colors)
    # convert to 3-channel map
    h, w = planeSegmentation.shape
    idMasks = cv2.resize(np.stack([colors[idMasks, 0],
                                              colors[idMasks, 1],
                                              colors[idMasks, 2]], axis=2), (w, h))
    return idMasks, bkgd_mask   


def get_info(phase):
    """
    return information of the json file, including dataset name, git hexsha, date created.
    """
    description = f'MP3D {phase} Dataset, multiview of SET_SIZE = {SET_SIZE}, MAX_SET_PER_LEVEL = {MAX_SET_PER_LEVEL}'
    repo = git.Repo(search_parent_directories=True)
    git_hexsha = repo.head.object.hexsha
    date_created = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rtn = {
        'info': {
            'description': description,
            'git_hexsha': git_hexsha,
            'date_created': date_created
        },
        'categories': [
            {'id': 0, 'name': 'plane'},
        ]
    }
    return rtn


def get_extendable_class_list():
    return [
        'wall', 
        'floor',
    ]
    

def filter_small_planes(planes, semantic_info, planeSegmentation, bkgd_idx=0, area_threshold=0.01):
    """
    input:  
    @planes: plane parameters
    @semantic_info: {1: {'plane_instance_id': 1}, 2: {'plane_instance_id': 32}}]
    @planeSegmentation: segmentation mask, each entry is semantic index (semantic_info.keys()+bkgd_idx)
    output: filtered planes whose projected area is larger than 1% of an image
    """
    assert(bkgd_idx == 0)
    threshold = area_threshold*len(planeSegmentation.flatten())
    rtn_planes = []
    rtn_semantic_info = {}
    rtn_planeSegmentation = np.ones(planeSegmentation.shape)*bkgd_idx
    good_idx = 1
    for i in semantic_info.keys():
        area = np.sum(planeSegmentation == i)
        if area >= threshold:
            rtn_planes.append(planes[i-1])
            rtn_semantic_info[good_idx] = semantic_info[i]
            rtn_planeSegmentation[planeSegmentation == i] = good_idx
            good_idx += 1
    return rtn_planes, rtn_semantic_info, rtn_planeSegmentation


def mp3d2habitat(planes):
    rotation = np.array([
        [1,0,0],
        [0,0,-1],
        [0,1,0],
    ])
    rotation = np.linalg.inv(rotation)
    return (rotation@np.array(planes).T).T


def is_valid_pair(pair):
    semantic_list = []
    num_planes = []
    for img_pickle in pair:
        # with open(img_pickle.replace('planes_from_ply','planes_from_ply_small'), 'rb') as f:
        with open(img_pickle, 'rb') as f:
            plane_summary = pickle.load(f)
        num_planes.append(plane_summary['numPlanes'])
        semantic_info = plane_summary['semantic_info']
        semantic_list.append(semantic_info)
    matching_proposals = find_match(semantic_list)
    if len(matching_proposals) <=2:
        return False
    # filter pairs where one is a subset of the other
    if len(matching_proposals) >= num_planes[0] - 2 or len(matching_proposals) >= num_planes[1] - 2:
        return False
    return True

def is_valid_set(img_set):
    pairs = np.array(list(combinations(img_set, 2)))
    for pair in pairs:
        if not is_valid_pair(pair):
            return False
    return True


def visualize_debug(img_set, cfg, house, cameras_dict, set_id):
    pairs = np.array(list(combinations(img_set, 2)))
    for pair in pairs:
        semantic_list, blend_debugs, segmentation_debugs = [], [], []
        meshes_list, cam_list = [], []
        
        bboxs_list = []
        datapoint = {0: {}, 1: {}}
        for pair_id, img_pickle in enumerate(pair):
            # Load single image prediciton
            img_id = img_pickle.split('/')[-1].split('.')[0]
            filename = os.path.join(cfg.rgb_path, house, img_id+'.png')
            height, width = cv2.imread(filename).shape[:2]

            datapoint[pair_id]['file_name'] = filename
            datapoint[pair_id]['image_id'] = house+'_'+img_id
            datapoint[pair_id]['height'] = height
            datapoint[pair_id]['width'] = width
            datapoint[pair_id]['camera'] = {'position': cameras_dict[img_id]['position'].tolist(), 
                                            'rotation': quaternion.as_float_array(cameras_dict[img_id]['rotation']).tolist()}
            
            with open(img_pickle, 'rb') as f:
                plane_summary = pickle.load(f)
            semantic_info = plane_summary['semantic_info']
            planeSegmentation = plane_summary['planeSegmentation']
            bkgd_idx = plane_summary['backgroundidx']
            
            if DEBUG:
                segmentation_debugs.append(planeSegmentation)
                img_path = os.path.join(cfg.image_path, house, img_id+'.pkl')
                with open(img_path, 'rb') as f:
                    observations = pickle.load(f)
                rgb = observations['color_sensor'][:,:,:3]
                planeSegmentation, bkgd_mask = segmap2instmap(planeSegmentation, semantic_info, bkgd_idx)
                blend_gt = (rgb * 0.3 + planeSegmentation * 0.7).astype(np.uint8)
                blend_gt[bkgd_mask] = rgb[bkgd_mask]
                blend_debugs.append(blend_gt)

            semantic_list.append(semantic_info)
        matching_proposals = find_match(semantic_list)
        save_path = os.path.join(cfg.debug_path, f"{str(set_id).zfill(4)}")
        os.makedirs(save_path, exist_ok=True)
        # save raw rgbs
        for img_pickle in img_set:
            img_id = img_pickle.split('/')[-1].split('.')[0]
            img_path = os.path.join(cfg.image_path, house, img_id+'.pkl')
            with open(img_path, 'rb') as f:
                observations = pickle.load(f)
            rgb = observations['color_sensor'][:,:,:3]
            cv2.imwrite(os.path.join(save_path, '0-' + img_id + '.jpg'), rgb[:,:,::-1])
        matching_fig = dm.draw_match(blend_debugs[0], blend_debugs[1], segmentation_debugs[0], segmentation_debugs[1], matching_proposals, vertical=False)
        matching_fig.save(os.path.join(save_path, f"1-{datapoint[0]['image_id']}__{datapoint[1]['image_id']}_match_gt.png"))
        


@ex.main
def stitch(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    dump_file_summary = {}
    dump_file_summary.update(get_info(PHASE))
    dump_file_summary['data'] = []
    print(f'[{PHASE}]')
    annotation_id = 0
    set_id = 0
    with open(os.path.join(cfg.dataset_split_path, f'scenes_{PHASE}.txt'), 'r') as f:
        houses = f.read().splitlines()
    for house in houses:
        plane_infos = sorted(list(glob(os.path.join(cfg.planeFit_path, house, 'planes_from_ply', '*.pkl'))))
        stitch_dict = get_stiching_dict(plane_infos)
        cameras_dict = get_camera_dicts(os.path.join(cfg.camera_path, house+'.pkl'))
        good_img_sets = []
        for level_id, img_list in stitch_dict.items():
            if len(img_list) <= 1:
                continue
            if len(img_list) >= MAX_N_NCR:
                img_list = np.random.choice(img_list, MAX_N_NCR, replace=False)
            img_sets = np.array(list(combinations(img_list, SET_SIZE)))
            np.random.shuffle(img_sets)
            level_good_img_sets = []
            loop = tqdm(img_sets)
            for img_set in loop:
                if is_valid_set(img_set):
                    level_good_img_sets.append({
                        'img_set': img_set,
                    })
                loop.set_description(f'{house} - L{level_id}: {len(level_good_img_sets)}/{MAX_SET_PER_LEVEL}')
                if len(level_good_img_sets) >= MAX_SET_PER_LEVEL:
                    break
            good_img_sets.extend(level_good_img_sets)
        print(f"{house} # good sets {len(good_img_sets)}")
        if len(good_img_sets) == 0:
            continue
        for good_img_set in good_img_sets:
            img_set = good_img_set['img_set']

            if DEBUG:
                visualize_debug(img_set, cfg, house, cameras_dict, set_id)
                set_id += 1

            datapoint = {}

            for idx, img_pickle in enumerate(img_set):
                datapoint[idx] = {}
                # Load single image prediciton
                img_id = img_pickle.split('/')[-1].split('.')[0]
                filename = os.path.join(cfg.rgb_path, house, img_id+'.png')
                height, width = cv2.imread(filename).shape[:2]

                datapoint[idx]['file_name'] = filename
                datapoint[idx]['image_id'] = house+'_'+img_id
                datapoint[idx]['height'] = height
                datapoint[idx]['width'] = width
                datapoint[idx]['camera'] = {'position': cameras_dict[img_id]['position'].tolist(), 
                                                'rotation': quaternion.as_float_array(cameras_dict[img_id]['rotation']).tolist()}
                
                
                with open(img_pickle, 'rb') as f:
                    plane_summary = pickle.load(f)
                planes = plane_summary['planes'][:plane_summary['numPlanes']]
                semantic_info = plane_summary['semantic_info']
                planeSegmentation = plane_summary['planeSegmentation']
                bkgd_idx = plane_summary['backgroundidx']

                # FIXME!!! Uncomment if planes are in Matterport3D frame, we need to convert them in AI Habitat frame
                from sparseplane.utils.mesh_utils import save_obj, get_camera_meshes, transform_meshes, get_plane_params_in_global, get_plane_params_in_local
                glob_planes = get_plane_params_in_global(planes, cameras_dict[img_id])
                glob_planes = mp3d2habitat(glob_planes)
                planes = get_plane_params_in_local(glob_planes, cameras_dict[img_id])

                datapoint[idx]['plane_instance_id'] = [int(semantic_info[key]['plane_instance_id']) for key in sorted(semantic_info.keys())]

                annots = []
                for i in semantic_info.keys():
                    binary_mask = planeSegmentation == i
                    category_info = {'is_crowd':0, 'id':0}
                    annot = create_annotation_info(annotation_id, datapoint[idx]['image_id'], category_info, binary_mask)
                    if annot is None:
                        tqdm.write(datapoint[idx]['image_id']+' annotation has zero area mask')
                        import pdb;pdb.set_trace()
                        continue
                    annotation_id += 1
                    annot['bbox_mode'] = BoxMode.XYWH_ABS.value
                    annot["plane"] = planes[i-1].tolist()
                    annots.append(annot)
                datapoint[idx]["annotations"] = annots
            dump_file_summary['data'].append(datapoint)
    print(f"{PHASE}: {len(dump_file_summary['data'])}")
    with open(os.path.join(cfg.output_path, f"{PHASE}.json"), 'w') as f:
        json.dump(dump_file_summary, f)
""""
            set_id += 1
            pairs = np.array(list(combinations(img_set, 2)))
            for pair in pairs:
                try:
                    semantic_list, blend_debugs, segmentation_debugs = [], [], []
                    meshes_list, cam_list = [], []
                    
                    bboxs_list = []
                    datapoint = {0: {}, 1: {}}
                    for pair_id, img_pickle in enumerate(pair):
                        # Load single image prediciton
                        img_id = img_pickle.split('/')[-1].split('.')[0]
                        filename = os.path.join(cfg.rgb_path, house, img_id+'.png')
                        height, width = cv2.imread(filename).shape[:2]

                        datapoint[pair_id]['file_name'] = filename
                        datapoint[pair_id]['image_id'] = house+'_'+img_id
                        datapoint[pair_id]['height'] = height
                        datapoint[pair_id]['width'] = width
                        datapoint[pair_id]['camera'] = {'position': cameras_dict[img_id]['position'].tolist(), 
                                                        'rotation': quaternion.as_float_array(cameras_dict[img_id]['rotation']).tolist()}
                        
                        with open(img_pickle, 'rb') as f:
                            plane_summary = pickle.load(f)
                        planes = plane_summary['planes'][:plane_summary['numPlanes']]
                        semantic_info = plane_summary['semantic_info']
                        planeSegmentation = plane_summary['planeSegmentation']
                        bkgd_idx = plane_summary['backgroundidx']

                        annots = []
                        for i in semantic_info.keys():
                            binary_mask = planeSegmentation == i
                            category_info = {'is_crowd':0, 'id':0}
                            annot = create_annotation_info(annotation_id, datapoint[pair_id]['image_id'], category_info, binary_mask)
                            if annot is None:
                                tqdm.write(datapoint[pair_id]['image_id']+' annotation has zero area mask')
                                import pdb;pdb.set_trace()
                                continue
                            annotation_id += 1
                            annot['bbox_mode'] = BoxMode.XYWH_ABS.value
                            annot["plane"] = planes[i-1].tolist()
                            annots.append(annot)
                        datapoint[pair_id]["annotations"] = annots
                        
                        if DEBUG:
                            segmentation_debugs.append(planeSegmentation)
                            img_path = os.path.join(cfg.image_path, house, img_id+'.pkl')
                            with open(img_path, 'rb') as f:
                                observations = pickle.load(f)
                            rgb = observations['color_sensor'][:,:,:3]
                            planeSegmentation, bkgd_mask = segmap2instmap(planeSegmentation, semantic_info, bkgd_idx)
                            blend_gt = (rgb * 0.3 + planeSegmentation * 0.7).astype(np.uint8)
                            blend_gt[bkgd_mask] = rgb[bkgd_mask]

                            # blend_gt = plot_category(blend_gt, seg_0_bkgd, semantic_info)
                            blend_debugs.append(blend_gt)
                            # import pdb;pdb.set_trace()
                            # cv2.imwrite(f'debug_{img_id}.png',cv2.cvtColor(blend_gt, cv2.COLOR_RGB2BGR))
                            # np.unique(planeSegmentation.reshape(-1,3),axis=0)

                        semantic_list.append(semantic_info)
                    matching_proposals = find_match(semantic_list) 
                    # only save pairs with gt matchings >= 3.
                    # if len(matching_proposals) <=2:
                    #     continue
                    # # filter pairs where one is a subset of the other
                    # if len(matching_proposals) == len(datapoint[0]["annotations"]) or len(matching_proposals) == len(datapoint[1]["annotations"]):
                    #     continue
                    if DEBUG:
                        save_path = os.path.join(cfg.debug_path, f"{str(set_id).zfill(4)}")
                        os.makedirs(save_path, exist_ok=True)
                        # save raw rgbs
                        for img_pickle in img_set:
                            img_id = img_pickle.split('/')[-1].split('.')[0]
                            img_path = os.path.join(cfg.image_path, house, img_id+'.pkl')
                            with open(img_path, 'rb') as f:
                                observations = pickle.load(f)
                            rgb = observations['color_sensor'][:,:,:3]
                            cv2.imwrite(os.path.join(save_path, '0-' + img_id + '.jpg'), rgb[:,:,::-1])
                        matching_fig = dm.draw_match(blend_debugs[0], blend_debugs[1], segmentation_debugs[0], segmentation_debugs[1], matching_proposals, vertical=False)
                        matching_fig.save(os.path.join(save_path, f"1-{datapoint[0]['image_id']}__{datapoint[1]['image_id']}_match_gt.png"))
                
                
                except BrokenSemanticInfo:
                    print(f"{house}/{level_id} len(semantic_info)!=numPlane")
                    continue
                """


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'
    ex.add_config('config.yaml')
    ex.run_commandline()