import os
import numpy as np
import pickle
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

import shutil
from scipy import ndimage
sns.set()

suncg_dir = '/x/syqian/suncg/renderings_ldr'
detection_bbox = '/x/syqian/suncg/faster_rcnn_proposals'
split_file = '/Pool1/users/jinlinyi/workspace_share/associative3D/script/v3_rpnet_split_relative3d/v3_test_subset_MGT0.txt'
result_pkl = '/x/syqian/eccv2020voxels_eval/full_scene_eval/ours_v3_test_subset'

output_folder = './supp/supp_stitching'

cmap =  [[79,129,189],
        [155,187,89],
        [128,100,162],
        [192,80,77],
        [75,172,198],
        [31,73,125],
        [247,150,70],
        [238,236,225],]

reds = [
    [242,220,219],
    [230,184,183],
    [218,150,148],
    [150,54,52],
    [99,37,35],
]

oranges = [
    [253,233,217],
    [252,213,180],
    [250,191,143],
    [247,150,70],
    [226,107,10],
]

purples = [
    [204, 192, 218], 
    [176, 163, 190], 
    [148, 134, 163], 
    [120, 106, 135], 
    [64, 49, 80]
]

sns_color = [(np.array(c)*255).astype(int) for c in sns.color_palette("OrRd", 5)]


def save_affinity_after_stitch(affinity_pred, sz_i, sz_j, matching, mesh_dir):
    try:
        max_sz = max(sz_i, sz_j)
        if max_sz < 5:
            max_sz = 5
        elif max_sz < 10:
            max_sz = 10
        text = np.array([['']*sz_j]*sz_i)
        for i,j in enumerate(matching):
            if j != -1:
                text[i][j] = '*'
        affinity_vis = affinity_pred[:max_sz, :max_sz]
        labels = (np.asarray(["{}\n{:.2f}".format(text,data) for text, data in zip(text.flatten(), affinity_pred[:sz_i, :sz_j].flatten())])).reshape(text.shape)
        labels_full = np.array([['']*max_sz]*max_sz).astype('<U6')
        labels_full[:sz_i,:sz_j] = labels
        plt.figure()
        sns.heatmap(affinity_vis, annot=labels_full, fmt='s', vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, 'affinity_pred.png'))
        plt.close()
    except:
        plt.figure()
        sns.heatmap(affinity_pred[:max_sz, :max_sz], vmin=0.0, vmax=1.0)
        plt.savefig(os.path.join(mesh_dir, 'affinity_pred.png'))
        plt.close()
        pass


def save_aff(aff, save_path):
    pass


def get_loc_white(bbox):
    x1,y1,x2,y2 = bbox
    return [x1+4, y1+4, x2-4, y2-4]

def load_img(mat_path, img_path):
    mat = loadmat(mat_path)
    bbox = mat['bboxes']
    im = Image.open(img_path)
    return im, bbox
    
def draw_bbox(img1, img2, bbox1, bbox2, matching_proposals):
    try:
        d1 = ImageDraw.Draw(img1)
        d2 = ImageDraw.Draw(img2)
        cmap_idx = 0
        for idx1, idx2 in enumerate(matching_proposals):
            if idx2 == -1:
                d1.rectangle(bbox1[idx1], fill=None, outline=(0,0,0), width=5)
            else:
                # import pdb;pdb.set_trace()
                d1.rectangle(bbox1[idx1], fill=None, outline=tuple(cmap[cmap_idx]), width=10)
                d1.rectangle(get_loc_white(bbox1[idx1]), fill=None, outline=(255,255,255), width=2)
                d2.rectangle(bbox2[idx2], fill=None, outline=tuple(cmap[cmap_idx]), width=10)
                d2.rectangle(get_loc_white(bbox2[idx2]), fill=None, outline=(255,255,255), width=2)
                cmap_idx += 1
        for idx, box in enumerate(bbox2):
            if idx not in matching_proposals:
                d2.rectangle(box, fill=None, outline=(0,0,0), width=5)
        return img1, img2
    except e:
        import pdb;pdb.set_trace()
        pass


def get_concat_v(im1, im2, distance = 50, vertical=True):
    if vertical:
        dst = Image.new('RGBA', (im1.width, im1.height + distance + im2.height), (255, 0, 0, 0))
        dst.paste(im2, (0, distance + im1.height))
    else:
        dst = Image.new('RGBA', (im1.width + distance + im2.width, im1.height), (255, 0, 0, 0))
        dst.paste(im2, (distance + im1.width, 0))
    dst.paste(im1, (0, 0))
    return dst


def get_centers(segmentation):
    """
    input: segmentation map, 0 is background
    output: center of mass of each segment
    """
    centers = []
    for i in range(int(np.max(segmentation))+1):
        if i == 0:
            continue
        elif i not in segmentation:
            centers.append(np.array([0,0]))
        else:
            mask = segmentation == i
            centers.append(np.array(ndimage.measurements.center_of_mass(mask))[::-1])
    return centers


def draw_dot(d, center, color, factor, dotsize=20):
    outer_offset = int(dotsize*factor)
    inner_offset = int(dotsize/20*16*factor)
    outer_bbox = (center[0]-outer_offset, center[1]-outer_offset, center[0]+outer_offset, center[1]+outer_offset) 
    inner_bbox = (center[0]-inner_offset, center[1]-inner_offset, center[0]+inner_offset, center[1]+inner_offset) 
    d.ellipse(outer_bbox, fill=tuple(color), outline=tuple(color), width=int(dotsize/20*5*factor))
    d.ellipse(inner_bbox, fill=None, outline=(255,255,255), width=int(dotsize/20*4*factor))


def draw_match(img1_path, img2_path, seg1, seg2, matching_proposals, gt_matching_proposals=None, pred_aff=None, before=True, th=0.5, distance = 45, factor=4, vertical=True, dotsize=20, outlier_color=None):
    # factor: resize the image before drawing, resume after finishing. This avoids artifacts and draw high resolution lines.
    if isinstance(img1_path, str) and isinstance(img2_path, str):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
    else:
        img1 = Image.fromarray(img1_path)
        img2 = Image.fromarray(img2_path)
    img1 = img1.resize((img1.width*factor, img1.height*factor))
    img2 = img2.resize((img2.width*factor, img2.height*factor))

    centers1 = get_centers(seg1)
    centers2 = get_centers(seg2)
    centers1 = [np.floor(center*factor).astype(np.int32) for center in centers1]
    centers2 = [np.floor(center*factor).astype(np.int32) for center in centers2]
    distance *= factor

    concat = get_concat_v(img1, img2, distance, vertical)
    d = ImageDraw.Draw(concat)
  
    draw_aff = np.zeros((len(centers1), len(centers2)))
    for [idx1, idx2] in matching_proposals:
        draw_aff[idx1][idx2] = 1
    draw_aff = draw_aff > 0.5

    gt_aff = np.zeros((len(centers1), len(centers2)))
    if gt_matching_proposals is None:
        gt_aff = gt_aff == 0
    else:
        for [idx1, idx2] in gt_matching_proposals:
            gt_aff[idx1][idx2] = 1
        gt_aff = gt_aff > 0.5

    if vertical:
        offset = distance + img1.height
    else:
        offset = distance + img1.width

    # Draw black bbox
    d.rectangle((0,0,img1.width,img1.height), fill=None, outline=(0,0,0), width=7*factor)
    if vertical:
        d.rectangle((0,offset,img2.width,img2.height+offset), fill=None, outline=(0,0,0), width=7*factor)
    else:
        d.rectangle((offset, 0, img2.width + offset, img2.height), fill=None, outline=(0,0,0), width=7*factor)
    # Plot black dots for non matching objs
    for i in range(draw_aff.shape[0]):
        if (draw_aff[i]==False).all():
            center = centers1[i]
            draw_dot(d, center, (0,0,0), factor, dotsize=dotsize)
    for j in range(draw_aff.shape[1]):
        if (draw_aff[:, j]==False).all():
            if vertical:
                center = centers2[j] + np.array([0, offset])
            else:
                center = centers2[j] + np.array([offset, 0])
            draw_dot(d, center, (0,0,0), factor, dotsize=dotsize)
    # Draw line
    color_it = 0
    for i in range(draw_aff.shape[0]):
        for j in range(draw_aff.shape[1]):
            if draw_aff[i][j]:
                if gt_aff[i][j]:
                    color = [0,176,80] # Green
                else:
                    if outlier_color is None:
                        color = [255,0,0] # Red
                    else:
                        color = outlier_color
                width_factor = 1
                center1 = centers1[i]
                if vertical:
                    line = (centers1[i][0], centers1[i][1], centers2[j][0], centers2[j][1] + offset)
                else:
                    line = (centers1[i][0], centers1[i][1], centers2[j][0] + offset, centers2[j][1])
                d.line(line, fill=tuple(color), width=int(7*width_factor*factor))
                d.line(line, fill=(255,255,255), width=int(2*width_factor*factor))
    
    # Plot colored dots for matching objs
    color_it = 0
    for i in range(draw_aff.shape[0]):
        if before:
            color = cmap[3]
        else:
            color = cmap[color_it%len(cmap)]
            color_it += 1
        for j in range(draw_aff.shape[1]):
            if draw_aff[i][j]:
                if before:
                    color = purples[-1]
                center1 = centers1[i]
                draw_dot(d, center1, color, factor, dotsize=dotsize)
                if vertical:
                    center2 = centers2[j] + np.array([0, offset])
                else:
                    center2 = centers2[j] + np.array([offset, 0])
                draw_dot(d, list(center2), color, factor, dotsize=dotsize)
    # concat = concat.resize((int(concat.width/factor), int(concat.height/factor)))
    return concat
    

def main():
    np.random.seed(2019)
    os.makedirs(output_folder, exist_ok=True)
    f = open(split_file, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[3:]
    # lines = np.random.choice(lines, 200)
    for line in tqdm(lines):
        try:
            # parse line
            splits = line.split(' ')
            img1_path = splits[0]
            img2_path = splits[8]
            house_name = img1_path.split('/')[-2]
            img1_name = img1_path.split('/')[-1]
            img2_name = img2_path.split('/')[-1]
            view1 = img1_name[:6]
            view2 = img2_name[:6]
            pair_id = house_name + '_' + view1 + '_' + view2
            if pair_id not in [
                '555967b7c2d57053671b86559db73aea_000001_000010',
                '5613748e9fe18fcccc38606be66f0ecf_000004_000008',
                '56fe2162c8d58cc2f90bbea9bf26b4ac_000016_000020',
                '676f515f0c2ba4e755b34525e6e8b448_000004_000003',
                '705063a254c63c203d623171ccd1a046_000035_000034',
                '7ef31013b98f53932e90b02a3f735f6c_000034_000020',
                '809cf0061eca9653c21911a3949d8d7e_000005_000007',
                '8120d2261436dfc5045efec82b369054_000008_000011',
                '876d8fd0ad2cf326533e9b8f16db798b_000012_000014',
            ]:
                continue
            with open(os.path.join(result_pkl, pair_id, 'results.pkl'), "rb") as f:
                data = pickle.load(f)
            matching_proposals = data['matching_pred']
            pred_aff = data['affinity_pred']
            img1, bbox1 = load_img(mat_path=os.path.join(detection_bbox, house_name, view1 + '_proposals.mat'), 
                    img_path=os.path.join(suncg_dir, house_name, view1+'_mlt.png'))
            img2, bbox2 = load_img(mat_path=os.path.join(detection_bbox, house_name, view2 + '_proposals.mat'), 
                    img_path=os.path.join(suncg_dir, house_name, view2+'_mlt.png'))
            # save_aff(pred_aff, os.path.join(output_folder, pair_id))
            os.makedirs(os.path.join(output_folder, pair_id), exist_ok=True)
            # save_affinity_after_stitch(pred_aff, len(bbox1), len(bbox2), matching_proposals, os.path.join(output_folder, pair_id))
            # img1, img2 = draw_bbox(img1, img2, bbox1, bbox2, matching_proposals)
            concat_before = draw_match(img1, img2, bbox1, bbox2, matching_proposals, pred_aff, vertical=True)
            concat_after = draw_match(img1, img2, bbox1, bbox2, matching_proposals, pred_aff, before=False, vertical=True)

            concat_before.save(os.path.join(output_folder, pair_id, 'match_before.png'))
            concat_after.save(os.path.join(output_folder, pair_id, 'match_after.png'))
            # img1.save(os.path.join(output_folder, pair_id, 'left.png'))
            # img2.save(os.path.join(output_folder, pair_id, 'right.png'))
        except:
            shutil.rmtree(os.path.join(output_folder, pair_id))
            continue

if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    # sns.palplot(sns.color_palette("YlOrRd", 5))
    # plt.savefig('debug.png')
    # exit()
    main()