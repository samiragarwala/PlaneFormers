from audioop import avg
from planeformers.utils.misc import *
import json
import pickle
import argparse
import os
from types import SimpleNamespace
from tqdm import tqdm


# Creates json file that will be used in evaluating sparseplanes in the multiview case
def create_json(params):

    with open(params.mp3d_corr_file, 'rb') as f:
        mp3d_corr_file = pickle.load(f)
    with open(params.multi_view_file, 'r') as f:
        multi_view_data = json.load(f)['data']

    mp3d_data = []
    phase = ['train', 'val', 'test']
    for i in range(len(phase)):
        with open(os.path.join(params.mp3d_json_path, 'cached_set_%s.json' % (phase[i])), 'r') as f:
            mp3d_data.append(json.load(f))

    mv = MultiViewInference(params, params.transformer_ckpt)
    ignore_idx = []
    edge_dict = {}

    eval_json_file = {}
    eval_json_file['data'] = []
    
    idx = 0

    for i in tqdm(range(len(multi_view_data))):

        output_preds = {}
        img_paths = []
        skip_sample = False

        for j in range(params.num_images):
            filename = multi_view_data[i][str(j)]['file_name']
            if mp3d_corr_file[filename] == None:
                skip_sample = True
                break

            img_paths.append(filename)
            tmp = mp3d_corr_file[filename]
            if tmp[0] == 0:
                path = "train"
            elif tmp[0] == 1:
                path = "val"
            else:
                path = "test"

            if params.use_gt_box:
                with open(os.path.join(params.path, path, str(tmp[1]) + '.pkl'), 'rb') as f:
                    view_ft = pickle.load(f)[tmp[2]]
                output_preds[str(j)] = view_ft
                output_preds[str(j)]['embedding'] = torch.tensor(output_preds[str(j)]['embedding'])
                output_preds[str(j)]['pred_plane'] = torch.tensor(output_preds[str(j)]['pred_plane'])

        if skip_sample:
            ignore_idx.append(i)
            continue
        
        if params.use_gt_box:
            features = output_preds
        else:
            features = mv.predict_features(img_paths)

        num_images = params.num_images
        features['num_images'] = num_images
        edges = mv.build_connectivity_graph(features)

        for e in range(edges.shape[0]):
            sample = {}

            tmp0 = mp3d_corr_file[img_paths[edges[e, 0]]]
            sample['0'] = mp3d_data[tmp0[0]]['data'][tmp0[1]][tmp0[2]]

            tmp1 = mp3d_corr_file[img_paths[edges[e, 1]]]
            sample['1'] = mp3d_data[tmp1[0]]['data'][tmp1[1]][tmp1[2]]

            gt_datapoint = [multi_view_data[i][str(edges[e, 0])], multi_view_data[i][str(edges[e, 1])]]
            gt_cam_transform = get_relative_pose_from_datapoint(gt_datapoint)
            gt_cam_transform['position'] = gt_cam_transform['position'].tolist()
            gt_cam_transform['rotation'] = quaternion.as_float_array(gt_cam_transform['rotation']).tolist()
            sample['rel_pose'] = gt_cam_transform
            
            gt_corr_list = []
            plane_instances_0 = multi_view_data[i][str(edges[e, 0])]['plane_instance_id']
            plane_instances_1 = multi_view_data[i][str(edges[e, 1])]['plane_instance_id']
            for idx0 in range(len(plane_instances_0)):
                for idx1 in range(len(plane_instances_1)):
                    if plane_instances_0[idx0] == plane_instances_1[idx1]:
                        gt_corr_list.append([idx0, idx1])
                        break
            sample['gt_corrs'] = gt_corr_list
            eval_json_file['data'].append(sample)
            idx += 1
        
        edge_dict[i] = edges.tolist()
    

    eval_json_file['edge_dict'] = edge_dict
    eval_json_file['ignore_idx'] = ignore_idx
    with open(params.output_file, 'w') as f:
        json.dump(eval_json_file, f)




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Transformer model for planar embedding correspondences')
    parser.add_argument('--mp3d_corr_file', help=".pkl file for matching up mp3d and multi-view dataset")
    parser.add_argument('--multi_view_file', help="Multi-view json file")
    parser.add_argument('--mp3d_json_path', help="mp3d json path")
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--use_gt_box', action='store_true')
    parser.add_argument('--transformer_ckpt', help='checkpoint for evaluation')
    parser.add_argument('--output_file')
    args = parser.parse_args()
    args.device = 'cuda'

    params = get_default_dataset_config("plane_params")
    params = SimpleNamespace(**vars(args), **vars(params))

    create_json(params)
