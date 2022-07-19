from audioop import avg
from planeformers.utils.misc import *
from planeformers.models.inference import *
import json
import pickle
import argparse
import os
from types import SimpleNamespace
from tqdm import tqdm

def build_plane_rcnn(params):
    cfg = get_cfg()
    get_sparseplane_cfg_defaults(cfg)
    cfg.merge_from_file(params.sparseplane_config)
    plane_rcnn_model = PlaneRCNN_Branch(cfg)
    return plane_rcnn_model

def merge_planes(num_images, edges, plane_list):
    merged_planes = []
    for edge_num in range(edges.shape[0]):

        corrs = plane_list[edge_num]
        for i in range(len(corrs[0])):
            found = False
            for j in range(len(merged_planes)):
                if merged_planes[j][edges[edge_num, 0]] == corrs[0][i]:
                    merged_planes[j][edges[edge_num, 1]] = corrs[1][i]
                    found = True
                    break
                if merged_planes[j][edges[edge_num, 1]] == corrs[1][i]:
                    merged_planes[j][edges[edge_num, 0]] = corrs[0][i]
                    found = True
                    break
            
            if not found:
                tmp = -np.ones((num_images,), dtype=np.intc)
                tmp[edges[edge_num, 0]] = corrs[0][i]
                tmp[edges[edge_num, 1]] = corrs[1][i]
                merged_planes.append(tmp)

    return merged_planes


# evaluate plane correspondence on gt box
def eval_multi_view(params):

    assert params.eval_corr or params.eval_camera


    if params.eval_corr:
        print("Evaluating plane correspondences")
    else:
        print("Evaluating camera")

    with open(params.mp3d_corr_file, 'rb') as f:
        mp3d_corr_file = pickle.load(f)
    with open(params.multi_view_file, 'r') as f:
        multi_view_data = json.load(f)['data']

    if params.run_sparseplane:
        with open(params.run_sparseplane, 'rb') as f:
            sparseplane_data = pickle.load(f)
        with open(params.edge_file, 'r') as f:
            edge_data = json.load(f)

    if params.eval_camera and params.run_baseline:
        with open(params.kmeans_rot, 'rb') as f:
            kmeans_rot = pickle.load(f)
        with open(params.kmeans_trans, 'rb') as f:
            kmeans_trans = pickle.load(f)

    plane_rcnn_model = build_plane_rcnn(params)
    ipaa_list = []
    trans_err_list = []
    rot_err_list = []

    mv = MultiViewInference(params, params.transformer_ckpt)
    num_eval = 0

    for i in tqdm(range(len(multi_view_data))):

        output_preds = {}
        img_paths = []
        skip_sample = False

        tran_logits = torch.zeros(params.num_images * params.num_images, 32)
        rot_logits = torch.zeros(params.num_images * params.num_images, 32)

        for j in range(params.num_images):
            filename = multi_view_data[i][str(j)]['file_name']
            if mp3d_corr_file[filename] == None:
                skip_sample = True
                break

            if not params.run_sparseplane:
                img_paths.append(filename)
                tmp = mp3d_corr_file[filename]
                if tmp[0] == 0:
                    path = "train"
                elif tmp[0] == 1:
                    path = "val"
                else:
                    path = "test"

                if params.eval_corr:
                    with open(os.path.join(params.path, path, str(tmp[1]) + '.pkl'), 'rb') as f:
                        view_ft = pickle.load(f)[tmp[2]]
                    output_preds[str(j)] = view_ft
                    output_preds[str(j)]['embedding'] = torch.tensor(output_preds[str(j)]['embedding'])
                    output_preds[str(j)]['pred_plane'] = torch.tensor(output_preds[str(j)]['pred_plane'])

        if skip_sample:
            if params.run_sparseplane:
                assert i in edge_data['ignore_idx']
            continue
        
        if not params.run_sparseplane:
            if params.eval_corr:
                for j in range(params.num_images):
                    for k in range(params.num_images):

                        output = plane_rcnn_model.inference(img_paths[j], img_paths[k])
                        pred = plane_rcnn_model.process(output)
                        tran_logits[j * params.num_images + k, :] = pred['camera']['logits']['tran'].cpu().reshape(-1)
                        rot_logits[j * params.num_images + k, :] = pred['camera']['logits']['rot'].cpu().reshape(-1)

                output_preds['pred_camera'] = {}
                output_preds['pred_camera']['tran_logits'] =  tran_logits
                output_preds['pred_camera']['rot_logits'] = rot_logits
                output_preds['num_images'] = params.num_images
                features = output_preds
            else:
                features = mv.predict_features(img_paths)

        num_images = params.num_images
        if not params.run_sparseplane:
            edges = mv.build_connectivity_graph(features)
            if not params.run_baseline:
                edge_cameras, edge_preds = mv.run_model_pairwise(num_images, edges, features, device=params.device)
                chained_cameras = mv.chain_cameras_wrapper(num_images, edges, edge_cameras, features, device=params.device)
                merged_planes, _ = mv.merge_planes(num_images, edges, params.plane_corr_threshold, edge_preds, features)

        ipaa_list_sample = []
        trans_err_sample = []
        rot_err_sample = []

        if params.run_baseline:
            edge_cameras = []
            plane_list = []
            for e in range(edges.shape[0]):

                if params.eval_camera:
                    trans_cluster = torch.argmax(features['pred_camera']['tran_logits'][edges[e, 0] * num_images + edges[e, 1], :]).item()
                    rot_cluster = torch.argmax(features['pred_camera']['rot_logits'][edges[e, 0] * num_images + edges[e, 1], :]).item()
                    pred_rot = quaternion.from_float_array(kmeans_rot.cluster_centers_[rot_cluster, :].reshape((-1)))
                    pred_trans = kmeans_trans.cluster_centers_[trans_cluster, :].reshape((3, 1))
                    edge_cameras.append({'rotation': pred_rot, 'position': pred_trans})
                else:
                    embs_0 = features[str(edges[e, 0])]['embedding']
                    embs_1 = features[str(edges[e, 1])]['embedding']
                    cost_mat = torch.cdist(embs_0, embs_1)
                    km_corrs = apply_km(cost_mat.cpu().numpy(), threshold=0.8)
                    plane_list.append(np.nonzero(km_corrs))
                    

        if params.run_sparseplane:
            edges = np.array(edge_data['edge_dict'][str(i)])
            edge_cameras = []
            plane_list = []
            for e in range(edges.shape[0]):
                sample = sparseplane_data[num_eval * (params.num_images - 1) + e]

                if params.eval_camera:
                    camera_dict = sample['best_camera']
                    camera_dict['rotation'] = quaternion.from_float_array(camera_dict['rotation'])
                    camera_dict['position'] = camera_dict['position'].reshape((3, 1))
                    edge_cameras.append(camera_dict)
                else:
                    km_corrs = np.nonzero(sample['best_assignment'])
                    plane_list.append(km_corrs)
            

        if params.run_sparseplane or params.run_baseline:
            if params.eval_camera:
                adj_mat = np.zeros((num_images, num_images))
                adj_mat[edges[:, 0], edges[:, 1]] = 1
                adj_mat[edges[:, 1], edges[:, 0]] = 1
                chained_cameras = mv.chain_cameras(num_images, edges, edge_cameras, adj_mat)
            else:
                merged_planes = merge_planes(num_images, edges, plane_list)
                

        for j in range(params.num_images):
            for k in range(j + 1, params.num_images):

                if params.eval_corr:

                    plane_instances_1 = multi_view_data[i][str(j)]['plane_instance_id']
                    plane_instances_2 = multi_view_data[i][str(k)]['plane_instance_id']
                    
                    gt_corr = np.zeros((len(plane_instances_1), len(plane_instances_2)), dtype=np.intc)
                    for idx1 in range(len(plane_instances_1)):
                        for idx2 in range(len(plane_instances_2)):
                            if plane_instances_1[idx1] == plane_instances_2[idx2]:
                                gt_corr[idx1, idx2] = 1
                                break

                    pred_corr = np.zeros((len(plane_instances_1), len(plane_instances_2)), dtype=np.intc)
                    for p in range(len(merged_planes)):
                        if merged_planes[p][j] != -1 and merged_planes[p][k] != -1:
                            pred_corr[merged_planes[p][j], merged_planes[p][k]] = 1

                    ipaa_list_sample.append(compute_ipaa(pred_corr, gt_corr))


                else:
                    rot_gt_j = quaternion.from_float_array(np.array(multi_view_data[i][str(j)]['camera']['rotation']))
                    trans_gt_j = np.array(multi_view_data[i][str(j)]['camera']['position']).reshape((3, 1))
                    rot_gt_k = quaternion.from_float_array(np.array(multi_view_data[i][str(k)]['camera']['rotation']))
                    trans_gt_k = np.array(multi_view_data[i][str(k)]['camera']['position']).reshape((3, 1))
                    gt_datapoint = [{'camera': {'rotation': rot_gt_j, 'position': trans_gt_j}}, {'camera': {'rotation': rot_gt_k, 'position': trans_gt_k}}]
                    gt_cam_transform = get_relative_pose_from_datapoint(gt_datapoint)

                    chained_cam_j = chained_cameras[j]
                    chained_cam_k = chained_cameras[k]
                    pred_rot = quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(chained_cam_k['rotation']).T@quaternion.as_rotation_matrix(chained_cam_j['rotation']))
                    pred_rot = quaternion.as_float_array(pred_rot)
                    pred_trans = (quaternion.as_rotation_matrix(chained_cam_k['rotation']).T@(chained_cam_j['position'].reshape((3, 1)) - chained_cam_k['position'].reshape((3, 1))))


                    trans_err_sample.append(np.linalg.norm(pred_trans - gt_cam_transform['position']))
                    rot_error = 2 * np.arccos(np.clip(np.abs(np.sum(pred_rot.reshape((-1)) * quaternion.as_float_array(gt_cam_transform['rotation']).reshape((-1)))), 0, 1)) * \
                        180/np.pi
                    rot_err_sample.append(rot_error)


        num_eval += 1

        if params.eval_corr:
            ipaa_list.extend(ipaa_list_sample)
        else:
            trans_err_list.extend(trans_err_sample)
            rot_err_list.extend(rot_err_sample)

    
    if params.eval_corr:
        ipaa_dict = {}
        thresh = np.arange(0, 1.1, 0.1).tolist()
        for t in thresh:
            ipaa_dict[t] = 0
        for i in range(len(ipaa_list)):
            for j in range(len(thresh)):
                if ipaa_list[i] >= thresh[j]:
                    ipaa_dict[thresh[j]] += 1
        for t in thresh:
            ipaa_dict[t] = ipaa_dict[t]/len(ipaa_list)

        print(ipaa_dict)

    else:
        trans_err_list = np.array(trans_err_list)
        trans_acc = np.mean((trans_err_list <= 1).astype(np.float32))
        print("Trans Error [mean, median, acc]: %.5f, %.5f, %.5f" % (np.mean(trans_err_list), np.median(trans_err_list), trans_acc))

        rot_err_list = np.array(rot_err_list)
        rot_acc = np.mean((rot_err_list <= 30).astype(np.float32))
        print("Rot Error [mean, median, acc]: %.5f, %.5f, %.5f" % (np.mean(rot_err_list), np.median(rot_err_list), rot_acc))
    
    print("\nNumber of samples evaluated %d, out of %d" % (num_eval, len(multi_view_data)))




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Transformer model for planar embedding correspondences')
    parser.add_argument('--mp3d_corr_file', help=".pkl file for matching up mp3d and multi-view dataset")
    parser.add_argument('--multi_view_file', help="Multi-view json file")
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--transformer_ckpt', help='checkpoint for eval')
    parser.add_argument('--eval_corr', action='store_true')
    parser.add_argument('--eval_camera', action='store_true')
    parser.add_argument('--run_baseline', action='store_true')
    parser.add_argument('--run_sparseplane', default=None, help="Sparseplane output pkl file")
    parser.add_argument('--edge_file', default=None, help="File containing edges")
    args = parser.parse_args()
    args.device = 'cuda'

    params = get_default_dataset_config("plane_params")
    params = SimpleNamespace(**vars(args), **vars(params))

    eval_multi_view(params)
