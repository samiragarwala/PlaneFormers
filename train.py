import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
import argparse
import datetime
import time
import os

from planeformers.utils.misc import *
from planeformers.models.planeformer import PlaneFormer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train_model(params):

    train_dataloader = create_data_loader("train", params)
    val_dataloader = create_data_loader("val", params)

    model = PlaneFormer(params, device=device)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # setting flag for not validating immediately after restoring ckpt
    if params.restore_ckpt:
        val_model = False
    else:
        val_model = True

    if not os.path.isdir(params.ckpt_dir):
        os.mkdir(params.ckpt_dir)

    if params.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.reg)
    elif params.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.reg)
    elif params.optimizer == "sgdm":
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, \
            nesterov=params.nesterov, weight_decay=params.reg)
    elif params.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=params.reg)
    elif params.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=params.lr, weight_decay=params.reg)

    if params.scheduler == "cos_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.t_max)
    elif params.scheduler == "cos_annealing_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=params.lr_restart)
    elif params.scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params.lr_gamma)
    elif params.scheduler != None:
        raise Exception("Invalid scheduler %s provided" % (params.scheduler))
    
    iter_num = 0
    try:
        os.mkdir("runs/" + params.run_name)
    except:
        pass
    writer = SummaryWriter("runs/" + params.run_name)
    
    # restore checkpoint for training
    if params.restore_ckpt:
        checkpoint = torch.load(params.restore_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not params.not_restore_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iter_num = checkpoint['iter_num']
        print("Restored model checkpoint at iteration %d" % (iter_num))
        
        # updating learning rate on restoring checkpoint as specified
        for param_group in optimizer.param_groups:
            param_group['lr'] = params.lr

        # updating scheduler based on number of iterations completed
        if params.scheduler and params.warmup_scheduler:
            for _ in range(iter_num):
                scheduler.step()
    else:
        writer.add_scalar("train/lr", params.lr, 0)

    print("Starting training")
    print("Training parameters: Optimizer %s, LR %.5f, Batch Size %d, Epochs %d" \
        % (params.optimizer, params.lr, params.batch_size, params.num_epochs))

    model.train()
    for e in range(params.num_epochs):
        for i, batch in enumerate(tqdm(train_dataloader)):

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

            optimizer.zero_grad()
            src_padding_mask = create_src_padding_mask(batch['emb'], torch.sum(batch['num_planes'], axis=1), \
                device=device)
            output = model(batch, src_padding_mask)

            loss_dict = compute_loss(batch, output, params, device=device)
            loss_dict['loss'].backward()

            # gradient clipping
            if params.clip_value:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_value)

            optimizer.step()

            if params.scheduler:
                scheduler.step()
            
            with torch.no_grad():
                if iter_num % params.print_freq == 0:

                    if params.scheduler:
                        writer.add_scalar("train/lr", np.array(scheduler.get_last_lr()), iter_num)
                
                    for key in loss_dict.keys():
                        writer.add_scalar("train/loss/" + key, loss_dict[key].item(), iter_num)
                    tqdm.write("Iteration: %d, loss: %.3f" % (iter_num, loss_dict['loss'].item()))
                                    
                    # mask indicating where GT camera corr is given
                    camera_corr_mask = (batch['gt_camera_corr'] == 1)
                    num_gt_camera = camera_corr_mask.to(torch.float).sum()
                    if num_gt_camera > 0 and num_gt_camera < batch['gt_camera_corr'].shape[0]:
                        # camera correspondence auroc
                        camera_auroc = compute_auroc(output['camera_corr'].reshape(-1), \
                            batch['gt_camera_corr'].reshape(-1))
                        writer.add_scalar("train/camera_corr/auroc", camera_auroc, iter_num)


                    if num_gt_camera > 0:
                        # plane corr auroc
                        plane_auroc = compute_auroc(output['plane_corr'][camera_corr_mask, :, :][output['plane_mask'][camera_corr_mask, :, :]].reshape(-1), \
                            batch['gt_plane_corr'][camera_corr_mask, :, :][batch['gt_plane_mask'][camera_corr_mask, :, :]].reshape(-1))
                        writer.add_scalar("train/plane_corr/auroc", plane_auroc, iter_num)


            if iter_num != 0 and iter_num % params.val_freq == 0:
                if val_model:
                    # validating model
                    validate(model, val_dataloader, params, device, writer, iter_num)

                    # saving model checkpoint
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'batch_size': params.batch_size,
                        'lr': params.lr,
                        'optimizer': params.optimizer,
                        'params': params,
                    }, os.path.join(params.ckpt_dir, params.run_name + '_iter%d.pt' % (iter_num)))
                else:
                    val_model = True

            iter_num += 1


def evaluate_model(params):

    print("Evaluating model")
    dataloader = create_data_loader(params.eval_set, params)

    model = PlaneFormer(params, device)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    print("Restoring model checkpoint")
    checkpoint = torch.load(params.eval_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    stats = validate(model, dataloader, params, device)

    for key in stats.keys():
        stats[key] = str(stats[key])

    with open(params.run_name + "_eval.json", 'w') as f:
        json.dump(stats, f, indent=6)


if __name__=="__main__":

    timestamp = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    # training modes
    parser = argparse.ArgumentParser(description='Transformer model for planar embedding correspondences')
    parser.add_argument('--train', action='store_true', help="Flag to train model")
    parser.add_argument('--restore_ckpt', default=None, help="Checkpoint to restart training from")
    parser.add_argument('--not_restore_optimizer', action='store_true')
    parser.add_argument('--subset', type=int, default=None, help="Size of dataset subset to train on")
    parser.add_argument('--project_ft', action='store_true', help="Project input features to 899D")

    # eval
    parser.add_argument('--eval', action='store_true', help="Run default validation code to eval model")
    parser.add_argument('--gen_eval_file', action='store_true', help="Create eval file for sparse planar eval code")
    parser.add_argument('--eval_ckpt', default=None, help="Checkpoint to evaluate model")
    parser.add_argument('--eval_set', default='test', help="Dataset to use for evaluation")
    parser.add_argument('--camera_search_len', type=int, default=3, help="Number of camera poses to consider during eval")
    parser.add_argument('--plane_corr_threshold', type=float, default=0.7, help="Threshold for Hungarian algorithm")
    parser.add_argument('--output_file', default=timestamp)
    parser.add_argument('--eval_input_path', help="Path to file which should be modified in gen_eval_file")
    parser.add_argument('--eval_add_cam_residual', action='store_true', \
        help="Add predicted residual to rot/trans while evaluating")
    parser.add_argument('--write_new_eval_file', action='store_true', help="Create new input file not based on template")
    parser.add_argument('--store_camera_outputs', action='store_true', help="Option to store camera predictions for finding weights in gen_eval_file")
    parser.add_argument('--camera_output_dir', help="Output directory for camera predictions ")
    parser.add_argument('--sparseplane_config', help="Config file for sparseplane model")


    # dataset parameters
    parser.add_argument('--json_path', type=str, required=True, help="Path to directory where json annotation files are saved")
    parser.add_argument('--path', type=str, required=True, help="Path to parent directory of dataset folder")
    parser.add_argument('--kmeans_rot', type=str,required=True, help="Path to K-Means classifier for rotation")
    parser.add_argument('--kmeans_trans', type=str, required=True, help="Path to K-Means classifier for translation")
    parser.add_argument('--emb_format', type=str, help="plane_params/plane_and_camera_params/balance_cam/hard_cam", required=True)
    parser.add_argument('--plane_param_scaling', type=float, help="Scaling factor for plane params in embedding", default=0.1)
    parser.add_argument('--use_plane_params', action='store_true', help='Include 3D plane params in embedding')
    parser.add_argument('--use_appearance_embeddings', action='store_true', help='Include appearance embeddings')
    parser.add_argument('--use_camera_conf_score', action='store_true', help='Include RPNet confidence score in embedding')
    parser.add_argument('--use_plane_mask', action='store_true', help='Include plane mask in embedding')
    parser.add_argument('--mask_height', default=24, help="Downsample mask height to this value")
    parser.add_argument('--mask_width', default=32, help="Downsample mask width to this value")
    parser.add_argument('--deg_cut_off', type=float, help="Degree cut-off for negative rotation camera correspondence", default=30.0)
    parser.add_argument('--trans_cut_off', type=float, help="Distance cut-off for negative translation camera correspondence", default=1.0)
    parser.add_argument('--transform', type=str, help="Transform to apply to dataset (random_camera_corr, gt_camera_corr, rpnet_preds, default: None)", default=None)
    parser.add_argument('--dataset_inference', action='store_true', help="Generate dataset examples in inference mode without GT")

    # training parameters
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--optimizer', default='sgdm', help="adam/adamw/sgm/sgdm/adagrad")
    parser.add_argument('--momentum', type=float, default=0.9, help="Value of momentum in optimizer")
    parser.add_argument('--nesterov', action='store_true', help="Use nesterov momentum")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for model training")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--model_name',  default=None, help="Model loss functions to use during training (plane_corr/camera_corr/plane_camera_corr)", required=True)
    parser.add_argument('--reg', type=float,  default=0.0, help="L2 penalty")
    parser.add_argument('--scheduler', type=str, default="cos_annealing", help="cos_annealing/cos_annealing_warm_restarts/exp")
    parser.add_argument('--t_max', type=int, default=40000, help="Number of iterations for training")
    parser.add_argument('--lr_restart', type=int, default=350, help="Number of iterations between scheduler warm restart")
    parser.add_argument('--lr_gamma', type=float, default=0.99995, help="Decay factor per iteration for learning rate in exp scheduler")
    parser.add_argument('--warmup_scheduler', action='store_true', help="Warm-up scheduler to iterations completed on restore checkpoint")
    parser.add_argument('--plane_corr_wt', type=float, default=1.0, help="Weight for plane corr loss in camera error correction model")
    parser.add_argument('--camera_corr_wt', type=float, default=1.0, help="Weight for camera corr loss in camera error correction model")
    parser.add_argument('--rot_loss_wt', type=float, default=1.0, help="Weight for rotation error loss in camera error correction model")
    parser.add_argument('--trans_loss_wt', type=float, default=0.5, help="Weight for translation error loss in camera error correction model")
    parser.add_argument('--clip_value', type=float, default=None, help="Max norm of gradient for parameters before rescaling")
    parser.add_argument('--use_l1_res_loss', action='store_true', help="Use L1 dist loss for camera residual training (Default: L2)")

    # freezing model heads during traininng
    parser.add_argument('--freeze_camera_corr_head', action='store_true')
    parser.add_argument('--freeze_plane_corr_head', action='store_true')
    parser.add_argument('--freeze_camera_residual_head', action='store_true')
    parser.add_argument('--freeze_transformer_network', action='store_true')

    # model parameters
    parser.add_argument('--transformer_on', action='store_true', help="Use transformer network instead to refine embeddings isntead of directly using embeddings")
    parser.add_argument('--padding_value', type=float, default=0.0, help="Padding value for transformer network")
    parser.add_argument('--d_model', type=int, default=899, help='Dimension of input embedding')
    parser.add_argument('--nhead', type=int, default=1, help="Number of heads in transformer encoder")
    parser.add_argument('--fc_dim', type=int, default=2048, help="Dimension of FC in transformer encoder")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout in transformer encoder")
    parser.add_argument('--nlayers', type=int, default=5, help="Number of layers in transformer network")
    parser.add_argument('--hard_thresh', type=float, default=1.4, help="Pairwise dist between app embeddings <= this value to be considered hard example")


    # logging parameters
    parser.add_argument('--print_freq', type=int, help="Frequency of logging data in iterations", default=100)
    parser.add_argument('--run_name', help="Name of the experimental run", default=timestamp)
    parser.add_argument('--val_freq', type=int, help="Frequency in iterations of validating and checkpointing model", default=1000)
    parser.add_argument('--ckpt_dir', default='./planeformers/checkpoints/', help="Directory to save model checkpoint in during training")

    args = parser.parse_args()
    print('epochs:', args.num_epochs)

    if args.train or args.restore_ckpt:
        train_model(args)
    elif args.eval:
        evaluate_model(args)
    elif args.gen_eval_file or args.store_camera_outputs:
        gen_eval_file(args, camera_search_len=args.camera_search_len, \
            device=device)
    else:
        raise Exception("No mode provided for training/evaluating model")


        

