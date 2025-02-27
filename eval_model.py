import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np
from pathlib import Path
import os
from render import threeDObject, renderObjects
from pytorch3d.structures import Pointclouds
import imageio

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=110, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', default="", type=str)  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--viz_vox_layers', action='store_true') 
    parser.add_argument('--output_path', default=Path(os.getcwd())/'data'/'shreyasj'/'2', type=Path)
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args, isovalue=0.5):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=isovalue)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point" or args.type == "point_h":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics

def interpret_model(args):

    def visualize_layers_with_input_image(input_image, layers_dict):
        
        # Function to visualize each layer in 3D
        def visualize_layer(layer, ax, layer_name):
            layer = layer.squeeze(0) #D x S x S x S
    
            # Calculate different statistics for RGB channels
            R_channel = torch.max(layer, dim=0).values
            G_channel = torch.mean(layer, dim=0)
            B_channel = torch.min(layer, dim=0).values

            # Helper function for normalization
            def safe_normalize(channel):
                min_val = channel.min()
                max_val = channel.max()
                if min_val==max_val:
                    normalized_channel = channel 
                else:
                    normalized_channel = (channel - min_val) / (max_val - min_val)
                return normalized_channel.detach().cpu().numpy()

            # Apply safe normalization to each channel
            R_norm_np = safe_normalize(R_channel)
            G_norm_np = safe_normalize(G_channel)
            B_norm_np = safe_normalize(B_channel)

            # Expand dimensions to [S, S, S, 1] for concatenation
            R_expanded = np.expand_dims(R_norm_np, axis=-1)
            G_expanded = np.expand_dims(G_norm_np, axis=-1)
            B_expanded = np.expand_dims(B_norm_np, axis=-1)

            rgb = np.concatenate([R_expanded, G_expanded, B_expanded], axis=-1)

            activations = torch.max(layer, dim=0).values
            voxels = activations > 0.3
    
            ax.voxels(voxels, facecolors=rgb, edgecolors='black', linewidth=0.5, shade=False)
            ax.title.set_text(layer_name)
        
        num_layers = len(layers_dict)
        fig = plt.figure(figsize=(20, 10))
        
        # Visualize the input image
        ax0 = fig.add_subplot(1, num_layers + 1, 1)
        ax0.imshow(input_image)  # Convert to numpy array for visualization
        ax0.title.set_text('Input Image')
        ax0.axis('off')
        
        # Visualize each layer in a subplot
        for i, (layer_name, layer_data) in enumerate(layers_dict.items(), start=1):
            ax = fig.add_subplot(1, num_layers + 1, i + 1, projection='3d')
            visualize_layer(layer_data, ax, layer_name)
        
        plt.tight_layout()
        plt.savefig(args.output_path / (f'interpret_{step}.png'))


    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}_{args.load_checkpoint}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        loaded_iter = checkpoint['step']
        print(f"Succesfully loaded iter {loaded_iter} from checkpoint_{args.type}_{args.load_checkpoint}.pth !")


    for step in range(10):
        feed_dict = next(eval_loader)
        images_gt, mesh_gt = preprocess(feed_dict, args)
        predictions = model(images_gt, args)
        visualize_layers_with_input_image(images_gt[0,...].cpu().numpy(), predictions)
    

def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}_{args.load_checkpoint}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        loaded_iter = checkpoint['step']
        print(f"Succesfully loaded iter {loaded_iter} from checkpoint_{args.type}_{args.load_checkpoint}.pth !")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        try:
            metrics = evaluate(predictions, mesh_gt, thresholds, args, isovalue=0.5)
            
            # visualization block
            if (step % args.vis_freq) == 0:
                    
                # plt.imsave(f'vis/{step}_{args.type}.png', rend)
                srcObj = None
                if args.type == "vox":
                    srcObj = threeDObject(f"Predicted {args.type}", "vox", predictions[0,:,:,:], args.device)
                elif args.type == "point" or args.type == "point_h":
                    srcObj = threeDObject(f"Predicted {args.type}", "point", Pointclouds(points=predictions, features=torch.zeros_like(predictions)), args.device)
                elif args.type == "mesh":
                    predictions.textures = pytorch3d.renderer.TexturesVertex((torch.ones_like(predictions.verts_packed()).to(args.device) * torch.tensor((0,0,0)).to(args.device)).unsqueeze(0))
                    srcObj = threeDObject(f"Predicted {args.type}", "mesh",  predictions, args.device)
                
                mesh_gt.textures = pytorch3d.renderer.TexturesVertex((torch.ones_like(mesh_gt.verts_packed()).to(args.device) * torch.tensor((0,0,0)).to(args.device)).unsqueeze(0))
                targObj = threeDObject("GT mesh", "mesh", mesh_gt, args.device)

                images_list = []
                for cam_angle in range(0, 360, 10):
                    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=2.5, elev=0, azim=cam_angle, degrees=True)
                    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=args.device)
                    image = renderObjects(objects=[srcObj, targObj],cameras=cameras)
                    images_list.append(image)
        
                full_path = args.output_path / (f'eval_{args.type}_{step}.gif')
                imageio.mimsave(full_path, images_list, fps=15, loop=0)
                plt.imsave(args.output_path / (f'eval_{args.type}_gtimg_{step}.png'), images_gt[0,...].cpu().numpy())
                print(f'Saved the 360 GIF for evaluationg of {args.type} in {full_path}')
        
            total_time = time.time() - start_time
            iter_time = time.time() - iter_start_time

            f1_05 = metrics['F1@0.050000']
            avg_f1_score_05.append(f1_05)
            avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
            avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
            avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
        
        except ValueError as ve:
            print(f"ValueError: {ve} when evaluating {args.type} at isovalue 0.5, trying 0.3 instead.")
            try:
               metrics = evaluate(predictions, mesh_gt, thresholds, args, isovalue=0.5)
            except ValueError as ve2:
                print(f"ValueError: {ve} when evaluating {args.type} at isovalue 0.3, skipping this sample.")

    print(f'Done! Note: Skipped {max_iter - len(avg_f1_score)} samples.')
    avg_f1_score = torch.stack(avg_f1_score).mean(0)
    save_plot(thresholds, avg_f1_score,  args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.viz_vox_layers:
        interpret_model(args)
    else:
        evaluate_model(args)
