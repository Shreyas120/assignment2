import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
import dataset_location
import torch
from render import threeDObject, renderObjects
import pytorch3d
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=15000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--output_path', default=Path(os.getcwd())/'data'/'shreyasj'/'1', type=Path)
    return parser

def fit_mesh(mesh_src, mesh_tgt, args, tol=1e-4):
    start_iter = 0
    start_time = time.time()
    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    global q
    srcs = []
    qidx = 0
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        if loss_vis < tol:
            print('Converged!')
            break
        
        if step%q[qidx]==0:
            srcs.append((step,new_mesh_src.clone()))
            qidx = min(qidx+1, len(q)-1)

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')
    return srcs

def fit_pointcloud(pointclouds_src, pointclouds_tgt, args, tol=1e-4):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    global q
    srcs = []
    qidx = 0
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        if loss_vis < tol:
            print('Converged!')
            break
        
        if step%q[qidx]==0:
            srcs.append((step,pointclouds_src.clone()))
            qidx = min(qidx+1, len(q)-1)

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    return srcs


def fit_voxel(voxels_src, voxels_tgt, args, tol=1e-4):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    global q
    srcs = []
    qidx = 0
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(torch.sigmoid(voxels_src),voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        if loss_vis < tol:
            print('Converged!')
            break
        
        if step%q[qidx]==0:
            srcs.append((step,torch.sigmoid(voxels_src).clone()))
            qidx = min(qidx+1, len(q)-1)

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    return srcs

def plot(srcs: list, tgt, type: str):
    
    if type == "vox":
        srcObj = threeDObject("Source", "vox", srcs[0][1][0,:,:,:], args.device)
        targObj = threeDObject("Target", "vox", tgt[0,:,:,:], args.device)
    elif type == "point":
        srcObj = threeDObject("Source", "point", Pointclouds(points=srcs[0][1], features=torch.zeros_like(srcs[0][1])), args.device)
        targObj = threeDObject("Target", "point", Pointclouds(points=tgt, features=torch.zeros_like(tgt)), args.device)
    elif type == "mesh":
        srcs[0][1].textures = pytorch3d.renderer.TexturesVertex((torch.ones_like( srcs[0][1].verts_packed()) * torch.tensor((0,0,0)).to(args.device)).unsqueeze(0))
        tgt.textures = pytorch3d.renderer.TexturesVertex((torch.ones_like( tgt.verts_packed()) * torch.tensor((0,0,0)).to(args.device)).unsqueeze(0))
        srcObj = threeDObject("Source", "mesh",  srcs[0][1], args.device)
        targObj = threeDObject("Target", "mesh", tgt, args.device)
    
    cam_angles = np.linspace(0, 3*np.pi, len(srcs))
    images_list = []
    for (step,src), cam_angle in zip(srcs, cam_angles):
        if type == "vox":
            srcObj.update(data=src[0,:,:,:], name=f"Optimizer Iteration: {step}")
        elif type == "mesh":
            src.textures =  pytorch3d.renderer.TexturesVertex((torch.ones_like( src.verts_packed()) * torch.tensor((0,0,0)).to(args.device)).unsqueeze(0))
            srcObj.update(data=src, name=f"Optimizer Iteration: {step}")
        elif type == "point":
            srcObj.update(data=Pointclouds(points=src, features=torch.zeros_like(src)), name=f"Optimizer Iteration: {step}")

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=2.5, elev=0, azim=cam_angle, degrees=False)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=args.device)

        image = renderObjects(objects=[srcObj, targObj],cameras=cameras)
        images_list.append(image)

    return images_list

def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()

    global q 

    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        q = generate_intervals(args.max_iter, mapping={0: 100})
        srcs = fit_voxel(voxels_src, voxels_tgt, args)
        images_list = plot(srcs, voxels_tgt, "vox") 


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        q = generate_intervals(args.max_iter, mapping={0: 100})
        srcs = fit_pointcloud(pointclouds_src, pointclouds_tgt, args) 
        images_list = plot(srcs, pointclouds_tgt, "point") 
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        q = generate_intervals(args.max_iter, mapping={0: 100})
        srcs = fit_mesh(mesh_src, mesh_tgt, args)        
        images_list = plot(srcs, mesh_tgt, "mesh") 

    full_path = args.output_path / (f'fit_{args.type}.gif')
    imageio.mimsave(full_path, images_list, fps=30, loop=0)
    print(f'Saved the 360 GIF for fitting {args.type} in {full_path}')


def generate_intervals(N, mapping={0: 100, 400: 200, 2000: 500, 5000: 1000}):
    step_size = 0 
    log_at = [] 
    step = 0  
    while step < N:
        log_at.append(step)
        if step in mapping:
            step_size = mapping[step]
        step += step_size
    log_at[0] = 1
    return log_at
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
