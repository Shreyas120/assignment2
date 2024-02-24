import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.ops import knn_points

# define losses
def voxel_loss(voxels_src,voxels_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = nn.BCELoss()
	return loss(voxels_src, voxels_tgt)
	# return loss(torch.sigmoid(voxels_src), voxels_tgt)

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# assert point_cloud_src.shape[0] == point_cloud_tgt.shape[0] # batch size should be same
	# assert point_cloud_src.shape[2] == point_cloud_tgt.shape[2] # dimension should be same

	b = point_cloud_src.shape[0]
	# n_points_src = point_cloud_src.shape[1] 
	# n_points_tgt = point_cloud_tgt.shape[1]
	# np_src = torch.full((b,), n_points_src, dtype=torch.int64, device=point_cloud_src.device)
	# np_tgt = torch.full((b,), n_points_tgt, dtype=torch.int64, device=point_cloud_src.device)
	# d12, _, _ = knn_points(point_cloud_src, point_cloud_tgt, lengths1=np_src, lengths2=np_tgt, K=1)
	# d21, _, _ = knn_points(point_cloud_tgt, point_cloud_src, lengths1=np_tgt, lengths2=np_src, K=1)

	d12, _, _ = knn_points(point_cloud_src, point_cloud_tgt, K=1)
	d21, _, _ = knn_points(point_cloud_tgt, point_cloud_src, K=1)
	loss = d12.sum() + d21.sum()
	
	loss = loss / b
	# print(f"Cham: {chamfer_distance(point_cloud_src, point_cloud_tgt)[0]}, Me: {loss}")
	return chamfer_distance(point_cloud_src, point_cloud_tgt)[0]
	return loss

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss for mesh 
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian