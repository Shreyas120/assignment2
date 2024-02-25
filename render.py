from pathlib import Path
import pytorch3d as p3d
import torch
import numpy as np
import cv2
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,)
from pytorch3d.structures import Pointclouds, Meshes

class threeDObject:
    name: str
    dtype: str
    data: any
    device: torch.device
    renderer: p3d.renderer

    def __init__(self, name: str, dtype: str, data: any, device: torch.device, 
                 renderer: p3d.renderer=None):
        
        self.name = name
        self.device = device
        self.dtype = dtype
        self.update(data=data)
        
        if renderer is None:
            self.renderer = renderer
            self.setRenderer()

    def setRenderer(self, renderer: p3d.renderer=None, image_size=512, 
                    lights=None, radius=0.01, background_color=(1, 1, 1)):
                    
        if renderer is None:
            if self.dtype == "mesh":
                raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,)
                self.renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings),
                            shader=HardPhongShader(device=self.device, lights=lights))
                
            elif self.dtype == "point":
                raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
                self.renderer = PointsRenderer(rasterizer=PointsRasterizer(raster_settings=raster_settings),
                                                compositor=AlphaCompositor(background_color=background_color),)
            elif  self.dtype == "vox":
                # raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
                # self.renderer = PointsRenderer(rasterizer=PointsRasterizer(raster_settings=raster_settings),
                #                                 compositor=AlphaCompositor(background_color=background_color),)
                raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,)
                self.renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings),
                            shader=HardPhongShader(device=self.device, lights=lights))
                
                # raise ValueError("Required to specify renderer for voxel type. Points or Mesh renderer can be used.")

            else:
                raise ValueError("Invalid dtype, renderer not defined")
        else:
            self.renderer = renderer
            
    def update(self, name=None, data=None):
        if self.dtype=='point':
            if not isinstance(data, Pointclouds):
                raise ValueError("Expected data to be of type pytorch3D.Structures.Pointclouds")
            self.data = data.to(self.device)

        elif self.dtype=='mesh':
            if not isinstance(data, Meshes):
                raise ValueError("Expected data to be of type pytorch3D.Structures.Meshes")
            self.data = data.to(self.device)

        elif self.dtype=='vox':
            if not isinstance(data, torch.Tensor):
                raise ValueError("Invalid data type for voxel or missing coordinates")
            self.data = data.to(self.device)
        
        self.name = name if name is not None else self.name

        
    def render(self, cameras:p3d.renderer.FoVPerspectiveCameras) -> np.array:

        if self.dtype == "vox":
            if isinstance(self.renderer, PointsRenderer):
                raise('Unable to render voxel as point cloud')
                #Render the voxel as point cloud, Pytorch3D does not support voxel rendering
                #Convert voxel grid to point cloud
                #Each voxel's center is considered as a point
                # indices = torch.nonzero(self.data)
                # print(self.data.shape)
                # print(indices.shape)
                # # points = voxel_indices
                # data = Pointclouds(points=[points])

            elif isinstance(self.renderer, MeshRenderer):
                if len(self.data.shape) == 3:
                    self.data = self.data.unsqueeze(0) #add batch dimension
                data = p3d.ops.cubify(self.data, thresh=0.5) 
                data.textures = p3d.renderer.TexturesVertex((torch.ones_like(data.verts_packed()) * torch.tensor((0,0,0)).to(self.device)).unsqueeze(0))
                
        else:
            data = self.data 

        rendered = self.renderer(data, cameras=cameras)
        img = rendered.cpu().detach().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        img = np.clip(img * 255, 0, 255).astype(np.uint8) #for compatability with PIL
        return addTextToImage(img, self.name)


def renderObjects(objects: list[threeDObject], cameras=p3d.renderer.FoVPerspectiveCameras, space_width=10):
    img = None
    space_color = (255, 255, 255)
    
    for obj in objects:
        if img is None:
            img = obj.render(cameras)
        else:
            space = np.full((img.shape[0], space_width, img.shape[2]), space_color, dtype=np.uint8)
            img = np.concatenate((img, space, obj.render(cameras)), axis=1)

    return img

def addTextToImage(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 0)  # Text color
    thickness = 2

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate the amount of space needed for the text
    text_height = text_size[1] + 30  # Adding some extra space for padding

    # Create a new image with extra space for the text at the top
    new_image_height = image.shape[0] + text_height
    new_image = np.full((new_image_height, image.shape[1], 3), (255,255,255), dtype=np.uint8)
    new_image[text_height:,:,:] = image  # Copy the original image starting below the new blank space
    
    # Calculate the position to place the text
    # Text will be placed in the new blank space, so we adjust `text_y` accordingly
    text_x = (new_image.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10  # Adjusted for the new space

    # Add the text to the new image
    cv2.putText(new_image, text, (text_x, text_y), font, font_scale, color, thickness)

    return new_image
