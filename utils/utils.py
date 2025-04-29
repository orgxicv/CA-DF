import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import kornia

def get_torch_save_path(filepath, subdir_name='features'):
    tmp_path1 = filepath.rsplit('/', 1)[0]
    last_dir = tmp_path1.rsplit('/', 1)[1]
    output_subdir = tmp_path1.replace(last_dir, f"{last_dir}_{subdir_name}")
    os.makedirs(output_subdir, exist_ok=True)
    save_path = os.path.join(output_subdir, os.path.splitext(os.path.basename(filepath))[0] + '.pt')
    return save_path


def resize(img, target_res=224, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def co_pca(features1, features2, dim=[128,128,128]):
    
    processed_features1 = {}
    processed_features2 = {}
    s5_size = features1['s5'].shape[-1]
    s4_size = features1['s4'].shape[-1]
    s3_size = features1['s3'].shape[-1]
    # Get the feature tensors
    s5_1 = features1['s5'].reshape(features1['s5'].shape[0], features1['s5'].shape[1], -1)
    s4_1 = features1['s4'].reshape(features1['s4'].shape[0], features1['s4'].shape[1], -1)
    s3_1 = features1['s3'].reshape(features1['s3'].shape[0], features1['s3'].shape[1], -1)

    s5_2 = features2['s5'].reshape(features2['s5'].shape[0], features2['s5'].shape[1], -1)
    s4_2 = features2['s4'].reshape(features2['s4'].shape[0], features2['s4'].shape[1], -1)
    s3_2 = features2['s3'].reshape(features2['s3'].shape[0], features2['s3'].shape[1], -1)
    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [[s5_1, s5_2], [s4_1, s4_2], [s3_1, s3_2]]):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1) # along the spatial dimension
        features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)

        # Compute the PCA
        # pca = faiss.PCAMatrix(features.shape[-1], target_dim)

        # Train the PCA
        # pca.train(features[0].cpu().numpy())

        # Apply the PCA
        # features = pca.apply(features[0].cpu().numpy()) # (t_x+t_y)x(d)

        # convert to tensor
        # features = torch.tensor(features, device=features1['s5'].device).unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        
        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        

        # Split the features
        processed_features1[name] = features[:, :, :features.shape[-1] // 2] # Bx(d)x(t_x)
        processed_features2[name] = features[:, :, features.shape[-1] // 2:] # Bx(d)x(t_y)

    # reshape the features
    processed_features1['s5']=processed_features1['s5'].reshape(processed_features1['s5'].shape[0], -1, s5_size, s5_size)
    processed_features1['s4']=processed_features1['s4'].reshape(processed_features1['s4'].shape[0], -1, s4_size, s4_size)
    processed_features1['s3']=processed_features1['s3'].reshape(processed_features1['s3'].shape[0], -1, s3_size, s3_size)

    processed_features2['s5']=processed_features2['s5'].reshape(processed_features2['s5'].shape[0], -1, s5_size, s5_size)
    processed_features2['s4']=processed_features2['s4'].reshape(processed_features2['s4'].shape[0], -1, s4_size, s4_size)
    processed_features2['s3']=processed_features2['s3'].reshape(processed_features2['s3'].shape[0], -1, s3_size, s3_size)

    # Upsample s5 spatially by a factor of 2
    processed_features1['s5'] = F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features2['s5'] = F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features1['s5'] = torch.cat([processed_features1['s4'], processed_features1['s5']], dim=1)
    processed_features2['s5'] = torch.cat([processed_features2['s4'], processed_features2['s5']], dim=1)

    # Set s3 as the new s4
    processed_features1['s4'] = processed_features1['s3']
    processed_features2['s4'] = processed_features2['s3']

    # Remove s3 from the features dictionary
    processed_features1.pop('s3')
    processed_features2.pop('s3')

    # current order are layer 8, 5, 2
    features1_gether_s4_s5 = torch.cat([processed_features1['s4'], F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features2_gether_s4_s5 = torch.cat([processed_features2['s4'], F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features1_gether_s4_s5, features2_gether_s4_s5



def find_flow_batch(image1, features1, features2, resolution=None):
    batch_size = image1.shape[0]
    
    if resolution is not None and not (resolution == 48):  # resize the feature map to the resolution
        features1 = F.interpolate(features1, size=resolution, mode='bilinear', align_corners=True)  
        features2 = F.interpolate(features2, size=resolution, mode='bilinear', align_corners=True)
    
    features1_2d = features1.reshape(batch_size, features1.shape[1], -1).permute(0, 2, 1)  
    features2_2d = features2.reshape(batch_size, features2.shape[1], -1).permute(0, 2, 1) 
    print(features1.shape, features2.shape)
    distances = torch.cdist(features1_2d, features2_2d, p=2)  
    nearest_patch_indices = torch.argmin(distances, dim=2)  
    
    # print(nearest_patch_indices.max(), nearest_patch_indices.min())

    height1, width1 = resolution, resolution
    height2, width2 = resolution, resolution
    # compute grid_x1 and grid_y1
    grid = kornia.utils.create_meshgrid(height1, width1, device='cuda').to(torch.float32)  # [1, h, w, 2]
    grid = grid.permute(0,3,1,2)

    grid_x1 = grid[0, 0, ...]  # [h, w]
    grid_y1 = grid[0, 1, ...]  


    # compute grid_x2 and grid_y2
    grid_x2 = nearest_patch_indices % width2
    grid_y2 = nearest_patch_indices // width2
    grid_x2 = grid_x2.reshape(height1, width1)  # [batch_size, h, w]
    grid_y2 = grid_y2.reshape(height1, width1)  

    grid_x2 = 2 * (grid_x2 / (width2 - 1) - 0.5)
    grid_y2 = 2 * (grid_y2 / (width2 - 1) - 0.5)

    @torch.jit.script
    def compute_displacement(grid_x1: torch.Tensor, grid_y1: torch.Tensor,
                            grid_x2: torch.Tensor, grid_y2: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            grid_x2 - grid_x1,
            grid_y2 - grid_y1
        ], dim=0).unsqueeze(0)

    displacement_field = compute_displacement(grid_x1, grid_y1, grid_x2, grid_y2)
        
    return displacement_field



