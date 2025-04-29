import torch
import torch.nn as nn
import argparse
import time
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
FEATMAP_SIZE = 64
SD_IMG_SIZE = 16 * FEATMAP_SIZE
DINO_IMG_SIZE = 14 * FEATMAP_SIZE

# Import project-specific modules
from data.visnir_dataset import VisNirDataset
from models.extractor_sd import process_features_and_mask, load_model
from models.extractor_dino import ViTExtractor
from utils.utils import resize, co_pca

# Load models
sd_model, aug = load_model(diffusion_ver="v1-5", image_size=SD_IMG_SIZE, num_timesteps=50)
dinov2_model = ViTExtractor(model_type='dinov2_vitb14', stride=14, device='cuda')

def load_and_preprocess(img_path, sd_size, dino_size):
    """Load and resize images for both models"""
    img = Image.open(img_path).convert("RGB")
    return {
        'sd_input': resize(img, sd_size, resize=True, to_pil=True, edge=False),
        'dino_input': resize(img, dino_size, resize=True, to_pil=True, edge=False)
    }

def extract_features(model_inputs, device):
    """Extract features using both SD and DINOv2 models"""
    sd_features = []
    dino_features = []
    
    # Process SD features
    with torch.no_grad():
        for img in [model_inputs['fixed_sd'], model_inputs['moving_sd']]:
            features = process_features_and_mask(sd_model, aug, img, 
                                                input_text=None, mask=False, raw=True)
            sd_features.append(features)
    
    # Apply co-PCA
    processed_sd = co_pca(*sd_features, dim=[256, 256, 256])
    
    # Process DINO features
    for img in [model_inputs['fixed_dino'], model_inputs['moving_dino']]:
        batch = dinov2_model.preprocess_pil(img)
        features = dinov2_model.extract_descriptors(batch.to(device), 
                                                   layer=11, facet='token')
        dino_features.append(features)
    
    return processed_sd, dino_features

def compute_pair_features(fixed_path, moving_path, device):
    """Compute combined features for image pair"""
    # Load and preprocess images
    fixed = load_and_preprocess(fixed_path, SD_IMG_SIZE, DINO_IMG_SIZE)
    moving = load_and_preprocess(moving_path, SD_IMG_SIZE, DINO_IMG_SIZE)
    
    # start_time = time.time()
    processed_sd, dino_features = extract_features({
        'fixed_sd': fixed['sd_input'],
        'moving_sd': moving['sd_input'],
        'fixed_dino': fixed['dino_input'],
        'moving_dino': moving['dino_input']
    }, device)
    
    # Process and combine features
    sd_features = [f.reshape(1, 1, -1, FEATMAP_SIZE**2).permute(0,1,3,2) 
                  for f in processed_sd]
    dino_features = [f / f.norm(dim=-1, keepdim=True) for f in dino_features]
    
    combined_features = []
    for sd_f, dino_f in zip(sd_features, dino_features):
        combined = torch.cat((sd_f, dino_f), dim=-1)
        reshaped = combined.permute(0,1,3,2).reshape(-1, combined.shape[-1], 
                                                   FEATMAP_SIZE, FEATMAP_SIZE)
        combined_features.append(reshaped)
    
    # print(f"Inference time for {moving_path}: {time.time()-start_time:.4f}s")
    return combined_features

def get_save_path(filepath, subdir='features'):
    """Generate save path for processed features"""
    base_dir = os.path.dirname(filepath)
    new_dir = f"{base_dir}_{subdir}"
    # save_dir = os.path.join(os.path.dirname(base_dir), new_dir)
    os.makedirs(new_dir, exist_ok=True)
    return os.path.join(new_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}.pt")

def process_dataset(dataloader, desc=None):
    """Process dataset and save features"""
    for fixed_img, moving_img, fix_path, moving_path in tqdm(dataloader, desc=desc):
        fixed_feats, moving_feats = compute_pair_features(fix_path[0], moving_path[0], device)
        torch.save(fixed_feats.squeeze(0), get_save_path(fix_path[0]))
        torch.save(moving_feats.squeeze(0), get_save_path(moving_path[0]))

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Feature extraction pipeline')
    parser.add_argument('--fixed_image_dir', default='data/vis_nir_split/train_vis',
                      help='Directory for fixed images')
    parser.add_argument('--moving_image_dir', default='data/vis_nir_split/train_nir_trans',
                      help='Directory for moving images')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    args = parser.parse_args()

    # Dataset preparation
    transform = transforms.Compose([transforms.ToTensor()])
    datasets = {
        'train': DataLoader(VisNirDataset(
            args.fixed_image_dir, 
            args.moving_image_dir, 
            transform,
            load_features=False
        ), batch_size=1, shuffle=False),
        'test': DataLoader(VisNirDataset(
            args.fixed_image_dir.replace('train', 'test', 1), 
            args.moving_image_dir.replace('train', 'test', 1),
            transform,
            load_features=False
        ), batch_size=1, shuffle=False)
    }

    # Process datasets
    process_dataset(datasets['train'], 'Processing training set')
    process_dataset(datasets['test'], 'Processing validation set')
    
    print("All processing completed!")