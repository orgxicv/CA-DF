import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import argparse

# Local imports
from models.CADF_model import RegisNet
from data.visnir_dataset import VisNirDataset
from models.layers import SpatialTransformer

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pth', 
                       help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='test_outputs',
                       help='Directory to save test results')
    return parser.parse_args()

def create_test_dataset(args):
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_set = VisNirDataset(
        fixed_image_dir="data/vis_nir_split/test_vis",
        moving_image_dir="data/vis_nir_split/test_nir_trans",
        transform=transform,
        load_features=True
    )
    
    return DataLoader(test_set, batch_size=1, shuffle=False)

def run_test(model, loader, device, args):
    model.eval()
    spa = SpatialTransformer(volsize=[args.height, args.width])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        for fixed_img, moving_img, fixed_feats, moving_feats, path in tqdm(loader):
            fixed_img = fixed_img.to(device)
            moving_img = moving_img.to(device)
            fixed_feats = fixed_feats.to(device)
            moving_feats = moving_feats.to(device)
            
            # Forward pass
            flow_pred, y_pred, fix, _ = model(fixed_feats, moving_feats)
            
            # Save warped image
            warp_img = spa(moving_img, flow_pred)
            warp_np = (255 * warp_img.cpu().squeeze(0).permute(1, 2, 0).numpy()).astype(np.uint8)
            
            # Save original and warped images for comparison
            filename = os.path.basename(path[0])
            Image.fromarray(warp_np).save(os.path.join(args.output_dir, f"{filename}"))

def main():
    args = config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test dataset
    test_loader = create_test_dataset(args)
    
    # Initialize model and load weights
    model = RegisNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")
    
    # Run test
    run_test(model, test_loader, device, args)
    print(f"Test completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()