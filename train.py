import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import argparse
from PIL import Image
from loguru import logger
import sys

# Local imports
from models.CADF_model import RegisNet
from data.visnir_dataset import VisNirDataset
from utils.utils_losses import LossFunction_Dense
from models.layers import SpatialTransformer

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam_weight', type=float, default=10.0, help='Loss weight')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    return parser.parse_args()

def setup_logger(output_dir):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    log_file = os.path.join(output_dir, "training.log")
    logger.add(log_file, enqueue=True, format=log_format)
    return logger


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_datasets(args):
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_set = VisNirDataset(
        fixed_image_dir="data/vis_nir_split/train_vis",
        moving_image_dir="data/vis_nir_split/train_nir_trans",
        transform=transform,
        load_features=True
    )
    
    val_set = VisNirDataset(
        fixed_image_dir="data/vis_nir_split/test_vis",
        moving_image_dir="data/vis_nir_split/test_nir_trans",
        transform=transform,
        load_features=True
    )
    
    return (
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=1, shuffle=False)
    )

def train_epoch(model, loader, criterion, optimizer, scheduler, device, args):
    model.train()
    total_loss = 0.0
    for _, _, fixed_feats, moving_feats, _ in tqdm(loader):
        fixed_feats, moving_feats = fixed_feats.to(device), moving_feats.to(device)
        
        optimizer.zero_grad()
        flow_pred, y_pred, fix, _ = model(fixed_feats, moving_feats)
        loss, _, _ = criterion(flow_pred, fix, y_pred)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    spa = SpatialTransformer(volsize=[args.height, args.width])
    
    with torch.no_grad():
        for fixed_img, moving_img, fixed_feats, moving_feats, path in tqdm(loader):
            fixed_img, moving_img = fixed_img.to(device), moving_img.to(device)
            
            flow_pred, y_pred, fix, _ = model(fixed_feats, moving_feats)
            loss, _, _ = criterion(flow_pred, fix, y_pred)
            total_loss += loss.item()
            
            # Save warped image
            warp_img = spa(moving_img, flow_pred)
            warp_np = (255 * warp_img.cpu().squeeze(0).permute(1, 2, 0).numpy()).astype(np.uint8)
            Image.fromarray(warp_np).save(f"outputs/{os.path.basename(path[0])}")
    
    return total_loss / len(loader)

def main():
    set_seed()
    # Argument parsing
    args = config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup directories and logger
    os.makedirs("outputs", exist_ok=True)
    logger = setup_logger("outputs")
    
    # Create datasets and model
    train_loader, val_loader = create_datasets(args)
    model = RegisNet().to(device)
    criterion = LossFunction_Dense(lam=args.lam_weight).to(device)
    
    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-6)
    
    # Training loop
    min_loss = float('inf')
    cnt = 0
    for epoch in range(args.num_epochs):
        cnt += 1
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        if cnt == 20:
            cnt = 0
            # Validation
            val_loss = validate(model, val_loader, criterion, device, args)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(model.state_dict(), "outputs/best_model.pth")
                logger.info(f"New best model saved: {min_loss:.4f}")
            else:
                logger.info(f"Skipping.")

if __name__ == "__main__":
    main()