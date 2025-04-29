import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class VisNirDataset(Dataset):
    """Dataset for image registration with optional pre-computed features."""
    
    def __init__(self, fixed_image_dir, moving_image_dir, transform=None, load_features=True):
        """
        Initialize the dataset.
        
        Args:
            fixed_image_dir: Path to directory containing fixed images
            moving_image_dir: Path to directory containing moving images
            transform: Optional image transformations
            load_features: Boolean controlling whether to load pre-computed features
        """
        self.fixed_image_dir = fixed_image_dir
        self.moving_image_dir = moving_image_dir
        self.transform = transform
        self.load_features = load_features
        self.image_files = sorted(f for f in os.listdir(fixed_image_dir) if f.endswith('.jpg'))

    def __len__(self):
        """Return number of image pairs in dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load and return a sample by index.
        
        Returns:
            Tuple containing:
            - fixed_image: Tensor of fixed image
            - moving_image: Tensor of moving image  
            - fixed_feats: Pre-computed features for fixed image (None if not loaded)
            - moving_feats: Pre-computed features for moving image (None if not loaded)
            - moving_path: Path to moving image
        """
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Load images
        fixed_img = self._load_image(self.fixed_image_dir, img_name)
        moving_img = self._load_image(self.moving_image_dir, img_name)
        
        # Initialize feature variables
        fixed_feats, moving_feats = None, None
        
        # Load pre-computed features if enabled
        if self.load_features:
            fixed_feats = torch.load(self._get_feat_path(self.fixed_image_dir, img_name), 
                                   map_location='cuda:0')
            moving_feats = torch.load(self._get_feat_path(self.moving_image_dir, img_name),
                                    map_location='cuda:0')
            
            return (fixed_img, moving_img, fixed_feats, moving_feats, 
                os.path.join(self.moving_image_dir, img_name))
        else:
            return (fixed_img, moving_img, os.path.join(self.fixed_image_dir, img_name),
                os.path.join(self.moving_image_dir, img_name))

    def _load_image(self, dir_path, filename):
        """Load and transform an image."""
        img = Image.open(os.path.join(dir_path, filename)).convert('RGB')
        return self.transform(img) if self.transform else img

    @staticmethod
    def _get_feat_path(img_dir, filename):
        """Generate path to saved features."""
        base_dir = os.path.dirname(os.path.dirname(img_dir))
        feat_dir = f"{os.path.basename(img_dir)}_features"
        return os.path.join(base_dir, feat_dir, f"{os.path.splitext(filename)[0]}.pt")