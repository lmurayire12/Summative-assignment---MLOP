"""
Image Preprocessing Module for Product Engagement Prediction

Handles image preprocessing, augmentation, and data pipeline creation.
Extends our previous neural network preprocessing experience to computer vision.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductImagePreprocessor:
    """
    Advanced image preprocessing for product engagement prediction.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 batch_size: int = 32):
        self.target_size = target_size
        self.batch_size = batch_size
        self.setup_augmentation_pipelines()
        
    def setup_augmentation_pipelines(self):
        """Setup different augmentation pipelines for training and validation."""
        
        # Training augmentation (aggressive)
        self.train_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.3
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Validation/Test augmentation (minimal)
        self.val_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Keras-based data generators (alternative approach)
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        self.val_datagen = ImageDataGenerator(rescale=1./255)
        
    def preprocess_single_image(self, image_path: Union[str, Path], 
                              for_training: bool = False) -> np.ndarray:
        """
        Preprocess a single image for inference or training.
        
        Args:
            image_path: Path to the image file
            for_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image array
        """
        
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
            
        # Apply appropriate transformation
        transform = self.train_transform if for_training else self.val_transform
        transformed = transform(image=image)
        processed_image = transformed['image']
        
        return processed_image
        
    def create_data_generators(self, data_dir: Path, use_albumentations: bool = False):
        """
        Create data generators for training and validation.
        
        Args:
            data_dir: Path to data directory
            use_albumentations: Whether to use Albumentations or Keras generators
            
        Returns:
            Tuple of (train_generator, validation_generator, test_generator)
        """
        
        if use_albumentations:
            return self._create_albumentations_generators(data_dir)
        else:
            return self._create_keras_generators(data_dir)
            
    def _create_keras_generators(self, data_dir: Path):
        """Create Keras-based data generators."""
        
        train_dir = data_dir / "train"
        test_dir = data_dir / "test"
        
        # Training generator with augmentation
        train_generator = self.train_datagen.flow_from_directory(
            train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation generator
        validation_generator = self.train_datagen.flow_from_directory(
            train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation', 
            shuffle=True,
            seed=42
        )
        
        # Test generator
        test_generator = self.val_datagen.flow_from_directory(
            test_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        logger.info("Keras data generators created successfully")
        logger.info(f"Training samples: {train_generator.samples}")
        logger.info(f"Validation samples: {validation_generator.samples}")
        logger.info(f"Test samples: {test_generator.samples}")
        
        return train_generator, validation_generator, test_generator
        
    def _create_albumentations_generators(self, data_dir: Path):
        """Create Albumentations-based custom generators."""
        
        # Get file lists
        train_files, train_labels = self._get_file_lists(data_dir / "train")
        test_files, test_labels = self._get_file_lists(data_dir / "test")
        
        # Split training into train/validation
        from sklearn.model_selection import train_test_split
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
        
        # Create custom generators
        train_generator = AlbumentationsDataGenerator(
            train_files, train_labels, self.batch_size,
            self.train_transform, shuffle=True
        )
        
        val_generator = AlbumentationsDataGenerator(
            val_files, val_labels, self.batch_size,
            self.val_transform, shuffle=False
        )
        
        test_generator = AlbumentationsDataGenerator(
            test_files, test_labels, self.batch_size,
            self.val_transform, shuffle=False
        )
        
        logger.info("Albumentations data generators created successfully")
        logger.info(f"Training samples: {len(train_files)}")
        logger.info(f"Validation samples: {len(val_files)}")
        logger.info(f"Test samples: {len(test_files)}")
        
        return train_generator, val_generator, test_generator
        
    def _get_file_lists(self, directory: Path):
        """Get file lists and labels from directory structure."""
        
        files = []
        labels = []
        
        # High engagement (label 1)
        high_dir = directory / "high_engagement"
        if high_dir.exists():
            high_files = list(high_dir.glob("*.jpg"))
            files.extend(high_files)
            labels.extend([1] * len(high_files))
            
        # Low engagement (label 0)
        low_dir = directory / "low_engagement"
        if low_dir.exists():
            low_files = list(low_dir.glob("*.jpg"))
            files.extend(low_files)
            labels.extend([0] * len(low_files))
            
        return files, labels
        
    def apply_test_time_augmentation(self, image: np.ndarray, 
                                   num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Apply test-time augmentation for more robust predictions.
        
        Args:
            image: Input image array
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented images
        """
        
        augmented_images = []
        
        # Original image
        augmented_images.append(self.val_transform(image=image)['image'])
        
        # Light augmentations for TTA
        tta_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for _ in range(num_augmentations - 1):
            augmented = tta_transform(image=image)['image']
            augmented_images.append(augmented)
            
        return augmented_images


class AlbumentationsDataGenerator(Sequence):
    """
    Custom data generator using Albumentations for more advanced augmentation.
    """
    
    def __init__(self, file_paths: List[Path], labels: List[int], 
                 batch_size: int, transform: A.Compose, shuffle: bool = True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        """Number of batches per epoch."""
        return len(self.file_paths) // self.batch_size
        
    def __getitem__(self, index):
        """Get batch of data."""
        
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load and process batch
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            # Load image
            image = cv2.imread(str(self.file_paths[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformation
            transformed = self.transform(image=image)
            processed_image = transformed['image']
            
            batch_images.append(processed_image)
            batch_labels.append(self.labels[idx])
            
        return np.array(batch_images), np.array(batch_labels)
        
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class ImageQualityAssessment:
    """
    Assess and filter image quality for better training data.
    """
    
    @staticmethod
    def calculate_image_metrics(image_path: Union[str, Path]) -> dict:
        """Calculate various quality metrics for an image."""
        
        # Load image
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        metrics = {
            'brightness': np.mean(image),
            'contrast': np.std(gray),
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'resolution': image.shape[:2],
            'file_size': Path(image_path).stat().st_size,
            'aspect_ratio': image.shape[1] / image.shape[0]
        }
        
        return metrics
        
    @staticmethod
    def filter_low_quality_images(image_dir: Path, 
                                min_sharpness: float = 100.0,
                                min_contrast: float = 20.0) -> List[Path]:
        """Filter out low quality images based on metrics."""
        
        filtered_images = []
        
        for image_path in image_dir.glob("*.jpg"):
            try:
                metrics = ImageQualityAssessment.calculate_image_metrics(image_path)
                
                if (metrics['sharpness'] >= min_sharpness and 
                    metrics['contrast'] >= min_contrast):
                    filtered_images.append(image_path)
                    
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
                
        logger.info(f"Filtered {len(filtered_images)} high-quality images from {image_dir}")
        return filtered_images


def main():
    """Main function to test preprocessing functionality."""
    
    # Initialize preprocessor
    preprocessor = ProductImagePreprocessor(target_size=(224, 224), batch_size=16)
    
    # Test data directory
    data_dir = Path.cwd() / "data"
    
    if data_dir.exists():
        # Create data generators
        train_gen, val_gen, test_gen = preprocessor.create_data_generators(
            data_dir, use_albumentations=False
        )
        
        print("✅ Data generators created successfully!")
        print(f"Training batches: {len(train_gen)}")
        print(f"Validation batches: {len(val_gen)}")
        print(f"Test batches: {len(test_gen)}")
        
        # Test single image preprocessing
        sample_images = list((data_dir / "train" / "high_engagement").glob("*.jpg"))
        if sample_images:
            sample_path = sample_images[0]
            processed = preprocessor.preprocess_single_image(sample_path)
            print(f"Sample image processed: {processed.shape}")
            
    else:
        print("❌ Data directory not found. Run data_acquisition.py first.")


if __name__ == "__main__":
    main()
