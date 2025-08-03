"""
Model Training and Inference Module for Product Engagement Prediction

Implements CNN models with transfer learning for image-based engagement prediction.
Extends our previous neural network optimization experience to computer vision.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
from tensorflow.keras import regularizers
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductEngagementModel:
    """
    Main model class for product engagement prediction from images.
    Combines our previous optimization experience with computer vision techniques.
    """
    
    def __init__(self, model_name: str = "product_engagement_cnn", 
                 input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.model_name = model_name
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.model_metadata = {}
        
    def create_custom_cnn(self, regularization_type: str = 'l2', 
                         dropout_rate: float = 0.3) -> Model:
        """
        Create custom CNN architecture for product engagement prediction.
        Building on our previous neural network optimization experience.
        """
        
        model = Sequential(name=f"{self.model_name}_custom")
        
        # Define regularizer
        regularizer = None
        if regularization_type == 'l1':
            regularizer = regularizers.l1(0.01)
        elif regularization_type == 'l2':
            regularizer = regularizers.l2(0.01)
            
        # Convolutional Base
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(dropout_rate * 0.5))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(dropout_rate * 0.7))
        
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(dropout_rate))
        
        # Classification Head
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu', kernel_regularizer=regularizer))
        model.add(Dropout(dropout_rate))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizer))
        model.add(Dropout(dropout_rate * 0.5))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        
        logger.info(f"Custom CNN created with {regularization_type} regularization")
        return model
        
    def create_transfer_learning_model(self, base_model_name: str = 'resnet50',
                                     fine_tune_layers: int = 10) -> Model:
        """
        Create transfer learning model with pre-trained base.
        """
        
        # Load pre-trained base model
        if base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
            
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ], name=f"{self.model_name}_{base_model_name}")
        
        logger.info(f"Transfer learning model created with {base_model_name} base")
        return model, base_model
        
    def setup_fine_tuning(self, model: Model, base_model: Model, 
                         fine_tune_layers: int = 10):
        """
        Setup fine-tuning by unfreezing top layers of base model.
        """
        
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - fine_tune_layers
        
        # Freeze layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        logger.info(f"Fine-tuning setup: {fine_tune_layers} layers unfrozen")
        return model
        
    def get_compilation_configs(self) -> Dict:
        """
        Get different compilation configurations for model comparison.
        Building on our previous optimization experience.
        """
        
        configs = {
            'adam_standard': {
                'optimizer': Adam(learning_rate=0.001),
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall']
            },
            'adam_low_lr': {
                'optimizer': Adam(learning_rate=0.0001),
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall']
            },
            'rmsprop': {
                'optimizer': RMSprop(learning_rate=0.001),
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall']
            },
            'adam_scheduled': {
                'optimizer': Adam(learning_rate=0.001),
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall']
            }
        }
        
        return configs
        
    def create_callbacks(self, model_save_path: Path, 
                        patience: int = 5) -> List:
        """
        Create comprehensive callbacks for training monitoring.
        """
        
        callbacks = [
            # Early Stopping
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpointing
            ModelCheckpoint(
                filepath=str(model_save_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            
            # Learning Rate Reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
        
    def train_model(self, model: Model, train_generator, validation_generator,
                   compilation_config: Dict, epochs: int = 30,
                   callbacks: List = None) -> Dict:
        """
        Train model with comprehensive monitoring.
        """
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Compile model
        model.compile(**compilation_config)
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model and history
        self.model = model
        self.history = history
        
        # Create training summary
        training_summary = {
            'model_name': self.model_name,
            'epochs_trained': len(history.history['loss']),
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'compilation_config': compilation_config,
            'training_date': datetime.now().isoformat()
        }
        
        logger.info("Training completed successfully!")
        return training_summary
        
    def evaluate_model(self, test_generator) -> Dict:
        """
        Comprehensive model evaluation on test set.
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
            
        logger.info("Evaluating model on test set...")
        
        # Reset test generator
        test_generator.reset()
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Get true labels
        true_labels = test_generator.classes
        
        # Calculate metrics
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, confusion_matrix,
                                   classification_report)
        
        evaluation_results = {
            'accuracy': accuracy_score(true_labels, predicted_classes),
            'precision': precision_score(true_labels, predicted_classes),
            'recall': recall_score(true_labels, predicted_classes),
            'f1_score': f1_score(true_labels, predicted_classes),
            'auc_roc': roc_auc_score(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, predicted_classes).tolist(),
            'classification_report': classification_report(
                true_labels, predicted_classes,
                target_names=['Low Engagement', 'High Engagement'],
                output_dict=True
            ),
            'total_test_samples': len(true_labels),
            'evaluation_date': datetime.now().isoformat()
        }
        
        logger.info(f"Evaluation completed. Accuracy: {evaluation_results['accuracy']:.4f}")
        return evaluation_results
        
    def save_model_with_metadata(self, save_dir: Path, 
                                evaluation_results: Dict = None):
        """
        Save model with comprehensive metadata for MLOps.
        """
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / f"{self.model_name}.h5"
        self.model.save(model_path)
        
        # Prepare metadata
        metadata = {
            'model_name': self.model_name,
            'model_path': str(model_path),
            'input_shape': self.input_shape,
            'model_architecture': 'CNN with transfer learning',
            'framework': 'TensorFlow/Keras',
            'save_date': datetime.now().isoformat(),
            'total_parameters': self.model.count_params(),
            'trainable_parameters': sum([tf.keras.backend.count_params(w) 
                                       for w in self.model.trainable_weights])
        }
        
        # Add evaluation results if provided
        if evaluation_results:
            metadata['evaluation_results'] = evaluation_results
            
        # Add training history if available
        if self.history:
            metadata['training_history'] = {
                key: [float(val) for val in values] 
                for key, values in self.history.history.items()
            }
            
        # Save metadata
        metadata_path = save_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return str(model_path), str(metadata_path)
        
    @classmethod
    def load_model_with_metadata(cls, model_path: Union[str, Path]) -> 'ProductEngagementModel':
        """
        Load saved model with metadata.
        """
        
        model_path = Path(model_path)
        
        # Load model
        model = load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
            
        # Create instance
        instance = cls(
            model_name=metadata.get('model_name', 'loaded_model'),
            input_shape=tuple(metadata.get('input_shape', (224, 224, 3)))
        )
        instance.model = model
        instance.model_metadata = metadata
        
        logger.info(f"Model loaded from {model_path}")
        return instance


class ModelComparison:
    """
    Compare multiple model variants for optimal performance.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model: ProductEngagementModel, 
                 evaluation_results: Dict):
        """Add model to comparison."""
        
        self.models[name] = model
        self.results[name] = evaluation_results
        
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, ProductEngagementModel]:
        """Get best performing model based on specified metric."""
        
        if not self.results:
            raise ValueError("No models added for comparison")
            
        best_name = max(self.results.keys(), 
                       key=lambda x: self.results[x].get(metric, 0))
        best_model = self.models[best_name]
        
        logger.info(f"Best model: {best_name} ({metric}: {self.results[best_name][metric]:.4f})")
        return best_name, best_model
        
    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report."""
        
        comparison_data = {}
        
        for name, results in self.results.items():
            comparison_data[name] = {
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1_score': results.get('f1_score', 0),
                'auc_roc': results.get('auc_roc', 0)
            }
            
        # Find best model for each metric
        best_models = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            best_name = max(comparison_data.keys(),
                          key=lambda x: comparison_data[x][metric])
            best_models[metric] = {
                'model_name': best_name,
                'score': comparison_data[best_name][metric]
            }
            
        report = {
            'comparison_data': comparison_data,
            'best_models_by_metric': best_models,
            'total_models_compared': len(self.models),
            'report_date': datetime.now().isoformat()
        }
        
        return report


def main():
    """Main function to demonstrate model training."""
    
    logger.info("Product Engagement Model Training Demo")
    
    # Initialize model
    model_trainer = ProductEngagementModel("demo_product_engagement")
    
    # Create custom CNN
    custom_model = model_trainer.create_custom_cnn(
        regularization_type='l2', 
        dropout_rate=0.3
    )
    
    print("✅ Custom CNN model created")
    print(f"Total parameters: {custom_model.count_params():,}")
    
    # Create transfer learning model
    transfer_model, base_model = model_trainer.create_transfer_learning_model('resnet50')
    
    print("✅ Transfer learning model created")
    print(f"Total parameters: {transfer_model.count_params():,}")
    
    # Get compilation configs
    configs = model_trainer.get_compilation_configs()
    print(f"✅ Compilation configurations: {list(configs.keys())}")
    
    print("\n💡 To train models, run:")
    print("1. python src/data_acquisition.py")
    print("2. Use the Jupyter notebook for end-to-end training")
    print("3. Or integrate with your training pipeline")


if __name__ == "__main__":
    main()
