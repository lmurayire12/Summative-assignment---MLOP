"""
Prediction Module for Product Engagement Prediction

Handles inference, batch prediction, and production-ready prediction services.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
import logging
from datetime import datetime
import base64
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductEngagementPredictor:
    """
    Production-ready predictor for product engagement from images.
    """
    
    def __init__(self, model_path: Union[str, Path], 
                 target_size: Tuple[int, int] = (224, 224)):
        self.model_path = Path(model_path)
        self.target_size = target_size
        self.model = None
        self.model_metadata = {}
        self.load_model()
        
    def load_model(self):
        """Load the trained model and metadata."""
        
        try:
            # Load model
            self.model = load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Load metadata if available
            metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded")
            else:
                logger.warning("Model metadata not found")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def preprocess_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            
        Returns:
            Preprocessed image array ready for prediction
        """
        
        try:
            # Handle different input types
            if isinstance(image_input, (str, Path)):
                # File path
                image = load_img(image_input, target_size=self.target_size)
                image_array = img_to_array(image)
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                if len(image_input.shape) == 3:
                    image = cv2.resize(image_input, self.target_size)
                    image_array = image.astype(np.float32)
                else:
                    raise ValueError("Image array must have 3 dimensions (H, W, C)")
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input.resize(self.target_size)
                image_array = np.array(image)
            else:
                raise ValueError("Unsupported image input type")
                
            # Normalize pixel values
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
                
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
            
    def predict_single(self, image_input: Union[str, Path, np.ndarray, Image.Image],
                      return_probabilities: bool = True) -> Dict:
        """
        Predict engagement for a single product image.
        
        Args:
            image_input: Image to predict on
            return_probabilities: Whether to include probability scores
            
        Returns:
            Dictionary with prediction results
        """
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_input)
            
            # Make prediction
            prediction_prob = self.model.predict(processed_image, verbose=0)[0][0]
            predicted_class = int(prediction_prob > 0.5)
            confidence = max(prediction_prob, 1 - prediction_prob)
            
            # Prepare result
            result = {
                'status': 'success',
                'predicted_class': 'high_engagement' if predicted_class == 1 else 'low_engagement',
                'predicted_label': 'High Engagement' if predicted_class == 1 else 'Low Engagement',
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            if return_probabilities:
                result['engagement_probability'] = float(prediction_prob)
                result['probabilities'] = {
                    'low_engagement': float(1 - prediction_prob),
                    'high_engagement': float(prediction_prob)
                }
                
            # Add business recommendation
            if predicted_class == 1:
                result['recommendation'] = "High potential for customer engagement - consider featuring prominently"
                result['marketing_action'] = "promote"
            else:
                result['recommendation'] = "Lower engagement potential - consider optimizing imagery or pricing"
                result['marketing_action'] = "optimize"
                
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def predict_batch(self, image_inputs: List[Union[str, Path, np.ndarray]], 
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict engagement for multiple product images efficiently.
        
        Args:
            image_inputs: List of images to predict on
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        
        results = []
        
        try:
            for i in range(0, len(image_inputs), batch_size):
                batch_inputs = image_inputs[i:i + batch_size]
                batch_images = []
                
                # Preprocess batch
                for image_input in batch_inputs:
                    try:
                        processed_image = self.preprocess_image(image_input)
                        batch_images.append(processed_image[0])  # Remove batch dimension
                    except Exception as e:
                        logger.warning(f"Failed to preprocess image {image_input}: {e}")
                        results.append({
                            'status': 'error',
                            'error_message': f"Preprocessing failed: {e}",
                            'image_input': str(image_input),
                            'timestamp': datetime.now().isoformat()
                        })
                        continue
                        
                if not batch_images:
                    continue
                    
                # Convert to numpy array and predict
                batch_array = np.array(batch_images)
                batch_predictions = self.model.predict(batch_array, verbose=0)
                
                # Process results
                for j, (image_input, prediction_prob) in enumerate(zip(batch_inputs, batch_predictions)):
                    if j < len(batch_images):  # Only process successfully preprocessed images
                        predicted_class = int(prediction_prob[0] > 0.5)
                        confidence = max(prediction_prob[0], 1 - prediction_prob[0])
                        
                        result = {
                            'status': 'success',
                            'image_input': str(image_input),
                            'predicted_class': 'high_engagement' if predicted_class == 1 else 'low_engagement',
                            'predicted_label': 'High Engagement' if predicted_class == 1 else 'Low Engagement',
                            'confidence': float(confidence),
                            'engagement_probability': float(prediction_prob[0]),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results.append(result)
                        
            logger.info(f"Batch prediction completed for {len(results)} images")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [{
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }]
            
    def predict_from_base64(self, base64_image: str) -> Dict:
        """
        Predict from base64 encoded image (useful for API endpoints).
        
        Args:
            base64_image: Base64 encoded image string
            
        Returns:
            Dictionary with prediction results
        """
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Make prediction
            result = self.predict_single(image)
            result['input_type'] = 'base64'
            
            return result
            
        except Exception as e:
            logger.error(f"Base64 prediction failed: {e}")
            return {
                'status': 'error',
                'error_message': f"Base64 decoding or prediction failed: {e}",
                'timestamp': datetime.now().isoformat()
            }
            
    def predict_with_explanations(self, image_input: Union[str, Path, np.ndarray],
                                use_grad_cam: bool = False) -> Dict:
        """
        Predict with visual explanations (Grad-CAM or similar).
        
        Args:
            image_input: Image to predict on
            use_grad_cam: Whether to generate Grad-CAM visualization
            
        Returns:
            Dictionary with prediction and explanation
        """
        
        # Basic prediction
        result = self.predict_single(image_input)
        
        if use_grad_cam and result['status'] == 'success':
            try:
                # Generate Grad-CAM heatmap
                heatmap = self._generate_grad_cam(image_input)
                result['explanation'] = {
                    'method': 'grad_cam',
                    'heatmap_available': heatmap is not None,
                    'description': 'Visual attention map showing important regions for prediction'
                }
                
                if heatmap is not None:
                    result['explanation']['heatmap_shape'] = heatmap.shape
                    
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed: {e}")
                result['explanation'] = {
                    'method': 'grad_cam',
                    'error': str(e)
                }
        else:
            result['explanation'] = {
                'method': 'confidence_based',
                'description': f"Prediction confidence: {result.get('confidence', 0):.2f}"
            }
            
        return result
        
    def _generate_grad_cam(self, image_input: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """
        Generate Grad-CAM heatmap for visual explanation.
        """
        
        try:
            import tensorflow as tf
            
            # Preprocess image
            processed_image = self.preprocess_image(image_input)
            
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Convolutional layer
                    last_conv_layer = layer
                    break
                    
            if last_conv_layer is None:
                logger.warning("No convolutional layer found for Grad-CAM")
                return None
                
            # Create model for Grad-CAM
            grad_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[last_conv_layer.output, self.model.output]
            )
            
            # Generate Grad-CAM
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(processed_image)
                predicted_class = tf.argmax(predictions[0])
                class_output = predictions[:, predicted_class]
                
            # Get gradients
            grads = tape.gradient(class_output, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            return None
            
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        
        info = {
            'model_path': str(self.model_path),
            'target_size': self.target_size,
            'model_loaded': self.model is not None,
            'metadata_available': bool(self.model_metadata)
        }
        
        if self.model:
            info.update({
                'total_parameters': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            })
            
        if self.model_metadata:
            info.update({
                'model_name': self.model_metadata.get('model_name', 'unknown'),
                'training_date': self.model_metadata.get('save_date', 'unknown'),
                'evaluation_accuracy': self.model_metadata.get('evaluation_results', {}).get('accuracy', 'unknown')
            })
            
        return info


class PredictionMonitor:
    """
    Monitor predictions for performance tracking and drift detection.
    """
    
    def __init__(self):
        self.prediction_log = []
        
    def log_prediction(self, prediction_result: Dict, image_path: Optional[str] = None):
        """Log prediction for monitoring."""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result.get('predicted_class'),
            'confidence': prediction_result.get('confidence'),
            'status': prediction_result.get('status'),
            'image_path': image_path,
            'processing_time': prediction_result.get('processing_time')
        }
        
        self.prediction_log.append(log_entry)
        
    def get_performance_stats(self, hours_back: int = 24) -> Dict:
        """Get performance statistics for recent predictions."""
        
        if not self.prediction_log:
            return {'error': 'No predictions logged'}
            
        # Filter recent predictions
        cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
        recent_predictions = [
            log for log in self.prediction_log 
            if datetime.fromisoformat(log['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_predictions:
            return {'error': 'No recent predictions found'}
            
        # Calculate statistics
        total_predictions = len(recent_predictions)
        successful_predictions = sum(1 for log in recent_predictions if log['status'] == 'success')
        high_engagement_predictions = sum(1 for log in recent_predictions 
                                        if log['prediction'] == 'high_engagement')
        
        avg_confidence = np.mean([log['confidence'] for log in recent_predictions 
                                if log['confidence'] is not None])
        
        stats = {
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions,
            'success_rate': successful_predictions / total_predictions if total_predictions > 0 else 0,
            'high_engagement_rate': high_engagement_predictions / successful_predictions if successful_predictions > 0 else 0,
            'average_confidence': float(avg_confidence) if not np.isnan(avg_confidence) else 0,
            'time_period_hours': hours_back,
            'report_timestamp': datetime.now().isoformat()
        }
        
        return stats


def main():
    """Main function to demonstrate prediction functionality."""
    
    logger.info("Product Engagement Prediction Demo")
    
    # Check if model exists
    model_path = Path("models/product_engagement_model.h5")
    
    if model_path.exists():
        # Initialize predictor
        predictor = ProductEngagementPredictor(model_path)
        
        # Get model info
        model_info = predictor.get_model_info()
        print("✅ Model loaded successfully")
        print(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # Test prediction if test images exist
        test_image_dir = Path("data/test/high_engagement")
        if test_image_dir.exists():
            test_images = list(test_image_dir.glob("*.jpg"))
            if test_images:
                sample_image = test_images[0]
                result = predictor.predict_single(sample_image)
                print(f"\n📸 Sample prediction:")
                print(f"Image: {sample_image}")
                print(f"Result: {json.dumps(result, indent=2)}")
        
    else:
        print("❌ Model not found. Train a model first using:")
        print("1. python src/data_acquisition.py")
        print("2. Run the Jupyter notebook for training")
        print("3. Save the trained model")
        
    print("\n💡 Prediction functions available:")
    print("- predict_single(): Single image prediction")
    print("- predict_batch(): Batch processing")
    print("- predict_from_base64(): API-ready base64 prediction")
    print("- predict_with_explanations(): Predictions with visual explanations")


if __name__ == "__main__":
    main()
