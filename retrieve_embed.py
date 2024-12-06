import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import json
from models import CNN

class EmbeddingExtractor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize with the same model architecture
        self.model = CNN(num_classes=10)
        # Load the state dict
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize embedding storage
        self.embedding = None
        
        # Hook function to capture embeddings after global pooling
        def hook(module, input, output):
            # Output shape will be [batch_size, 128, 1, 1]
            # We want to capture this before the view operation
            self.embedding = output.detach()
        
        # Register hook on the global pooling layer
        self.model.global_pool.register_forward_hook(hook)
        
        # Transform pipeline - matching the CIFAR10 preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # CIFAR10 image size
            transforms.ToTensor(),
            # If you used normalization during training, uncomment these lines:
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.embeddings = {}
    
    def extract_embedding(self, image_path):
        """Extract embedding for a single image"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass to trigger the hook
        with torch.no_grad():
            _ = self.model(image)
            # Reshape the embedding to match the architecture [128] features
            embedding = self.embedding.squeeze().cpu().numpy()
            
        return embedding
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in directory structure and save embeddings"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each class directory
        for class_dir in range(10):
            class_path = os.path.join(input_dir, str(class_dir))
            output_class_path = os.path.join(output_dir, str(class_dir))
            os.makedirs(output_class_path, exist_ok=True)
            
            if not os.path.exists(class_path):
                continue
                
            print(f"Processing class {class_dir}...")
            
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    
                    try:
                        # Extract embedding
                        embedding = self.extract_embedding(image_path)
                        
                        # Save embedding - should be 128 dimensional
                        output_path = os.path.join(output_class_path, 
                                                 os.path.splitext(image_file)[0] + '.npy')
                        np.save(output_path, embedding)
                        
                        # Store metadata
                        self.embeddings[image_file] = {
                            'class': class_dir,
                            'embedding_path': output_path,
                            'original_path': image_path,
                            'embedding_size': embedding.shape
                        }
                        
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
    
    def save_metadata(self, output_dir):
        """Save metadata for all processed images"""
        metadata_path = os.path.join(output_dir, 'embedding_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.embeddings, f, indent=2)
    
    def create_ml_dataset(self, output_dir):
        """Create a consolidated dataset suitable for ML training"""
        features = []
        labels = []
        image_ids = []
        
        for image_file, metadata in self.embeddings.items():
            embedding = np.load(metadata['embedding_path'])
            features.append(embedding.flatten())  # Should already be 128-dimensional
            labels.append(metadata['class'])
            image_ids.append(image_file)
        
        X = np.array(features)
        y = np.array(labels)
        
        np.save(os.path.join(output_dir, 'X_features.npy'), X)
        np.save(os.path.join(output_dir, 'y_labels.npy'), y)
        
        with open(os.path.join(output_dir, 'image_ids.json'), 'w') as f:
            json.dump(image_ids, f)
        
        print(f"Dataset created with features shape: {X.shape}")
        print(f"Each feature vector represents 128 dimensions from the global pooling layer")
        
        return X, y

# Usage example
if __name__ == "__main__":
    model_path = "/Users/jinjiahui/Desktop/CS470Project/models/target_model.mod"
    input_dir = "/Users/jinjiahui/Desktop/CS470Project/cifar10_test_images_by_class"
    output_dir = "/Users/jinjiahui/Desktop/CS470Project/embed_cifar10_test_images_by_class"
    
    extractor = EmbeddingExtractor(model_path)
    extractor.process_directory(input_dir, output_dir)
    extractor.save_metadata(output_dir)
    X, y = extractor.create_ml_dataset(output_dir)
    print(f"Created dataset with shape: {X.shape}, {y.shape}")

    # data = np.load("/Users/jinjiahui/Desktop/CS470Project/embed_cifar10_test_images_by_class/0/test_image_00010.npy")
    # print(data)
