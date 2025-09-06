import torch
import torch.nn as nn
from torchvision import models
import os

class ResNet50Embedding(nn.Module):
    """
    ResNet50 model modified for feature extraction (embedding generation).
    Removes the final classification layer to output 2048-dimensional embeddings.
    """
    def __init__(self, pretrained_resnet50):
        super(ResNet50Embedding, self).__init__()
        
        # Copy all layers except the final classification layer (fc)
        self.conv1 = pretrained_resnet50.conv1
        self.bn1 = pretrained_resnet50.bn1
        self.relu = pretrained_resnet50.relu
        self.maxpool = pretrained_resnet50.maxpool
        
        self.layer1 = pretrained_resnet50.layer1
        self.layer2 = pretrained_resnet50.layer2
        self.layer3 = pretrained_resnet50.layer3
        self.layer4 = pretrained_resnet50.layer4
        
        self.avgpool = pretrained_resnet50.avgpool
        # Note: We're NOT copying the fc layer (classifier)
        
    def forward(self, x):
        """
        Forward pass that returns 2048-dimensional embeddings instead of class predictions.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 2048)
        
        return x  # Return embeddings instead of class predictions

class ResNet50Embedding2(nn.Module):
    
    def __init__(self, resnet, embedding_dim=2048, pretrained=True):
        super(ResNet50Embedding2, self).__init__()
        self.resnet = resnet
        
        # Remove the final classification layer (fc layer)
        # ResNet50 originally outputs 1000 classes, we want features instead
        self.features = nn.Sequential(OrderedDict([*(list( self.resnet.named_children())[:-1])]))
        
        # Optional: Add a custom embedding layer to reduce dimensionality
        self.embedding_dim = embedding_dim
        if embedding_dim != 2048:  # 2048 is the original feature size
            self.embedding_layer = nn.Linear(2048, embedding_dim)
        else:
            self.embedding_layer = None
            
    def forward(self, x):
        # Extract features using ResNet50 backbone (without final FC layer)
        features = self.features(x)
        
        # Flatten the features: (batch_size, 2048, 1, 1) -> (batch_size, 2048)
        features = features.view(features.size(0), -1)
        
        # Apply custom embedding layer if specified
        if self.embedding_layer is not None:
            embeddings = self.embedding_layer(features)
        else:
            embeddings = features
            
        return embeddings

from collections import OrderedDict

def load_checkpoint_model(checkpoint_path):
    """
    Load the ResNet50 model from checkpoint file.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the pretrained ResNet50 model structure
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 6162 )
    
    # Load the checkpoint
    try:
        loaded_state_dict = torch.load(
            checkpoint_path, 
            map_location=torch.device('cpu'), 
            weights_only=False
        )
        

        new_state_dict = OrderedDict()
        for key in loaded_state_dict["state_dict"].keys():
            if 'model._orig_mod.model.' in key:
                # Remove the Lightning wrapper prefixes
                new_key = key.replace('model._orig_mod.model.', '')
                new_state_dict[new_key] = loaded_state_dict["state_dict"][key]
        
        # Load the state dict with strict=False to handle missing keys
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using pretrained ResNet50 instead...")

    # feature_extractor = nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
    return model

def convert_to_embedding_model(checkpoint_path, output_path="font_embedding_model.pth"):
    
    resnet50 = load_checkpoint_model(checkpoint_path)
    resnet50.eval()
    
    # Create the embedding model
    print("Creating embedding model...")
    embedding_model = nn.Sequential(OrderedDict([*(list( resnet50.named_children())[:-1])]))
    embedding_model.eval()
    
    # Test the model with a dummy input to ensure it works
    print("Testing embedding model...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image 224x224
        embeddings = embedding_model(dummy_input)
        print(f"Output embedding shape: {embeddings.shape}")  # Should be [1, 2048]
    
    # Save the embedding model
    print(f"Saving embedding model to: {output_path}")
    
    # Save the complete model (architecture + weights)
    torch.save(embedding_model, output_path)
    
    print(f"Embedding model saved successfully!")
    print(f"Complete model: {output_path}")
    
    return embedding_model

def load_embedding_model(model_path):
    print(f"Loading embedding model from: {model_path}")
    
    # Load the complete model
    embedding_model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    embedding_model.eval()
    
    print("Embedding model loaded successfully!")
    return embedding_model

def extract_embeddings(model, image_tensor):
    model.eval()
    with torch.no_grad():
        embeddings = model(image_tensor)
    return embeddings

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "name=4x-epoch=84-step=1649340.ckpt"
    output_model_path = "font_embedding_model.pth"
    
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file '{checkpoint_path}' not found!")
        print("Please ensure the file path is correct.")
        exit(1)
    
    # Convert the model
    try:
        embedding_model = convert_to_embedding_model(checkpoint_path, output_model_path)
        
        # Example usage: Load and test the saved model
        print("\n" + "="*50)
        print("Testing the saved embedding model...")
        
        loaded_model = load_embedding_model(output_model_path)
        
        # Create a dummy image batch for testing
        dummy_images = torch.randn(4, 3, 224, 224)  # Batch of 4 images
        embeddings = extract_embeddings(loaded_model, dummy_images)
        
        print(f"Input batch shape: {dummy_images.shape}")
        print(f"Output embeddings shape: {embeddings.shape}")
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()