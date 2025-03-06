import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor

# Custom Dataset Class
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            caption_dir (str): Path to the directory with captions.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.transform = transform

        # Get list of image and caption files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.caption_files = sorted([f for f in os.listdir(caption_dir) if f.endswith('.txt')])

        # Ensure that each image has a corresponding caption file
        assert len(self.image_files) == len(self.caption_files), "Number of images and captions must match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        # Load caption
        caption_path = os.path.join(self.caption_dir, self.caption_files[idx])
        with open(caption_path, "r") as f:
            caption = f.read().strip()

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        return image, caption

# Preprocessing Function
def preprocess_dataset(image_dir, caption_dir, batch_size=32):
    """
    Preprocesses the dataset and creates a DataLoader.

    Args:
        image_dir (str): Path to the directory with images.
        caption_dir (str): Path to the directory with captions.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader for the dataset.
    """
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # Normalize with CLIP stats
    ])

    # Create dataset
    dataset = ImageCaptionDataset(image_dir, caption_dir, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# Example Usage
if __name__ == "__main__":
    # Paths to image and caption folders
    image_dir = "path/to/images"
    caption_dir = "path/to/captions"

    # Preprocess dataset and create DataLoader
    dataloader = preprocess_dataset(image_dir, caption_dir, batch_size=32)

    # Load CLIP processor for tokenizing captions
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Iterate through the DataLoader
    for batch_idx, (images, captions) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print("Images shape:", images.shape)  # Should be [batch_size, 3, 224, 224]
        print("Captions:", captions)

        # Tokenize captions using CLIP processor
        text_inputs = clip_processor(text=captions, return_tensors="pt", padding=True)
        print("Tokenized captions:", text_inputs.input_ids.shape)  # Should be [batch_size, sequence_length]

        # Example: Break after first batch for demonstration
        break
