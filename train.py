import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load CLIP and Stable Diffusion Models
def load_models():
    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Load Stable Diffusion model
    stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    stable_diffusion_pipe = stable_diffusion_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    return clip_model, clip_processor, clip_text_model, clip_tokenizer, stable_diffusion_pipe

# Step 2: Load Dataset
def load_custom_dataset():
    # Replace with your dataset loading logic
    # Example: Load a dataset from Hugging Face Datasets
    dataset = load_dataset("path/to/your/dataset")
    return dataset

# Step 3: Contrastive Learning for Embedding Alignment
def contrastive_loss(text_embeddings, image_embeddings, temperature=0.07):
    # Normalize embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    # Compute similarity matrix
    logits = (text_embeddings @ image_embeddings.T) / temperature
    labels = torch.arange(len(logits)).to(logits.device)

    # Cross-entropy loss
    loss_text = torch.nn.functional.cross_entropy(logits, labels)
    loss_image = torch.nn.functional.cross_entropy(logits.T, labels)
    loss = (loss_text + loss_image) / 2
    return loss

# Step 4: Train the Model
def train_model(clip_model, dataset, epochs=5, batch_size=32):
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)
    clip_model.train()

    for epoch in range(epochs):
        for batch in dataset:
            # Preprocess batch (text and images)
            inputs = clip_processor(text=batch["text"], images=batch["image"], return_tensors="pt", padding=True)
            inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}

            # Forward pass
            outputs = clip_model(**inputs)
            text_embeddings = outputs.text_embeds
            image_embeddings = outputs.image_embeds

            # Compute contrastive loss
            loss = contrastive_loss(text_embeddings, image_embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Step 5: Generate Task-Specific Images
def generate_image(query, clip_model, clip_tokenizer, stable_diffusion_pipe):
    # Encode query using CLIP text encoder
    inputs = clip_tokenizer(query, return_tensors="pt", padding=True)
    text_embeddings = clip_model.get_text_features(**inputs)

    # Generate image using Stable Diffusion
    generated_image = stable_diffusion_pipe(prompt=query, guidance_scale=7.5).images[0]
    return generated_image

# Step 6: Evaluate Results
def evaluate_results(query, generated_image, clip_model, clip_processor):
    # Encode query and generated image
    text_inputs = clip_processor(text=query, return_tensors="pt", padding=True)
    image_inputs = clip_processor(images=generated_image, return_tensors="pt", padding=True)

    text_embeddings = clip_model.get_text_features(**text_inputs)
    image_embeddings = clip_model.get_image_features(**image_inputs)

    # Compute cosine similarity
    similarity = cosine_similarity(text_embeddings.detach().numpy(), image_embeddings.detach().numpy())
    return similarity

# Main Function
def main():
    # Load models
    clip_model, clip_processor, clip_text_model, clip_tokenizer, stable_diffusion_pipe = load_models()

    # Load dataset
    dataset = load_custom_dataset()

    # Train the model
    train_model(clip_model, dataset)

    # Take user query as input
    query = input("Enter your query: ")

    # Generate task-specific image
    generated_image = generate_image(query, clip_model, clip_tokenizer, stable_diffusion_pipe)

    # Evaluate results
    similarity_score = evaluate_results(query, generated_image, clip_model, clip_processor)
    print(f"Cosine Similarity: {similarity_score}")

    # Save or display the generated image
    generated_image.save("generated_image.png")
    print("Image saved as generated_image.png")

if __name__ == "__main__":
    main()
