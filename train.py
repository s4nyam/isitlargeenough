import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Dict
import ptp_utils
import seq_aligner

# Constants from your code
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class ContrastivePromptRefinement(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Load Stable Diffusion with attention control
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            torch_dtype=torch.float16
        ).to(device)
        self.sd_tokenizer = self.sd_pipeline.tokenizer
        
        # Projection layers for contrastive learning
        self.text_proj = nn.Linear(768, 512)
        self.image_proj = nn.Linear(768, 512)
        
        # Temperature parameter for contrastive loss
        self.temperature = 0.07
        
        self.device = device
        self.to(device)
    
    def encode_text(self, text):
        inputs = self.clip_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_features = self.clip_model.get_text_features(**inputs)
        return self.text_proj(text_features)
    
    def encode_image(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.clip_model.get_image_features(**inputs)
        return self.image_proj(image_features)
    
    def contrastive_loss(self, e_gen, e_task_text, e_task_image):
        # Normalize embeddings
        e_gen = F.normalize(e_gen, dim=-1)
        e_task_text = F.normalize(e_task_text, dim=-1)
        e_task_image = F.normalize(e_task_image, dim=-1)
        
        # Positive pairs
        pos_sim_text = torch.matmul(e_task_text, e_gen.T) / self.temperature
        pos_sim_image = torch.matmul(e_task_text, e_task_image.T) / self.temperature
        
        # Negative pairs (in-batch negatives)
        neg_sim_text = torch.matmul(e_task_text, e_gen.T) / self.temperature
        neg_sim_image = torch.matmul(e_task_text, e_task_image.T) / self.temperature
        
        # Contrastive loss
        logits_text = torch.cat([pos_sim_text, neg_sim_text], dim=1)
        logits_image = torch.cat([pos_sim_image, neg_sim_image], dim=1)
        
        labels = torch.arange(len(e_task_text)).to(self.device)
        
        loss_text = F.cross_entropy(logits_text, labels)
        loss_image = F.cross_entropy(logits_image, labels)
        
        return (loss_text + loss_image) / 2
    
    def generate_with_attention_control(self, prompts, controller):
        """Generate images with attention control from Prompt-to-Prompt"""
        images = ptp_utils.text2image_ldm_stable(
            self.sd_pipeline,
            prompts,
            controller,
            num_inference_steps=NUM_DIFFUSION_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=None,
            low_resource=False
        )
        return images
    
    def generate_refined_prompt(self, e_task_text, top_k=5):
        """Enhanced prompt refinement with attention to key words"""
        # Get most relevant words from original prompt using CLIP attention
        inputs = self.clip_tokenizer(e_task_text, return_tensors="pt").to(self.device)
        outputs = self.clip_model.text_model(**inputs, output_attentions=True)
        
        # Get attention weights and identify important words
        attention = outputs.attentions[-1].mean(dim=1)[0, 0]  # [seq_len, seq_len]
        word_importance = attention.mean(dim=0)[1:-1]  # Exclude [CLS] and [SEP]
        important_indices = torch.topk(word_importance, k=min(top_k, len(word_importance))[1]
        
        # Reconstruct prompt emphasizing important words
        tokens = self.clip_tokenizer.tokenize(e_task_text)
        refined_prompt = " ".join([f"*{tokens[i]}*" if i in important_indices else tokens[i] 
                                 for i in range(len(tokens))])
        
        return f"A highly detailed and realistic {refined_prompt}"
    
    def forward(self, generic_prompt, task_specific_prompt, task_specific_image=None, 
               use_attention_control=True):
        # Encode all inputs
        e_gen = self.encode_text(generic_prompt)
        e_task_text = self.encode_text(task_specific_prompt)
        
        if task_specific_image is not None:
            e_task_image = self.encode_image(task_specific_image)
        else:
            e_task_image = e_task_text.clone()
        
        # Contrastive alignment
        loss = self.contrastive_loss(e_gen, e_task_text, e_task_image)
        
        # Generate refined prompt
        refined_prompt = self.generate_refined_prompt(task_specific_prompt)
        
        # Generate image with optional attention control
        if use_attention_control:
            # Use Prompt-to-Prompt attention control
            controller = AttentionStore()
            generated_image = self.generate_with_attention_control([refined_prompt], controller)
        else:
            # Standard generation
            generated_image = self.sd_pipeline(
                refined_prompt, 
                num_inference_steps=NUM_DIFFUSION_STEPS,
                guidance_scale=GUIDANCE_SCALE
            ).images[0]
        
        return {
            "loss": loss,
            "refined_prompt": refined_prompt,
            "generated_image": generated_image,
            "e_gen": e_gen,
            "e_task_text": e_task_text,
            "e_task_image": e_task_image
        }

# The AttentionControl, AttentionStore, and related classes from your code remain unchanged
# They can be used directly with our enhanced pipeline

def prepare_dataset():
    """Load a dataset of generic and task-specific prompts with images"""
    dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k")
    return dataset

def train_model(model, dataset, epochs=5, batch_size=8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            
            generic_prompts = [p for p in batch["generic_prompt"]]
            task_prompts = [p for p in batch["task_specific_prompt"]]
            task_images = [img for img in batch["image"]]
            
            outputs = model(generic_prompts, task_prompts, task_images)
            loss = outputs["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataset):.4f}")

def evaluate_model(model, test_prompts):
    """Evaluate with attention visualization"""
    results = []
    for prompt_pair in test_prompts:
        generic_prompt, task_prompt = prompt_pair
        
        # Without attention control (baseline)
        output_baseline = model(generic_prompt, task_prompt, use_attention_control=False)
        
        # With attention control
        output_controlled = model(generic_prompt, task_prompt, use_attention_control=True)
        
        results.append({
            "generic_prompt": generic_prompt,
            "task_prompt": task_prompt,
            "refined_prompt": output_controlled["refined_prompt"],
            "baseline_image": output_baseline["generated_image"],
            "controlled_image": output_controlled["generated_image"]
        })
    
    return results

def main():
    # Initialize model
    model = ContrastivePromptRefinement(device=device)
    
    # Prepare dataset
    dataset = prepare_dataset()
    
    # Train model
    train_model(model, dataset)
    
    # Test prompts
    test_prompts = [
        ("A cat", "A flying cat with feathery wings"),
        ("A cloud", "A cloud shaped like dragon"),
        ("A fish", "A fish flying in the sky"),
        ("A garden", "A garden on a comet"),
        ("A tea party", "A tea party of animals")
    ]
    
    # Evaluate
    results = evaluate_model(model, test_prompts)
    
    # Save and display results
    for i, res in enumerate(results):
        print(f"\nExample {i+1}:")
        print(f"Generic prompt: {res['generic_prompt']}")
        print(f"Task prompt: {res['task_prompt']}")
        print(f"Refined prompt: {res['refined_prompt']}")
        res["baseline_image"].save(f"baseline_image_{i}.png")
        res["controlled_image"].save(f"controlled_image_{i}.png")
        print(f"Images saved to baseline_image_{i}.png and controlled_image_{i}.png")

if __name__ == "__main__":
    main()
