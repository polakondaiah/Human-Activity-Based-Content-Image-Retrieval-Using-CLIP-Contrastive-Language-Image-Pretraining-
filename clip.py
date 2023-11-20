import os
import json
import torch
from PIL import Image
from torchvision import transforms
from clip import load  # Assuming you're using OpenAI's CLIP
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Constants - Paths to your data

QUERY_DATA_PATH = './data/query_images'
GALLERY_DATA_PATH = './data/gallery'

QUERY_IMAGE_INFO_JSON = './data/query_info.json'
GALLERY_IMAGE_INFO_JSON = './data/gallery_info.json'

# Load JSON mapping
def load_json_mapping(json_file_path):
    with open(json_file_path, 'r') as file:
        mapping = json.load(file)
    return mapping

# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by CLIP
        transforms.ToTensor(),  # Convert to tensor
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

# Load Data
def load_data(dataset_path, json_mapping):
    data = []
    for img_filename in os.listdir(dataset_path):
        if img_filename in json_mapping:
            full_path = os.path.join(dataset_path, img_filename)
            image = preprocess_image(full_path)
            label = json_mapping[img_filename]
            data.append((image, label))
    return data

# CLIP Model Initialization
def initialize_clip_model():
    model, preprocess = load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    return model, preprocess

# Function to get image embeddings using CLIP
def get_image_embeddings(model, image_data):
    images, _ = zip(*image_data)  # Separate images and labels
    images = torch.stack(images)  # Stack images to create a batch
    images = images.to(next(model.parameters()).device)  # Move images to the model's device

    with torch.no_grad():
        embeddings = model.encode_image(images).float()
    return embeddings


# Function for image retrieval
def retrieve_images(query_embeddings, gallery_embeddings, top_k=50):
    # Move tensors to CPU and convert to NumPy arrays
    query_embeddings_np = query_embeddings.cpu().numpy()
    gallery_embeddings_np = gallery_embeddings.cpu().numpy()

    # Compute cosine similarity
    similarities = cosine_similarity(query_embeddings_np, gallery_embeddings_np)

    # Retrieve indices of top-k similar images
    retrieved_indices = np.argsort(-similarities, axis=1)[:, :top_k]
    return retrieved_indices



# Function to evaluate the model
def evaluate_model(retrieved_indices, query_labels, gallery_labels, k_values=[1, 10, 50]):
    mAP_scores = {}
    mean_rank = []

    for k in k_values:
        average_precisions = []
        for i, indices in enumerate(retrieved_indices):
            relevant_ranks = [idx for idx, gallery_idx in enumerate(indices) if gallery_labels[gallery_idx] == query_labels[i]]
            if relevant_ranks:
                first_relevant_rank = relevant_ranks[0] + 1  # +1 because ranks start from 1, not 0
                mean_rank.append(first_relevant_rank)
                relevant_items = len(relevant_ranks[:k])
                average_precision = relevant_items / k
            else:
                average_precision = 0
            average_precisions.append(average_precision)
        mAP_scores[k] = np.mean(average_precisions)

    mean_rank_score = np.mean(mean_rank) if mean_rank else None
    return mAP_scores, mean_rank_score



# Main Execution Flow
if __name__ == "__main__":
    query_json_mapping = load_json_mapping(QUERY_IMAGE_INFO_JSON)
    gallery_json_mapping = load_json_mapping(GALLERY_IMAGE_INFO_JSON)

    query_data = load_data(QUERY_DATA_PATH, query_json_mapping)
    gallery_data = load_data(GALLERY_DATA_PATH, gallery_json_mapping)

    clip_model, clip_preprocess = initialize_clip_model()

    # Extract embeddings
    query_embeddings = get_image_embeddings(clip_model, query_data)
    gallery_embeddings = get_image_embeddings(clip_model, gallery_data)

    # Retrieve images
    retrieved_indices = retrieve_images(query_embeddings, gallery_embeddings)

    # Evaluate the model
    query_labels = [label for _, label in query_data]  # Extract labels from query data
    gallery_labels = [label for _, label in gallery_data]  # Extract labels from gallery data

   # Evaluate the model
    evaluation_results, mean_rank_score = evaluate_model(retrieved_indices, query_labels, gallery_labels)
    print(f"Evaluation Results: {evaluation_results}")
    print(f"Mean Rank: {mean_rank_score}")

