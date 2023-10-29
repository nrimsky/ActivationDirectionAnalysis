"""
Plots cosine similarity between / pca of vectors from different layers of the same model / same layers of different models
Usage: python analyze_vectors.py --dataset_name <dataset_name>
"""

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Load vectors
def load_vectors(directory, dataset_name):
    vectors = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            if dataset_name not in filename:
                continue
            path = os.path.join(directory, filename)
            parts = filename[:-3].split('_')
            layer = int(parts[-3])
            model_name = parts[-2]
            if model_name not in vectors:
                vectors[model_name] = {}
            v = torch.load(path)
            vectors[model_name][layer] = v / torch.norm(v)
    return vectors

def analyze_vectors(dataset_name):

    vectors = load_vectors("vectors", dataset_name)

    # Initialize PCA
    pca = PCA(n_components=2)

    # Comparing vectors from different layers of the same model
    for model_name, layers in vectors.items():
        keys = list(layers.keys())
        keys.sort()
        
        # Cosine Similarity
        matrix = np.zeros((len(keys), len(keys)))
        for i, layer1 in enumerate(keys):
            for j, layer2 in enumerate(keys):
                cosine_sim = cosine_similarity(layers[layer1].reshape(1, -1), layers[layer2].reshape(1, -1))
                matrix[i, j] = cosine_sim.flatten()[0]
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=False, xticklabels=keys, yticklabels=keys, cmap='coolwarm')
        plt.title(f"Cosine Similarities between Layers of Model: {model_name}, Dataset: {dataset_name}")
        plt.savefig(f"cosine_similarities_{model_name}_{dataset_name}.png", format='png')

        # PCA Projections
        data = [layers[layer].numpy() for layer in keys]
        projections = pca.fit_transform(data)
        plt.figure(figsize=(10, 8))
        plt.scatter(projections[:, 0], projections[:, 1], c=keys, cmap='viridis')
        plt.colorbar().set_label('Layer Number')
        for i, layer in enumerate(keys):
            plt.annotate(layer, (projections[i, 0], projections[i, 1]))
        plt.title(f"PCA Projections of Layers for Model: {model_name}, Dataset: {dataset_name}")
        plt.savefig(f"pca_layers_{model_name}_{dataset_name}.png", format='png')


    # Remove 'Llama-2-13b-chat-hf' from vectors
    try:
        del vectors['Llama-2-13b-chat-hf']
    except KeyError:
        print("Llama-2-13b-chat-hf not in vectors")

    # Comparing vectors from the same layer but different models
    common_layers = sorted(list(set(next(iter(vectors.values())).keys())))  # Sorted common layers
    model_names = list(vectors.keys())

    def model_label(model_name):
        if "chat" in model_name:
            return "Chat"
        return "Base"

    # Plot PCA of all layers over all models on a single plot with layer, model labels
    plt.clf()
    data = []
    labels = []
    for layer in common_layers:
        for model_name in model_names:
            data.append(vectors[model_name][layer].numpy())
            labels.append(model_label(model_name))
    data = np.array(data)
    projections = pca.fit_transform(data)

    plt.figure(figsize=(10, 8))

    # Separate chat and base data
    chat_data = np.array([projections[i] for i, label in enumerate(labels) if label == "Chat"])
    base_data = np.array([projections[i] for i, label in enumerate(labels) if label == "Base"])

    # Plot chat models with crosses
    plt.scatter(chat_data[:, 0], chat_data[:, 1], c='blue', marker='x', label='Chat')

    # Plot base models with circles
    plt.scatter(base_data[:, 0], base_data[:, 1], c='red', marker='o', label='Base')

    # Colorbar and legend
    plt.legend()

    # Annotate each point
    for i, layer in enumerate(common_layers):
        for j, model_name in enumerate(model_names):
            plt.annotate(str(layer), (projections[i*len(model_names)+j, 0], projections[i*len(model_names)+j, 1]))

    plt.title(f"PCA Projections of Layers for Models: {', '.join(model_names)}, Dataset: {dataset_name}")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.savefig(f"pca_layers_all_models_{dataset_name}.png", format='png')

    # Plotting cosine similarity between the first 2 models per layer number
    model1_name, model2_name = model_names[0], model_names[1]
    cosine_sims_per_layer = []

    for layer in common_layers:
        cosine_sim = cosine_similarity(vectors[model1_name][layer].reshape(1, -1), vectors[model2_name][layer].reshape(1, -1))[0][0]
        cosine_sims_per_layer.append(cosine_sim)

    plt.figure(figsize=(10, 6))
    plt.plot(common_layers, cosine_sims_per_layer, marker='o', linestyle='-')
    plt.xlabel("Layer Number")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Cosine Similarity per Layer between {model1_name} and {model2_name}, Dataset: {dataset_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cosine_sim_per_layer_{model1_name}_vs_{model2_name}_{dataset_name}.png", format='png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="test")
    args = parser.parse_args()
    analyze_vectors(args.dataset_name)
