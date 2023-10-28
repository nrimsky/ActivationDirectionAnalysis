"""
Plots cosine similarity between / pca of vectors from different layers of the same model / same layers of different models
Usage: python analyze_vectors.py --dataset_name <dataset_name>
"""

import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load vectors
def load_vectors(directory):
    vectors = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            path = os.path.join(directory, filename)
            parts = filename[:-3].split('_')
            layer = int(parts[-3])
            model_name = parts[-2]
            dataset_name = parts[-1]
            if model_name not in vectors:
                vectors[model_name] = {}
            if layer not in vectors[model_name]:
                vectors[model_name][layer] = {}
            vectors[model_name][layer][dataset_name] = torch.load(path)
    return vectors

def analyze_vectors():
    vectors = load_vectors("vectors")
    try:
        del vectors['Llama-2-13b-chat-hf']
    except KeyError:
        pass
    common_layers = sorted(list(set(next(iter(vectors.values())).keys())))
    model_names = list(vectors.keys())
    model1_name, model2_name = model_names[0], model_names[1]
    cosine_sims_per_layer = defaultdict(list)
    for layer in common_layers:
        for dataset_name in vectors[model1_name][layer]:
            cosine_sim = cosine_similarity(vectors[model1_name][layer][dataset_name].reshape(1, -1), vectors[model2_name][layer][dataset_name].reshape(1, -1))
            cosine_sims_per_layer[dataset_name].append(cosine_sim.flatten()[0])
    plt.figure(figsize=(10, 6))
    for dataset_name, cosine_sims in cosine_sims_per_layer.items():
        plt.plot(common_layers, cosine_sims, label=dataset_name, marker='o')

    # ensure all x ticks
    plt.xticks(common_layers)
    plt.grid()
    plt.xlabel("Layer Number")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Cosine Similarity between representation vectors: Llama 2 7B Base and Chat")
    plt.legend()
    plt.savefig(f"cosine_similarities_{model1_name}_{model2_name}.png", format='png')

if __name__ == "__main__":
    analyze_vectors()
