import pandas as pd
import numpy as np
import os
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description="Cosine Similarity Heatmap")

    parser.add_argument("--tuning_method", default=None)
    parser.add_argument("--model", default='vit_base')

    return parser

args = get_args_parser().parse_args()

# Read the CSV file
base_path = '/disk/scratch2/raman/Layer-Masking/'
csv_path = os.path.join(base_path, args.model, args.tuning_method + '_' + args.model + '.csv')
df = pd.read_csv(csv_path)

# Load the numpy vectors
vectors = []
for index, row in df.iterrows():
    vector_path = row['Vector Path']
    #print(vector_path)
    vector = np.load(vector_path)
    vectors.append(vector)

# Compute the cosine similarity between each pair of vectors
num_vectors = len(vectors)
similarity_matrix = np.zeros((num_vectors, num_vectors))

for i in range(num_vectors):
    for j in range(num_vectors):
        cos_dist = distance.cosine(vectors[i], vectors[j])
        cos_similarity = 1 - cos_dist
        similarity_matrix[i, j] = cos_similarity
        print("Cosine Sim b/w vector {} and {}: ".format(str(i), str(j)), cos_similarity)

# Create a DataFrame with the similarity results
similarity_df = pd.DataFrame(similarity_matrix)

# Create a confusion matrix using seaborn and save it as a figure
plt.figure(figsize=(10, 10))

sns.heatmap(similarity_df, annot=True, fmt='.2f', cmap='coolwarm')
plt.xlabel("Vectors")
plt.ylabel("Vectors")
plt.title("Cosine Similarity between different randomly generated vectors for {} method".format(args.tuning_method))
plt.savefig('cosine_similarity_cm_{}.png'.format(args.tuning_method))
