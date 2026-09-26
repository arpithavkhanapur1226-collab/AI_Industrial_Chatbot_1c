import faiss
import numpy as np

# Create three example vectors
vectors = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.9, 0.1, 0.0]
], dtype="float32")

# Create FAISS index
index = faiss.IndexFlatL2(3)

# Add vectors to FAISS
index.add(vectors)

print("Number of vectors stored:", index.ntotal)

# Search vector
query = np.array([
    [1.0, 0.0, 0.0]
], dtype="float32")

# Find the two most similar vectors
distances, indices = index.search(query, 2)

print("\nClosest vector indices:")
print(indices)

print("\nDistances:")
print(distances)