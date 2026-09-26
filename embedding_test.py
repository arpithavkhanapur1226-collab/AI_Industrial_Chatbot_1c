from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentences
sentences = [
    "An ultrasonic sensor measures distance using sound waves.",
    "Ultrasonic sensors are used for obstacle detection."
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])