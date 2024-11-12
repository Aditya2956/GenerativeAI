from matplotlib.patches import FancyArrowPatch
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def get_embedding(text: str) -> torch.Tensor:

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def plot_embeddings(embeddings: np.ndarray, sentences: list) -> None:
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue')

    plt.annotate(sentences[0], (reduced_embeddings[0, 0], reduced_embeddings[0, 1]), fontsize=9)
    plt.annotate(sentences[1], (reduced_embeddings[1, 0], reduced_embeddings[1, 1]), fontsize=9)
    plt.annotate(sentences[2], (reduced_embeddings[2, 0], reduced_embeddings[2, 1]), fontsize=9)
    plt.annotate(sentences[3], (reduced_embeddings[3, 0], reduced_embeddings[3, 1]), fontsize=9)

    # Adjusted FancyArrowPatch between sentences[0] and sentences[1]
    arrow1 = FancyArrowPatch((reduced_embeddings[0, 0], reduced_embeddings[0, 1]),
                             (reduced_embeddings[1, 0], reduced_embeddings[1, 1]),
                             arrowstyle='->', color='red', mutation_scale=8)  # Decreased mutation_scale
    plt.gca().add_patch(arrow1)

    # Adjusted FancyArrowPatch between sentences[2] and sentences[3]
    arrow2 = FancyArrowPatch((reduced_embeddings[2, 0], reduced_embeddings[2, 1]),
                             (reduced_embeddings[3, 0], reduced_embeddings[3, 1]),
                             arrowstyle='->', color='red', mutation_scale=8)  # Decreased mutation_scale
    plt.gca().add_patch(arrow2)

    plt.title('Plotting Attention/Context aware Vectors')
    plt.xlabel('Dimension Reduced Embedding Vector X Axis')
    plt.ylabel('Dimension Reduced Embedding Vector Y Axis')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    sentence1 = "I am a man"
    sentence2 = "I am a woman"
    sentence3 = "I am an actor"
    sentence4 = "I am an actress"

    print (model)
    print("Generating embeddings for all sentences...")

    embedding1 = get_embedding(sentence1)
    embedding2 = get_embedding(sentence2)
    embedding3 = get_embedding(sentence3)
    embedding4 = get_embedding(sentence4)


    combined_embeddings = torch.cat([embedding1, embedding2, embedding3, embedding4], dim=0).detach().numpy()

    sentences = [sentence1, sentence2, sentence3, sentence4]
    plot_embeddings(combined_embeddings, sentences)