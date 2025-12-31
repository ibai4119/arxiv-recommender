import numpy as np

from arxiv_rec.models.index import VectorIndex


def test_vector_index_add_and_search():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = VectorIndex.from_embeddings(embeddings)

    assert index.size == 2

    scores, indices = index.search(np.array([[1.0, 0.0]], dtype=np.float32), k=1)
    assert scores.shape == (1, 1)
    assert indices.shape == (1, 1)
