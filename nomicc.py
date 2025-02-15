
from nomic import embed
import numpy as np

output = embed.text(
    texts=['The text you want to embed.'],
    model='nomic-embed-text-v1.5',
    task_type='search_document',
)

embeddings = np.array(output['embeddings'])
print(embeddings[0])  # prints: (768,)