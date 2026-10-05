from django.conf import settings
from .llm_client import client

def embed_texts(texts: list[str],batch_size:int=100)-> list[list[float]]:
    """This function convert the list of texts into vectors of floating pointing numbers
       and it is responsible for sending the requests in batches (like upto 100 requests per batch for api)
    """
    vectors = []
    for i in range(0,len(texts),batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        for item in sorted(resp.data,key = lambda d : d.index):
            if len(item.embedding) != settings.EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Expected {settings.EMBEDDING_DIMENSIONS} dims, "
                    f"got {len(item.embedding)} from {settings.EMBEDDING_MODEL}"
                )
            vectors.append(item.embedding)
    return vectors

def embed_text(text : str)->list[float]:
    return embed_texts([text])[0]