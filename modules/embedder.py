from google import genai
from google.genai.types import EmbedContentConfig
# import asyncio
from dotenv import load_dotenv
from os import getenv

load_dotenv()


API_KEY = getenv("GEMINI_API_KEY")
MODEL = getenv("GEMINI_EMBEDDING_MODEL_NAME", "text-multilingual-embedding-002")

class Embedder:
    def __init__(self):
        self.client = genai.Client(api_key=API_KEY)
        
    def embed(self, content: list, dimensions: int):
    # Example response:
    # embeddings=[ContentEmbedding(values=[-0.06302902102470398, 0.00928034819662571, 0.014716853387653828, -0.028747491538524628, ... ],
    # statistics=ContentEmbeddingStatistics(truncated=False, token_count=13.0))]
    # metadata=EmbedContentMetadata(billable_character_count=112)
        response = self.client.models.embed_content(
            model=MODEL,
            contents=content,
            config=EmbedContentConfig(
                output_dimensionality=dimensions
            ),
        )
        return response