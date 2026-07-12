import sys
print("Starting...")
import hydra
print("Imported hydra")
from sentence_transformers import SentenceTransformer
print("Imported SentenceTransformer")
from src.ingestion.indexer import Indexer
print("Imported Indexer")
print("Done!")
