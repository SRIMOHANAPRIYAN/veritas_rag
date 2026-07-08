"""Common utilities for VeritasRAG."""
import os

# Bypass Keras segmentation faults on Python 3.13 during transformers import
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
