import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM
from deepeval.models import DeepEvalBaseEmbeddingModel

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Any, Optional, List

class DeepEvalLlamaIndexModel(DeepEvalBaseLLM):
    def __init__(self, llama_index_model: LLM):
        self.llm: LLM = llama_index_model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:        
        result = self.llm.complete(prompt, formatted = True)
        return str(result)

    async def a_generate(self, prompt: str) -> str:
        result = await self.llm.acomplete(prompt, formatted = True)
        return str(result)

    def get_model_name(self):
        return self.llm.metadata.model_name
    
class DeepEvalLlamaIndexEmbedingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, llama_index_embedding: BaseEmbedding):
        self.embedding: BaseEmbedding = llama_index_embedding

    def load_model(self):
        return self.model

    def embed_text(self, text, *args, **kwargs) -> List[float]:
        float_list = self.embedding.get_text_embedding(text).embedding    
        return float_list
    
     
    def a_embed_text(self, text, *args, **kwargs) -> List[float]:
        float_list = self.embedding.aget_text_embedding(text).embedding    
        return float_list

    def embed_texts(self, texts, *args, **kwargs) -> List[List[float]]:
        float_list = self.embedding.get_text_embedding_batch(texts).embedding    
        return float_list

    def a_embed_texts(self, texts, *args, **kwargs) -> List[List[float]]:
        float_list = self.embedding.aget_text_embedding_batch(texts).embedding    
        return float_list
    
    def get_model_name(self):
        return self.embedding.model_name