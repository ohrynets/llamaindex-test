"""Multi-document agents Pack."""

from typing import Any, Dict, List

from llama_index.core import Settings, SummaryIndex, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.schema import Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata, RetrieverTool
from llama_index.llms.openai import OpenAI
from llama_index.core.llms.llm import LLM
import pymupdf4llm
from llama_index.core.readers.base import BaseReader
from llama_index.core.agent import ReActAgent
import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.response_synthesizers import Refine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.prompts.system import MARKETING_WRITING_ASSISTANT

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Type, Union
from assistant.multi_document_agent.personal_assitant import MultiDocumentAssistantAgentsPack

from llama_index.core.agent import (
    StructuredPlannerAgent,
    FunctionCallingAgentWorker,
    ReActAgentWorker,
)
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    NodeRelationship,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)

import uuid
import duckdb


class ParquetFileReader(BaseReader):
    def __init__(self, text_field: str = "text", id_field : str = "page_id", metadata_schema: tuple = None):       
       self.text_field = text_field
       self.id_field = id_field
       self.metadata_schema = metadata_schema
       
    def load_data(self, file, extra_info={}):
        metada_fields = ",".join([f'"{r}" := {r}' for r in self.metadata_schema])
        records = duckdb.query(f"SELECT {self.text_field}, {self.id_field}, struct_pack({metada_fields}) FROM read_parquet('{file}')")
        # load_data returns a list of Document objects
        nodes = []
        for row in records.fetchall():
            metadata = row[2]
            doc = Document(text=row[0], id_=row[1], extra_info={**extra_info, **metadata})
            nodes.append(doc)
        return nodes



    
class ParquetDocumentAssistantAgentsPack(MultiDocumentAssistantAgentsPack):
    """Multi-document Agents pack.
            main_llm=azurellm, agent_llm=azurellm, 
            embeding_llm=embed_model, 
            docs_store_path=docs_store_path, 
            pdf_images_path=pdf_images_path,
            storage_dir=storage_dir, verbose=True, number_of_docs=1
    """

    def __init__(
        self,
        main_llm: LLM,
        agent_llm: LLM,
        embeding_llm: LLM,
        docs_store_path: str,
        storage_dir: str,
        pdf_images_path: str,
        verbose: bool = False,
        number_of_files: int = None,
        text_field = "text", 
        id_field = "page_id", 
        metadata_schema = ("url", "title", "status"),
        **kwargs: Any,
    ) -> None:
        self.text_field = text_field
        self.id_field = id_field
        self.metadata_schema = metadata_schema

        super().__init__(
                main_llm,
                agent_llm,
                embeding_llm,
                docs_store_path,
                storage_dir,
                pdf_images_path,
                verbose,
                number_of_files,
                **kwargs)        
    
    def build_index(self):
        
        file_extractor={".parquet": ParquetFileReader(text_field=self.text_field, id_field=self.id_field, metadata_schema=self.metadata_schema)}                
        reader = SimpleDirectoryReader(input_dir=self.docs_store_path, num_files_limit=self.number_of_files, file_extractor=file_extractor)        
        
        documents = reader.load_data(show_progress=True)
        
        transformations=[   
                self.embed_model
            ]
        vector_index = VectorStoreIndex(nodes=documents, transformations=transformations, show_progress=True)        
        return vector_index