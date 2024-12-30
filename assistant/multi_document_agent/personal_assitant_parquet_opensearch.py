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
from assistant.multi_document_agent.personal_assitant_parquet import ParquetDocumentAssistantAgentsPack, ParquetFileReader
from llama_index.core.postprocessor import LLMRerank 

from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE

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
from llama_index.core.response_synthesizers import ( 
    ResponseMode,
)
from llama_index.core.schema import (
    MetadataMode,
)
from llama_index.core.prompts import PromptTemplate
import uuid
import duckdb
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient

class OpensearchParquetDocumentAssistantAgentsPack(ParquetDocumentAssistantAgentsPack):
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
        opensearch_endpoint : str = None,
        embedding_field: str = None,
        index_name: str = None,
        load_data: bool = True,
        **kwargs: Any,
    ) -> None:
        self.text_field = text_field
        self.id_field = id_field
        self.metadata_schema = metadata_schema
        self.opensearch_endpoint = opensearch_endpoint
        self.embedding_field = embedding_field
        self.index_name = index_name
        self.load_data = load_data
        super().__init__(
                main_llm,
                agent_llm,
                embeding_llm,
                docs_store_path,
                storage_dir,
                pdf_images_path,
                verbose,
                number_of_files,
                load_data=load_data,
                **kwargs)        
    
    def initialize_vector_index(self, docs_store_path, storage_dir, load_data=False):
        self._init_awsauth()
        client = OpensearchVectorClient(
            self.opensearch_endpoint, self.index_name, 1024, embedding_field=self.embedding_field, text_field=self.text_field,
            http_auth=self.awsauth, use_ssl=True,
            verify_certs=True, connection_class=RequestsHttpConnection,
            settings={"index.number_of_shards": 2}
        )
        # initialize vector store
        self.vector_store = OpensearchVectorStore(client)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        if load_data:
            self.vector_index = self.build_index()       
            self.vector_index.storage_context.persist()
        else:
            self.vector_index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model = self.embed_model)   
            

    def build_index(self):
        file_extractor={".parquet": ParquetFileReader(text_field=self.text_field, id_field=self.id_field, metadata_schema=self.metadata_schema)}                
        reader = SimpleDirectoryReader(input_dir=self.docs_store_path, num_files_limit=self.number_of_files, file_extractor=file_extractor)        
        
        documents = reader.load_data(show_progress=True)
        
        transformations=[   
                self.embed_model
            ]
        return VectorStoreIndex.from_documents(documents=documents, storage_context=self.storage_context, transformations=transformations, show_progress=True)    
    
    def delete_index(index_name, endpoint):
        delete_url = f'{endpoint}/{index_name}'
        response = requests.delete(delete_url, auth=self.awsauth, headers=headers)
    
    def _init_awsauth(self) -> None:
        region_name=os.environ['AWS_REGION']
        client = boto3.client("opensearch", region_name=region_name)
        client_credentials = client._get_credentials()
        self.credentials = client_credentials.get_frozen_credentials()        
        self.awsauth = AWS4Auth(self.credentials.access_key, self.credentials.secret_key, region_name, "es", session_token=self.credentials.token)
