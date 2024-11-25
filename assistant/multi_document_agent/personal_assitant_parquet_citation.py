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
from assistant.multi_document_agent.personal_assitant_parquet import ParquetDocumentAssistantAgentsPack
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

CITATION_QA_TEMPLATE_CUSTOM = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "Keep the original Markdown formating from the sources."
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CITATION_REFINE_TEMPLATE_CUSTOM = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Please format the answer using Markdown formating preserving the original source Markdown tags."
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. "
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!"
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

class CitationParquetDocumentAssistantAgentsPack(ParquetDocumentAssistantAgentsPack):
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
    
    def create_query_engine(self):
        # self.reranker = LLMRerank(llm=self.main_llm, choice_batch_size=5, top_n=5)
        self.reranker = ColbertRerank(
                                        top_n=5,
                                        model="colbert-ir/colbertv2.0",
                                        tokenizer="colbert-ir/colbertv2.0",
                                        keep_retrieval_score=True,
                                    )
        postprocessor = SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
        
        
        self.vector_query_engine = CitationQueryEngine.from_args(
            self.vector_index,
            similarity_top_k=3,
            # here we can control how granular citation sources are, the default is 512
            citation_chunk_size=512,
            llm=self.agent_llm,  
            node_postprocessors=[postprocessor],
            citation_qa_template=CITATION_QA_TEMPLATE_CUSTOM,
            citation_refine_template=CITATION_REFINE_TEMPLATE_CUSTOM,
        )
    
        self.vector_query_engine_reranked = CitationQueryEngine.from_args(
            self.vector_index,
            similarity_top_k=2,
            # here we can control how granular citation sources are, the default is 512
            citation_chunk_size=512,
            llm=self.agent_llm,
            node_postprocessors=[self.reranker, postprocessor],
            response_mode = ResponseMode.COMPACT,
            citation_qa_template=CITATION_QA_TEMPLATE_CUSTOM,
            citation_refine_template=CITATION_REFINE_TEMPLATE_CUSTOM,
            metadata_mode=MetadataMode.NONE
        )