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
from llama_index.core.postprocessor import LLMRerank 
from llama_index.core import QueryBundle
from llama_index.core.postprocessor import SimilarityPostprocessor
import boto3
from requests_aws4auth import AWS4Auth

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
#Download model python3 -m spacy download en_core_web_md

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownNodeParser

import nltk
import spacy
import uuid


class PdfFileReader(BaseReader):
    def __init__(self, pdf_images_path: str, page_chunks: bool = False):
       self.pdf_images_path = pdf_images_path
       self.page_chunks = page_chunks
       
    def load_data(self, file, extra_info={}):
        md_content = pymupdf4llm.to_markdown(file, write_images=True, page_chunks=self.page_chunks, image_path=self.pdf_images_path)
        # load_data returns a list of Document objects
        nodes = []
        if self.page_chunks:
            for d in md_content:
                # res = mrkdown_parser.aget_nodes_from_documents(d)
                # for n in res:
                    # print(n)
                doc_id = f"{d['metadata']['title']}:{d['metadata']['page']}"
                doc = Document(text=d['text'], id_=doc_id, extra_info={**extra_info, **d['metadata']})
                nodes.append(doc)
        else:
            doc_id = f"{file}"
            doc = Document(text=md_content, id_=doc_id)                
            nodes.append(doc)
        return nodes

class MultiDocumentAssistantAgentsPack(BaseLlamaPack):
    """Multi-document Agents pack.

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
        similarity_cutoff: float = 0.65,
        load_data: bool = False,  
        **kwargs: Any,
    ) -> None:
        """Init params."""
        Settings.llm = main_llm
        self.embed_model = embeding_llm
        Settings.embed_model = self.embed_model
        
        # Build agents dictionary
        self.agents = {}
        self.pdf_images_path = pdf_images_path
        self.docs_store_path = docs_store_path
        self.verbose = verbose
        self.agent_llm = agent_llm
        self.main_llm = main_llm
        self.number_of_files = number_of_files
        self.similarity_cutoff = similarity_cutoff
        self.agent_llm = agent_llm
        self.load_data = load_data
        # this is for the baseline
        
        
        self.initialize_vector_index(docs_store_path, storage_dir)
        
        self.create_query_engine()
        query_engine_tools = [
                QueryEngineTool (
                    query_engine=self.vector_query_engine,
                    metadata=ToolMetadata(
                        name="vector_tool",
                        description="Useful to answer question based on relevant documents. ",
                    )
                )
        ] 

        self.tool_interactive_reflection_agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=query_engine_tools, llm=agent_llm, verbose=self.verbose
        )
        
        self.main_agent = ReActAgentWorker.from_tools(tools=query_engine_tools, llm=main_llm,
             verbose=self.verbose
        )

        chat_history = [
            ChatMessage(
                content="You are an assistant that generates answer based on documents. Please provide the answer in John Cleese style.",
                role=MessageRole.SYSTEM,
            )
        ]
        self.top_agent = self.main_agent.as_agent(chat_history=chat_history, verbose=self.verbose)

    def initialize_vector_index(self, docs_store_path, storage_dir, load_data=False):
        if not load_data:
            self.vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=storage_dir)
            )
        else:
            print(f"Folder with documents:{docs_store_path}. Loadding ...")
            self.vector_index = self.build_index()
            # save the initial index
            self.vector_index.storage_context.persist(persist_dir=storage_dir)

    

    def create_query_engine(self):
        self.reranker = LLMRerank(llm=self.main_llm, choice_batch_size=5, top_n=5)
        self.reranker = ColbertRerank(
                                        top_n=5,
                                        model="colbert-ir/colbertv2.0",
                                        tokenizer="colbert-ir/colbertv2.0",
                                        keep_retrieval_score=True,
                                    )
        postprocessor = SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
        #self.vector_query_engine = self.vector_index.as_query_engine(llm=self.agent_llm, similarity_top_k=10, node_postprocessors=[postprocessor])
        self.vector_query_engine = self.vector_index.as_query_engine(llm=self.agent_llm, similarity_top_k=10, node_postprocessors=[])
         
        self.vector_query_engine_reranked = self.vector_index.as_query_engine(llm=self.agent_llm, similarity_top_k=10,
                                                                             node_postprocessors=[self.reranker, postprocessor])

        return 
    
    def build_index(self):        
        file_extractor={".pdf": PdfFileReader(self.pdf_images_path, page_chunks=self.page_chunks)}
        config = LanguageConfig(language="english", spacy_model="en_core_web_md")
        if self.number_of_docs != None:
            reader = SimpleDirectoryReader(input_dir=self.docs_store_path, num_files_limit=self.number_of_files, file_extractor=file_extractor)
        else:
            reader = SimpleDirectoryReader(input_dir=self.docs_store_path, file_extractor=file_extractor)
        documents = reader.load_data(show_progress=True)
        parser = MarkdownNodeParser()
        markdown_pages = parser.get_nodes_from_documents(documents)

        sentence_splitter = SentenceSplitter(
                chunk_size=3000,
                chunk_overlap=200,
            )
        
        def id_func(i: int, doc: BaseNode) -> str:
            return f"{doc.id_}_{str(uuid.uuid4())}"


        nltk.download('punkt_tab')
        spacy.cli.download("en_core_web_md")
        splitter = SemanticDoubleMergingSplitterNodeParser(
                language_config=config,
                initial_threshold=0.4,
                appending_threshold=0.6,
                merging_threshold=0.6,
                max_chunk_size=5000,
                merging_range=2,
                splitter=[sentence_splitter],
                id_func=id_func,
            )
        nodes = splitter.get_nodes_from_documents(markdown_pages)
        transformations=[   
                self.embed_model
            ]
        vector_index = VectorStoreIndex(nodes=nodes, transformations=transformations, show_progress=True)
        
        return vector_index
   
    def query_index(self, query: str, top_k: int = 5):
        return self.vector_query_engine.query(query)
    
    def query_rerank(self, query: str, top_k: int = 5):                
        return self.vector_query_engine_reranked.query(query)
    
    def query_agent(self, query: str, top_k: int = 5):
        return self.top_agent.query(query)
    
    def process_documets(self, docs_store_path):
        file_extractor={".pdf": PdfFileReader(self.pdf_images_path)}
        #TDOD: Implement this method to process the documents
        pass
    
    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "main_llm": self.main_llm,
            "agent_llm": self.agent_llm,
            "top_agent": self.top_agent,
            "doc_agents": self.agents,
            "vector_index": self.vector_index,
            "vector_query_engine": self.vector_query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        #return self.top_agent.query(*args, **kwargs)
        #return self.vector_query_engine.query(*args, **kwargs)
        #result = self.query_rerank(args[0], top_k=5)
        result = self.query_index(args[0], top_k=5)
        return result 