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

from llama_index.agent.introspective import IntrospectiveAgentWorker
from llama_index.agent.introspective import (
    ToolInteractiveReflectionAgentWorker,
)
from llama_index.agent.introspective import SelfReflectionAgentWorker
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

from llama_index.core.agent import (
    StructuredPlannerAgent,
    FunctionCallingAgentWorker,
    ReActAgentWorker,
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
nltk.download('punkt_tab')
spacy.cli.download("en_core_web_md")

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
        number_of_docs: int = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        Settings.llm = main_llm
        Settings.embed_model = embeding_llm
        # Build agents dictionary
        self.agents = {}
        self.pdf_images_path = pdf_images_path
        self.verbose = verbose
        # this is for the baseline
        file_extractor={".pdf": PdfFileReader(self.pdf_images_path)}
        
        if os.path.exists(storage_dir):
            self.vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=storage_dir)
            )
        else:
            config = LanguageConfig(language="english", spacy_model="en_core_web_md")
            if number_of_docs!=None:
                reader = SimpleDirectoryReader(input_dir=docs_store_path, num_files_limit=number_of_docs, file_extractor=file_extractor)
            else:
                reader = SimpleDirectoryReader(input_dir=docs_store_path, file_extractor=file_extractor).load_data(show_progress=True)
            documents = reader.load_data(show_progress=True)
            parser = MarkdownNodeParser()
            markdown_pages = parser.get_nodes_from_documents(documents)

            sentence_splitter = SentenceSplitter(
                chunk_size=2000,
                chunk_overlap=200,
            )

            splitter = SemanticDoubleMergingSplitterNodeParser(
                language_config=config,
                initial_threshold=0.2,
                appending_threshold=0.4,
                merging_threshold=0.4,
                max_chunk_size=3000,
                merging_range=2,
                splitter=[sentence_splitter]
            )
            nodes = splitter.get_nodes_from_documents(markdown_pages)
            transformations=[   
                embeding_llm
            ]
            self.vector_index = VectorStoreIndex.from_documents(nodes, transformations=transformations, 
                                                       show_progress=True)
            # save the initial index
            self.vector_index.storage_context.persist(persist_dir=storage_dir)
            
        query_engine_tools = [
                QueryEngineTool (
                    query_engine=self.vector_index.as_query_engine(llm=agent_llm, similarity_top_k=5),
                    metadata=ToolMetadata(
                        name="vector_tool",
                        description="Useful to answer question based on relevant documents. " \
                                "Please mention the document name in the answer and provide relevant citations.",
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
        self.top_agent = self.main_agent.as_agent(chat_history=chat_history, verbose=verbose)

    def process_documets(self, docs_store_path):
        file_extractor={".pdf": PdfFileReader(self.pdf_images_path)}
        
        return reader

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "top_agent": self.top_agent,
            "doc_agents": self.agents,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.top_agent.query(*args, **kwargs)
        #return self.vector_index.as_query_engine().query(*args, **kwargs)