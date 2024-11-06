from assistant.multi_document_agent.personal_assitant import MultiDocumentAssistantAgentsPack
from assistant.multi_document_agent.personal_assitant import PdfFileReader
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)
import os
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.callbacks import CallbackManager
import uuid
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core import SimpleDirectoryReader
import duckdb
import ragas
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
import nltk
import spacy
from llama_index.llms.azure_openai import AzureOpenAI
from ragas.run_config import RunConfig
from llama_index.core.node_parser import MarkdownNodeParser
import duckdb
import pandas

from llama_index.readers.confluence import ConfluenceReader

azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT','https://ai-proxy.lab.epam.com')
azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_model=os.getenv('AZURE_OPENAI_MODEL')

ollama_base_url=os.environ['OLLAMA_BASE_URL']

confluence_api_token=os.getenv('CONFLUENCE_API_TOKEN')
confluence_api_url=os.getenv('CONFLUENCE_API_URL')


ollama = Ollama(model="llama3.1:8b", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)

token = {"access_token": confluence_api_token, "token_type": "Bearer"}
oauth2_dict = {"client_id": "Auto_EPG-ITAI_Bot@epam.com", "token": token}

base_url = confluence_api_url
space_key='EPGITAI'
cql = f'type="page" AND ancestor=2401961571 AND space="{space_key}"'
#cql = f'space="EPGITAI"'

reader = ConfluenceReader(base_url=base_url, oauth2=oauth2_dict)
#TODO: Add support for pagination and attachements
documents = reader.load_data(cql=cql, limit = 10, max_num_results=10, include_attachments=False)
#documents = reader.load_data(cql=cql, include_attachments=False)
#documents = reader.load_data(space_key=space_key, include_attachments=True, page_status="current")
count = 1
documents_dict = list()
while len(documents):
    print(f"Batch {count} of documents: {len(documents)}")
    for d in documents:
        documents_dict.append({'url': d.metadata['url'], 
                            'title': d.metadata['title'], 
                            'page_id': d.metadata['page_id'], 
                            'status': d.metadata['status'], 
                            'text': d.text})
        print(f"PageId: {d.metadata['page_id']} Title: {d.metadata['title']}")
        #print(f"URL: {d.metadata['url']}")
    #break
    cursor = reader.get_next_cursor()
    documents = reader.load_data(cql=cql, cursor=cursor, limit = 10, max_num_results=10, include_attachments=False)    
    count = count + 1


kb_pages_df = pandas.DataFrame.from_records(documents_dict)
kb_pages = 'kb_pages2.parquet'
duckdb.sql("CREATE TABLE kb_pages AS SELECT * FROM kb_pages_df")

# insert into the table "my_table" from the DataFrame "kb_pages_df"
#duckdb.sql("INSERT INTO kb_pages SELECT * FROM kb_pages_df")


duckdb.sql(f"COPY (SELECT * FROM kb_pages) TO '{kb_pages}' (FORMAT PARQUET)")