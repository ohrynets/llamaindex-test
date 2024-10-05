from assistant.multi_document_agent.personal_assitant import MultiDocumentAssistantAgentsPack
from assistant.multi_document_agent.personal_assitant import PdfFileReader
from dotenv import load_dotenv
load_dotenv()
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

ollama_base_url=os.environ['OLLAMA_BASE_URL']
pdf_images_path=os.environ['PDF_IMAGES_STORE_PATH']
docs_store_path=os.environ['DOCS_STORE_PATH']
langfuse_url = os.environ['LANGFUSE_URL']
langfuse_public_key = os.environ['LANGFUSE_PUBLIC_KEY']
langfuse_secret_key = os.environ['LANGFUSE_SECRET_KEY']
storage_dir = os.environ['STORAGE_DIR']

#Configure 
session_id = str(uuid.uuid4())
print(f"Session_id: {session_id}")
langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host=langfuse_url,
    session_id=session_id,
    debug=False,
)
#Settings.callback_manager = CallbackManager([langfuse_callback_handler])
ollama = Ollama(model="llama3.1:8b", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)
ollama_agent = Ollama(model="llama3.2:latest", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)
embed_model = OllamaEmbedding(model_name="bge-m3", request_timeout=120.0, base_url=ollama_base_url)


hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = ollama
Settings.embed_model = hf_embed_model
print(f"Folder with documents:{docs_store_path}")
file_extractor={".pdf": PdfFileReader(pdf_images_path)}
#reader = SimpleDirectoryReader(input_dir=docs_store_path, num_files_limit=10, file_extractor=file_extractor)
reader = SimpleDirectoryReader(input_dir=docs_store_path, num_files_limit=1)            
documents = reader.load_data(show_progress=True)

QUESTION_GENERATE_QUERY="""You are a Teacher/Professor. \
Your task is to setup on questions for an upcoming quiz/examination. \
The questions should be diverse in nature across the document. \
Restrict the questions to the context information provided. \
Don't use new line separator."""
data_generator = RagDatasetGenerator.from_documents(documents, llm=ollama,
                                                    question_gen_query=QUESTION_GENERATE_QUERY, 
                                                    num_questions_per_chunk=1,
                                                    show_progress=True)
print(data_generator.question_gen_query)
eval_questions = data_generator.generate_questions_from_nodes()
eval_questions_df = eval_questions.to_pandas()
duckdb.sql("CREATE TABLE eval_data AS SELECT * FROM eval_questions_df")
duckdb.sql("INSERT INTO eval_data SELECT * FROM eval_questions_df")
duckdb.sql("COPY (SELECT * FROM eval_data) TO 'eval_questions.parquet' (FORMAT PARQUET)")
print(eval_questions)