from assistant.multi_document_agent.personal_assitant import MultiDocumentAssistantAgentsPack
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

hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = ollama
Settings.embed_model = hf_embed_model

agent = MultiDocumentAssistantAgentsPack(main_llm=ollama, agent_llm=ollama_agent, 
                                         embeding_llm=hf_embed_model, 
                                         docs_store_path=docs_store_path, 
                                         pdf_images_path=pdf_images_path,
                                         storage_dir=storage_dir, verbose=True)
#response = agent.run("Give me the location of Homewood Suites by Hilton. Do not guess only use context.")
#response = agent.run("Give me the address of Homewood Suites by Hilton where I was staying and you have an invoce from this hotel.")
#response = agent.run("How many times Oleg Grynets stayed in hotels?")
response = agent.run("What genAI-enabled or -assisted experiences benefits will be delivered to enterprises according to Forester?")
print(response)


