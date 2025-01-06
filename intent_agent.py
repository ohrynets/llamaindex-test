from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from assistant.multi_document_agent.personal_assitant import MultiDocumentAssistantAgentsPack
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)
import os
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.callbacks import CallbackManager
import uuid
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import CitationQueryEngine

ollama_base_url=os.environ['OLLAMA_BASE_URL']
pdf_images_path=os.environ['PDF_IMAGES_STORE_PATH']
docs_store_path=os.environ['DOCS_STORE_PATH']
langfuse_url = os.environ['LANGFUSE_BASE_URL']
langfuse_public_key = os.environ['LANGFUSE_PUBLIC_KEY']
langfuse_secret_key = os.environ['LANGFUSE_SECRET_KEY']
storage_dir = os.environ['STORAGE_DIR']
phoenix_url = os.environ['PHOENIX_URL']
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

from phoenix.otel import register

tracer_provider = register(
  project_name="llama-test",
  endpoint=phoenix_url,
  batch=True,
  verbose=True,
)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
Settings.callback_manager = CallbackManager([langfuse_callback_handler])
print(f"Ollama Url: {ollama_base_url}")
print(f"Langfuse Url: {langfuse_url}")
ollama_agent = Ollama(model="llama3.2:latest", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)
embed_model = OllamaEmbedding(model_name="bge-m3", request_timeout=120.0, base_url=ollama_base_url)


agent = MultiDocumentAssistantAgentsPack(main_llm=ollama_agent, agent_llm=ollama_agent, 
                                         embeding_llm=embed_model, 
                                         docs_store_path=docs_store_path, 
                                         pdf_images_path=pdf_images_path,
                                         storage_dir=storage_dir, verbose=True, number_of_files=1)   


async def main():
    result = agent.run("Give me the defenition of Data-Intensive Applications")
    print(result)
    print("========================================")
    result = agent.run("Could you help me explaining the weather patterns?")
    print(result)
    print("========================================")
    result = agent.run("What is the current weather in London?")
    print(result)
    print("========================================")

    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())