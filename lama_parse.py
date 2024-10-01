from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
import pymupdf4llm
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
import nest_asyncio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)
from llama_index.llms.vllm import Vllm
from llama_index.core.llms import ChatMessage
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
                         EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)

# set defalut text and image embedding functions
embedding_function = OpenCLIPEmbeddingFunction()
nest_asyncio.apply()
mrkdown_parser = MarkdownNodeParser()
class PdfFileReader(BaseReader):

    def load_data(self, file, extra_info={}):
        md_content = pymupdf4llm.to_markdown(file, write_images=True, page_chunks=True, image_path=".pdf_images/")

        # load_data returns a list of Document objects
        nodes = []
        for d in md_content:
            doc_id = f"{d['metadata']['title']}:{d['metadata']['page']}"
            doc = Document(text=d['text'], id_=doc_id, extra_info={**extra_info, **d['metadata']})
            nodes.append(doc)
        return nodes

class ImageExtractor(TransformComponent):

    def __call__(self, nodes, **kwargs):
        pattern = r'!\[.*?\]\((.*?)\)'
        #loaderTabular = ImageTabularChartReader(keep_image=True)
        #loader = ImageVisionLLMReader(keep_image=True)
        loader = ImageReader(keep_image=True, text_type="plain_text")
        images = []
        for node in nodes:
            images.append(node)
            image_pathes = re.findall(pattern=pattern, string=node.text)
            for image_path in image_pathes:
                print(image_path)
                #imgs = loaderTabular.load_data(image_path)
                #images.extend(imgs)
                imgs = loader.load_data(image_path)
                for img in imgs:
                    if img.text is not None and len(img.text) > 0:
                        images.append(img)
        return images
     
    
# settings
model_8B_url="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
model_70B_url="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q3_K_S.gguf"                                                                                     
llama_cpp = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_8B_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None, 
    temperature=0.5,
    max_new_tokens=1024,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    #context_window=3900,
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 33},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
    verbose=True,
)
#print(f"OLLAMA_BASE_URL:{ollama_base_url}")
#Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
ollama = Ollama(model="llama3.1:8b", request_timeout=120.0, base_url=ollama_base_url)
model_8B_url="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
model_70B_url="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q3_K_S.gguf"                                                                                     
llama_cpp = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_8B_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None, 
    temperature=0.5,
    max_new_tokens=1024,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    #context_window=3900,
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 33},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
    verbose=False,
)

ollama_embeding = OllamaEmbedding(model_name="nomic-embed-text", request_timeout=120.0, base_url=ollama_base_url, ollama_additional_kwargs={"mirostat": 0},)
hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cuda")
Settings.llm = llama_cpp
Settings.embed_model = hf_embed_model
#Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
# set up parser

# use SimpleDirectoryReader to parse our file
file_extractor={".pdf": PdfFileReader()}
# rag pipeline

# load it again to confirm it worked
from llama_index.core import StorageContext, load_index_from_storage

image_loader = ImageLoader()

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

#if directory exists, load the index from storage
if os.path.exists(storage_dir):
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=storage_dir)
    )
else:
    reader = SimpleDirectoryReader(input_dir=docs_store_path, file_extractor=file_extractor).load_data(show_progress=True)
    transformations=[   
        ImageExtractor(),
    ]
    index = VectorStoreIndex.from_documents(reader, transformations=transformations, show_progress=True)
    print("ref_docs ingested: ", len(index.ref_doc_info))
    print("number of input documents: ", len(reader))    
    # save the initial index
    index.storage_context.persist(persist_dir=storage_dir)

# Check if embedding model is loaded correctly
    
retriever = index.as_retriever(verbose=True)
response = retriever.retrieve("Give me the address of Homewood Suites by Hilton. Do not guess only use context.")
for res in response:
    print(res)


def query(query_engine, query: BaseQueryEngine):
    response = query_engine.query(query)
    print(response)

query_engine = index.as_query_engine(streaming=True, response_mode="refine", verbose=True)
query(query_engine,
      '''
        Give me all the Tongue Twisters from all documents in the context.
        Don't use previous knowledge, only information from the context.
        Format the response in Markdown format.
      ''')
#.print_response_stream()
print("")
query(query_engine,
      '''
        Give me tyhe list of all documents with Tongue Twisters.
        Don't use previous knowledge, only information from the context.
        Format the response in Markdown format.
      ''')