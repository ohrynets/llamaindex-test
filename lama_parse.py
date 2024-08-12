from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_parse import LlamaParse

# settings
Settings.llm = OpenAI(model="gpt-3.5-turbo",temperature=0)

# rag pipeline
reader = SimpleDirectoryReader("/mnt/docs").load_data()
for idx, docs in enumerate(reader):
    print(docs.text)
index = VectorStoreIndex.from_documents(reader)
query_engine = index.as_query_engine()