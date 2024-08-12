from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
import os

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)
ollama_base_url=os.environ['OLLAMA_BASE_URL']
print(f"OLLAMA_BASE_URL:{ollama_base_url}")
#llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
llm = Ollama(model="llama3.1:8b", request_timeout=120.0, base_url=ollama_base_url)
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 20+(2*4)? Use a tool to calculate every step.")

print(response)