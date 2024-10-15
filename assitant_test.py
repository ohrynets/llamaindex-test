from assistant.multi_document_agent.personal_assitant import MultiDocumentAssistantAgentsPack
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
import duckdb
from llama_index.llms.openai import OpenAI

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

import pandas as pd

from datasets import Dataset 
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.integrations.llama_index import evaluate
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings

# util function to init Ragas Metrics
def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

ollama_base_url=os.environ['OLLAMA_BASE_URL']
pdf_images_path=os.environ['PDF_IMAGES_STORE_PATH']
docs_store_path=os.environ['DOCS_STORE_PATH']
langfuse_url = os.environ['LANGFUSE_URL']
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
ollama = Ollama(model="llama3.1:8b", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)
qwen = Ollama(model="qwen2.5:32b", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)

ollama_agent = Ollama(model="llama3.2:latest", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)
embed_model = OllamaEmbedding(model_name="bge-m3", request_timeout=120.0, base_url=ollama_base_url)
critic_llm = OpenAI(model="gpt-4-turbo")

import nest_asyncio

nest_asyncio.apply()

hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = ollama
Settings.embed_model = embed_model

agent = MultiDocumentAssistantAgentsPack(main_llm=ollama_agent, agent_llm=ollama_agent, 
                                         embeding_llm=embed_model, 
                                         docs_store_path=docs_store_path, 
                                         pdf_images_path=pdf_images_path,
                                         storage_dir=storage_dir, verbose=True, number_of_docs=1)

eval_questions = duckdb.query("SELECT question, ground_truth FROM read_parquet('eval_questions.parquet') LIMIT 100")
queries = []
contexts = []
answers = []
ground_truths = []
 
for row in eval_questions.fetchall():
    question = row[0]
    ground_truth = row[1]
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    
    try:
        response = agent.run(question)
        print(f"Response: {response}")
        queries.append(question)
        contexts.append(ground_truth)
        answers.append(response.response)
        ground_truths.append(ground_truth)
    except Exception as e:
        response = None
        print(f"Error processing question '{question}': {e}")
    
    print("--------------------------------------------------")
    


data = {
    "question": queries,
    "contexts": contexts,
    "answer": answers,
    "ground_truth": ground_truths,
}

eval_dataset = Dataset.from_dict(data)

def evaluate_and_save(metrics, llm, agent, eval_questions):
    result = evaluate(
        query_engine=agent.get_modules()["vector_index"].as_query_engine(),
        metrics=metrics,
        dataset=eval_questions,
        llm=llm,
        embeddings=Settings.embed_model,
    )

    eval_result_df = result.to_pandas()
    duckdb.sql("CREATE TABLE eval_result AS SELECT * FROM eval_result_df")
    duckdb.sql("INSERT INTO eval_result SELECT * FROM eval_result_df")
    duckdb.sql("COPY (SELECT * FROM eval_result) TO 'eval_result.parquet' (FORMAT PARQUET)")
    return result

evalation_result = evaluate_and_save(metrics, qwen, agent, eval_dataset)
print(evalation_result)
#{'faithfulness': nan, 'answer_relevancy': 0.3304, 'context_precision': 1.0000, 'context_recall': 0.6019}
#{'faithfulness': nan, 'answer_relevancy': 0.4121, 'context_precision': 0.9211, 'context_recall': 0.7560}
#{'faithfulness': 0.7818, 'answer_relevancy': 0.6639, 'context_precision': 0.6395, 'context_recall': 0.7416}
#{'faithfulness': 1.0000, 'answer_relevancy': 0.3567, 'context_precision': 0.9250, 'context_recall': 0.7131}
#{'faithfulness': 0.6000, 'answer_relevancy': 0.4819, 'context_precision': 0.9250, 'context_recall': 0.8762}

    
#response = agent.run("Give me the location of Homewood Suites by Hilton. Do not guess only use context.")
#response = agent.run("Give me the address of Homewood Suites by Hilton where I was staying and you have an invoce from this hotel.")
#response = agent.run("What is included in data quality management to ensure that data is fit for its intended uses?")
#print(response)
#response = agent.run("What are the key phases involved in the AI Product Life Cycle?")
#response = agent.run("What action did not occur, instead of going to the store?")
#response = agent.run("What genAI-enabled or -assisted experiences benefits will be delivered to enterprises according to Forester?")
#print(response)


