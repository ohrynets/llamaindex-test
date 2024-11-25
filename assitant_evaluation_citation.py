from assistant.multi_document_agent.personal_assitant_parquet_citation import CitationParquetDocumentAssistantAgentsPack
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
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from deepeval import assert_test
from deepeval.models import DeepEvalBaseLLM
from ragas.evaluation import Result
from ragas.llms import LlamaIndexLLMWrapper
from ragas.evaluation import evaluate as ragas_evaluate
from llama_index.core.base.llms.base import BaseLLM

import pandas as pd
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.llama_pack.base import BaseLlamaPack

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
openai_api_version=os.getenv('OPENAI_API_VERSION')
azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT','https://ai-proxy.lab.epam.com')
azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_model=os.getenv('AZURE_OPENAI_MODEL')
evaluation_ds = os.getenv('EVALUATION_DS')
evaluation_result = os.getenv('EVALUATION_RESULT')

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
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever
#LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
#Settings.callback_manager = CallbackManager([langfuse_callback_handler])
print(f"Ollama Url: {ollama_base_url}")
print(f"Langfuse Url: {langfuse_url}")
ollama = Ollama(model="llama3.1:8b", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)
ollama_agent = Ollama(model="llama3.2:latest", request_timeout=120.0, base_url=ollama_base_url, 
                is_function_calling_model=True)
embed_model = OllamaEmbedding(model_name="bge-m3", request_timeout=120.0, base_url=ollama_base_url)
#nvidia = NVIDIA(model="meta/llama-3.1-70b-instruct")

azurellm = AzureOpenAI(
    engine=azure_openai_model,
    model=azure_openai_model,
    api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=openai_api_version,
)

import nest_asyncio

nest_asyncio.apply()

Settings.llm = azurellm
Settings.embed_model = embed_model
print(f"evaluation data:{evaluation_ds}")
    
agent = CitationParquetDocumentAssistantAgentsPack(main_llm=azurellm, agent_llm=azurellm, 
                                         embeding_llm=embed_model, 
                                         docs_store_path=docs_store_path, 
                                         pdf_images_path=pdf_images_path,
                                         storage_dir=storage_dir, verbose=True, number_of_docs=1,
                                         similarity_cutoff=0.70)

eval_questions = duckdb.query(f"SELECT question, ground_truth, contexts FROM read_parquet('{evaluation_ds}') LIMIT 100")


def build_eval_dataset(agent: BaseLlamaPack, eval_questions):
    session_id = str(uuid.uuid4())
    agent_model = agent.get_modules()["main_llm"]
    metadata = agent_model.to_dict()
    queries = []
    contexts = []
    answers = []
    ground_truths = []
    answered = []
    
    for row in eval_questions.fetchall():
        question = row[0]
        ground_truth = row[1]
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        try:
            response = agent.run(question)
            print(f"==================================================")
            print(f"{response.response}")
            print(f"==================================================")
            source_id = 1
            for n in response.source_nodes:
                if source_id > 1:
                    print(f"--------------------------------------------------")
                print(f"Source {source_id}: {n.node.metadata["url"]}")                
                source_id += 1
            print(f"==================================================")
            answered.append(response.response is not None)
            queries.append(question)            
            context = [n.node.text for n in response.source_nodes]
            contexts.append(context)
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
        "answered": answered
    }
    eval_dataset = Dataset.from_dict(data)
    return eval_dataset

eval_dataset = build_eval_dataset(agent, eval_questions)

#{'faithfulness': 0.7896, 'answer_relevancy': nan, 'context_precision': 0.7302, 'context_recall': 0.8025}


