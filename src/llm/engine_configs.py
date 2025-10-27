from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAI
from google.oauth2 import service_account
from google.cloud import aiplatform
from typing import Dict, Any
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

if GCP_CREDENTIALS and GCP_PROJECT and GCP_REGION:
    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
        credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
    )
    vertexai.init(project=GCP_PROJECT, location=GCP_REGION,
                  credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS))

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
"qwen2.5-coder-32b": {
        "constructor": ChatOpenAI,
        "params": {"model": "qwen2.5-coder-32b", "temperature": 0, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "****"}
    },
    "qwen2.5-coder-32b-instruct": {
        "constructor": ChatOpenAI,
        "params": {"model": "qwen2.5-coder-32b-instruct", "temperature": 0, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "*****"}
    },

    "Qwen2.5-Coder-7B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {"model": "Qwen2.5-Coder-7B-Instruct", "temperature": 0, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "*****"}
    },
    "deepseek-chat": {
        "constructor": ChatOpenAI,
        "params": {"model": "deepseek-chat", "temperature": 0, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "****"}
    },
    "deepseek-distill-qwen-14b": {
        "constructor": ChatOpenAI,
        "params": {"model": "deepseek-distill-qwen-14b", "temperature": 0, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "****"}
    },
    "qwen2.5-coder-7b": {
        "constructor": ChatOpenAI,
        "params": {"model": "qwen2.5-coder-7b", "temperature": 0, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "****"}
    },
    "qwen2.5-coder-14b": {
        "constructor": ChatOpenAI,
        "params": {"model": "qwen2.5-coder-14b", "temperature": 0.6, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "***"}
    },
    "qwen3-8b": {
        "constructor": ChatOpenAI,
        "params": {"model": "qwen3-8b", "temperature": 0.6, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "*****"}
    },
    "llama3.1-8b": {
        "constructor": ChatOpenAI,
        "params": {"model": "llama3.1-8b", "temperature": 0.6, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "****"}
    },
    "gemma3-4b": {
        "constructor": ChatOpenAI,
        "params": {"model": "gemma3-4b", "temperature": 0.6, "openai_api_base": 'http://localhost:8000',
                   "openai_api_key": "******"}
    },
    "gpt-4o": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o", "temperature": 0}
    },
    "gpt-4o-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o-mini", "temperature": 0}
    }

}
