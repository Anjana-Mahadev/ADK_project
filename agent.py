# agent.py

import requests
from bs4 import BeautifulSoup
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
import ollama
from bytez import Bytez

import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()


# =================================================
# 1. DETERMINISTIC FALLBACK TOOLS
# =================================================

def explain_gcp_service(service_name: str) -> str:
    explanations = {
        "compute engine": "Compute Engine provides virtual machines (VMs) on Google Cloud.",
        "cloud storage": "Cloud Storage is an object storage service for unstructured data.",
        "bigquery": "BigQuery is a serverless data warehouse for analytics.",
        "gke": "Google Kubernetes Engine (GKE) manages Kubernetes clusters."
    }
    return explanations.get(
        service_name.lower(),
        "No explanation available for this service."
    )


def recommend_gcp_service(use_case: str) -> str:
    use_case = use_case.lower()
    if "vm" in use_case or "server" in use_case:
        return "Recommended service: Compute Engine"
    if "storage" in use_case or "file" in use_case:
        return "Recommended service: Cloud Storage"
    if "analytics" in use_case:
        return "Recommended service: BigQuery"
    if "container" in use_case or "kubernetes" in use_case:
        return "Recommended service: Google Kubernetes Engine (GKE)"
    return "Unable to determine the best GCP service."


def estimate_gcp_cost(service: str, hours: int) -> str:
    pricing = {
        "compute engine": 0.05,
        "gke": 0.10,
        "bigquery": 0.02
    }
    service = service.lower()
    if service not in pricing:
        return "Cost estimation not available."
    return f"Estimated cost for {service} for {hours} hours is ${pricing[service] * hours:.2f}"


# =================================================
# 2. SCRAPE MULTIPLE GCP PAGES (LAZY)
# =================================================

GCP_DOC_URLS = [
    "https://cloud.google.com/compute/docs/overview",
    "https://cloud.google.com/storage/docs/overview",
    "https://cloud.google.com/bigquery/docs/overview",
    "https://cloud.google.com/kubernetes-engine/docs/overview"
]

def scrape_gcp_docs() -> str:
    texts = []
    for url in GCP_DOC_URLS:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            main = soup.find("main")
            if main:
                text = "\n".join(
                    line.strip() for line in main.get_text().splitlines() if line.strip()
                )
                texts.append(text)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return "\n\n".join(texts)


# =================================================
# 3. VECTOR STORE (RAG) — LAZY INIT
# =================================================

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        raw_text = scrape_gcp_docs()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents = splitter.create_documents([raw_text])
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _vectorstore = FAISS.from_documents(documents, embeddings)
    return _vectorstore


# =================================================
# 4. RETRIEVER TOOL
# =================================================

def retrieve_gcp_docs(query: str) -> str:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant documentation found."
    return "\n\n".join(doc.page_content for doc in docs)


# =================================================
# 5. RAG-FIRST AGENT
# =================================================


root_agent = LlmAgent(
    name="my_first_agent",
    model=Gemini(model='gemini-3-flash-preview'),
    description="RAG-first GCP agent with deterministic tool fallbacks.",
    instruction="""
You are a Google Cloud Platform (GCP) assistant.

FOLLOW THESE RULES STRICTLY:

1. For ANY GCP-related question, FIRST call `retrieve_gcp_docs`.
2. If relevant documentation is retrieved:
   - Answer ONLY using the retrieved documentation.
3. If documentation retrieval fails or is insufficient:
   - Use the appropriate fallback tool:
     • explain_gcp_service
     • recommend_gcp_service
     • estimate_gcp_cost
4. Prefer tools over guessing.
5. If neither documentation nor tools provide an answer, reply:
   "I could not find this information in the official GCP documentation or available tools."

NEVER hallucinate.
Be factual, concise, and grounded.
""",
    tools=[
        retrieve_gcp_docs,
        explain_gcp_service,
        recommend_gcp_service,
        estimate_gcp_cost,
    ],
)
