{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "72c7a6f2-7c93-454d-836e-fa2c6445971d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: langchain-huggingface in C:\\Users\\hp\\rag_env\\Lib\\site-packages (1.2.0)\n",
      "Requirement already satisfied: huggingface-hub<1.0.0,>=0.33.4 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-huggingface) (0.36.2)\n",
      "Requirement already satisfied: langchain-core<2.0.0,>=1.2.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-huggingface) (1.2.9)\n",
      "Requirement already satisfied: tokenizers<1.0.0,>=0.19.1 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-huggingface) (0.22.2)\n",
      "Requirement already satisfied: filelock in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (3.20.3)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (2026.2.0)\n",
      "Requirement already satisfied: packaging>=20.9 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (26.0)\n",
      "Requirement already satisfied: pyyaml>=5.1 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (6.0.3)\n",
      "Requirement already satisfied: requests in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (2.32.5)\n",
      "Requirement already satisfied: tqdm>=4.42.1 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (4.67.3)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (4.15.0)\n",
      "Requirement already satisfied: jsonpatch<2.0.0,>=1.33.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (1.33)\n",
      "Requirement already satisfied: langsmith<1.0.0,>=0.3.45 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (0.6.9)\n",
      "Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (2.12.5)\n",
      "Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (9.1.4)\n",
      "Requirement already satisfied: uuid-utils<1.0,>=0.12.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (0.14.0)\n",
      "Requirement already satisfied: jsonpointer>=1.9 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from jsonpatch<2.0.0,>=1.33.0->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (3.0.0)\n",
      "Requirement already satisfied: httpx<1,>=0.23.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (0.28.1)\n",
      "Requirement already satisfied: orjson>=3.9.14 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (3.11.7)\n",
      "Requirement already satisfied: requests-toolbelt>=1.0.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (1.0.0)\n",
      "Requirement already satisfied: xxhash>=3.0.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (3.6.0)\n",
      "Requirement already satisfied: zstandard>=0.23.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (0.25.0)\n",
      "Requirement already satisfied: anyio in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (4.12.1)\n",
      "Requirement already satisfied: certifi in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (2026.1.4)\n",
      "Requirement already satisfied: httpcore==1.* in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (1.0.9)\n",
      "Requirement already satisfied: idna in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (3.11)\n",
      "Requirement already satisfied: h11>=0.16 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (0.16.0)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from pydantic<3.0.0,>=2.7.4->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (0.7.0)\n",
      "Requirement already satisfied: pydantic-core==2.41.5 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from pydantic<3.0.0,>=2.7.4->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (2.41.5)\n",
      "Requirement already satisfied: typing-inspection>=0.4.2 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from pydantic<3.0.0,>=2.7.4->langchain-core<2.0.0,>=1.2.0->langchain-huggingface) (0.4.2)\n",
      "Requirement already satisfied: charset_normalizer<4,>=2 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from requests->huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (3.4.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from requests->huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (2.6.3)\n",
      "Requirement already satisfied: colorama in C:\\Users\\hp\\rag_env\\Lib\\site-packages (from tqdm>=4.42.1->huggingface-hub<1.0.0,>=0.33.4->langchain-huggingface) (0.4.6)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Ignoring invalid distribution ~uggingface-hub (C:\\Users\\hp\\rag_env\\Lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution ~uggingface-hub (C:\\Users\\hp\\rag_env\\Lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution ~uggingface-hub (C:\\Users\\hp\\rag_env\\Lib\\site-packages)\n"
     ]
    }
   ],
   "source": [
    "pip install -U langchain-huggingface\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "bb620bc2-075b-4fed-96a2-ea7638353377",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "from langchain_core.runnables import RunnablePassthrough\n",
    "from langchain_core.output_parsers import StrOutputParser\n",
    "\n",
    "prompt = ChatPromptTemplate.from_template(\"\"\"\n",
    "You are a helpful assistant.\n",
    "Use the context and chat history to answer.\n",
    "\n",
    "Chat History:\n",
    "{chat_history}\n",
    "\n",
    "Context:\n",
    "{context}\n",
    "\n",
    "Question:\n",
    "{question}\n",
    "\"\"\")\n",
    "\n",
    "rag_chain = (\n",
    "    {\n",
    "        \"context\": retriever | (lambda docs: \"\\n\\n\".join(d.page_content for d in docs)),\n",
    "        \"question\": RunnablePassthrough(),\n",
    "        \"chat_history\": lambda _: memory.buffer\n",
    "    }\n",
    "    | prompt\n",
    "    | llm\n",
    "    | StrOutputParser()\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a3336d43-639d-4a04-ad56-952d1b10f770",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\hp\\rag_env\\Lib\\site-packages\\tqdm\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "from langchain_community.document_loaders import TextLoader\n",
    "from langchain_huggingface import HuggingFaceEmbeddings\n",
    "from langchain_community.vectorstores import FAISS\n",
    "from langchain_ollama import OllamaLLM\n",
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "from langchain_core.output_parsers import StrOutputParser\n",
    "from langchain_core.runnables import RunnablePassthrough\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0c650938-68c0-445d-97d2-aa8ebe95b65b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load your local document\n",
    "loader = TextLoader(r\"AI_Full_Course_Overview.txt\")\n",
    "docs = loader.load()\n",
    "\n",
    "# Split text into smaller chunks\n",
    "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\n",
    "splits = text_splitter.split_documents(docs)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8c0fb86c-1e95-4bb6-9585-86e890dceea9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1️⃣ Create embeddings FIRST\n",
    "from langchain_ollama import OllamaEmbeddings\n",
    "\n",
    "embeddings = OllamaEmbeddings(model=\"nomic-embed-text\")\n",
    "\n",
    "# 2️⃣ Create vector store\n",
    "from langchain_community.vectorstores import FAISS\n",
    "\n",
    "vectorstore = FAISS.from_documents(splits, embedding=embeddings)\n",
    "\n",
    "# 3️⃣ Create retriever\n",
    "retriever = vectorstore.as_retriever()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "796bb275-3b5d-4e13-bc75-890ae3bcec17",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['ChatOllama',\n",
       " 'OllamaEmbeddings',\n",
       " 'OllamaLLM',\n",
       " 'PackageNotFoundError',\n",
       " '__all__',\n",
       " '__builtins__',\n",
       " '__cached__',\n",
       " '__doc__',\n",
       " '__file__',\n",
       " '__loader__',\n",
       " '__name__',\n",
       " '__package__',\n",
       " '__path__',\n",
       " '__spec__',\n",
       " '__version__',\n",
       " '_compat',\n",
       " '_raise_package_not_found_error',\n",
       " '_utils',\n",
       " 'chat_models',\n",
       " 'embeddings',\n",
       " 'llms']"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import langchain_ollama\n",
    "dir(langchain_ollama)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ba814d5d-58d0-4b42-9001-2a0cd0f6f76c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_ollama import ChatOllama\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7f028f4f-2eb1-4c73-ae3b-62a4ff4a9a25",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_ollama import ChatOllama\n",
    "\n",
    "llm = ChatOllama(\n",
    "    model=\"gemma3:270m\",\n",
    "    temperature=0.3\n",
    ")\n",
    "\n",
    "# embeddings = OllamaEmbeddings(model=\"gemma3:latest\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "ee736cd5-a9ea-41de-bc7d-bbdb13670cde",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "ff9b7cf3-6d74-49f7-a32f-ab388cd204ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = ChatPromptTemplate.from_template(\"\"\"\n",
    "Use the following context to answer the question:\n",
    "{context}\n",
    "\n",
    "Question: {question}\n",
    "\"\"\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f0e022c5-422e-438a-b2f1-a6ba43e08445",
   "metadata": {},
   "outputs": [],
   "source": [
    "rag_chain = (\n",
    "    {\"context\": retriever | (lambda docs: \"\\n\\n\".join([d.page_content for d in docs])),\n",
    "     \"question\": RunnablePassthrough()}\n",
    "    \n",
    "    | prompt\n",
    "    | llm\n",
    "    | StrOutputParser()\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d5272c47-e691-40bd-913c-046614bd927c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Machine Learning and Deep Learning are distinct but related fields. Machine Learning focuses on developing algorithms that learn from data without explicit programming. Deep Learning, on the other hand, is a subfield of Machine Learning that uses artificial neural networks with multiple layers to learn complex patterns and representations from data.\n"
     ]
    }
   ],
   "source": [
    "query = \"What are the key differences between Machine Learning and Deep Learning?\"\n",
    "response = rag_chain.invoke(query)\n",
    "print(response)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "0484de49-0198-4e08-8881-b36af95d7e75",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Machine Learning section covers the following topics:\n",
      "\n",
      "*   **Programming:** Python, R\n",
      "*   **Machine Learning (ML):** Understanding supervised, unsupervised, and reinforcement learning, algorithms like Linear Regression, Decision Trees, Random Forest, SVM, and KNN, and using scikit-learn for model training, testing, and evaluation.\n",
      "*   **Large Language Models (LLMs):** Understanding architecture (transformers, attention mechanism), building and running LLMs locally using Ollama, using LangChain to connect models with external data, and using LangChain to connect models with external APIs and vector databases.\n",
      "*   **Computer Vision:** OpenCV for image processing, filtering, and transformation, Face Detection and Object Detection using SSD, YOLO, and Haar cascades, and Fire Detection and anomaly detection using custom-trained CNN models.\n",
      "*   **Data Handling and ETL:** ETL (Extract, Transform, Load) for moving data from raw sources to usable formats, Data Wrangling, Manipulation, and Transformation using Pandas and NumPy, Data Preprocessing: cleaning, normalizing, and preparing data for ML/DL.\n",
      "*   **Data Visualization & Analysis:** Creating visual insights with Matplotlib, Seaborn, and Plotly, Performing exploratory data analysis (EDA) to understand patterns and trends, and creating visual insights with Matplotlib, Seaborn, and Plotly.\n"
     ]
    }
   ],
   "source": [
    "query = \"Explain what topics are covered in the Machine Learning section.\"\n",
    "response = rag_chain.invoke(query)\n",
    "print(response)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "90610559-ca44-411b-8c77-6ae3f86f044a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-02-09 20:51:23.756 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.756 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.756 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.756 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.760 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.760 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.761 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.761 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.763 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.763 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.764 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.765 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.765 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.767 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.768 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.772 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.773 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.774 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-09 20:51:23.775 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import tempfile\n",
    "import os\n",
    "\n",
    "from langchain_ollama import ChatOllama, OllamaEmbeddings\n",
    "from langchain_community.document_loaders import PyPDFLoader, TextLoader\n",
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "from langchain_community.vectorstores import FAISS\n",
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "from langchain_core.runnables import RunnablePassthrough\n",
    "from langchain_core.output_parsers import StrOutputParser\n",
    "\n",
    "# -------------------------------\n",
    "# Models\n",
    "# -------------------------------\n",
    "llm = ChatOllama(\n",
    "    model=\"llama3:8b-instruct-q4_K_M\",\n",
    "    temperature=0.2\n",
    ")\n",
    "\n",
    "embeddings = OllamaEmbeddings(\n",
    "    model=\"nomic-embed-text\"\n",
    ")\n",
    "\n",
    "# -------------------------------\n",
    "# Streamlit UI\n",
    "# -------------------------------\n",
    "st.set_page_config(page_title=\"Local RAG Chatbot\", layout=\"centered\")\n",
    "st.title(\"🤖 Local RAG Chatbot\")\n",
    "st.caption(\"Upload PDF or TXT files and ask questions from them (100% local)\")\n",
    "\n",
    "# -------------------------------\n",
    "# Session state\n",
    "# -------------------------------\n",
    "if \"vectorstore\" not in st.session_state:\n",
    "    st.session_state.vectorstore = None\n",
    "\n",
    "if \"history\" not in st.session_state:\n",
    "    st.session_state.history = []\n",
    "\n",
    "# -------------------------------\n",
    "# File upload\n",
    "# -------------------------------\n",
    "uploaded_files = st.file_uploader(\n",
    "    \"Upload PDF or Text files\",\n",
    "    type=[\"pdf\", \"txt\"],\n",
    "    accept_multiple_files=True\n",
    ")\n",
    "\n",
    "if uploaded_files and st.button(\"📥 Process Files\"):\n",
    "    documents = []\n",
    "\n",
    "    with st.spinner(\"Processing and embedding files...\"):\n",
    "        for file in uploaded_files:\n",
    "            suffix = file.name.split(\".\")[-1]\n",
    "\n",
    "            with tempfile.NamedTemporaryFile(delete=False, suffix=f\".{suffix}\") as tmp:\n",
    "                tmp.write(file.read())\n",
    "                tmp_path = tmp.name\n",
    "\n",
    "            if suffix == \"pdf\":\n",
    "                loader = PyPDFLoader(tmp_path)\n",
    "            else:\n",
    "                loader = TextLoader(tmp_path, encoding=\"utf-8\")\n",
    "\n",
    "            documents.extend(loader.load())\n",
    "            os.remove(tmp_path)\n",
    "\n",
    "        splitter = RecursiveCharacterTextSplitter(\n",
    "            chunk_size=800,\n",
    "            chunk_overlap=150\n",
    "        )\n",
    "\n",
    "        splits = splitter.split_documents(documents)\n",
    "\n",
    "        st.session_state.vectorstore = FAISS.from_documents(\n",
    "            splits,\n",
    "            embedding=embeddings\n",
    "        )\n",
    "\n",
    "    st.success(\"✅ Files indexed successfully\")\n",
    "\n",
    "# -------------------------------\n",
    "# Chat section\n",
    "# -------------------------------\n",
    "if st.session_state.vectorstore:\n",
    "    query = st.text_input(\"Ask a question\")\n",
    "\n",
    "    if st.button(\"Send\") and query:\n",
    "        retriever = st.session_state.vectorstore.as_retriever()\n",
    "\n",
    "        prompt = ChatPromptTemplate.from_template(\"\"\"\n",
    "You are a helpful assistant.\n",
    "Answer ONLY using the provided context.\n",
    "\n",
    "Chat History:\n",
    "{chat_history}\n",
    "\n",
    "Context:\n",
    "{context}\n",
    "\n",
    "Question:\n",
    "{question}\n",
    "\"\"\")\n",
    "\n",
    "        rag_chain = (\n",
    "            {\n",
    "                \"context\": retriever | (lambda docs: \"\\n\\n\".join(d.page_content for d in docs)),\n",
    "                \"question\": RunnablePassthrough(),\n",
    "                \"chat_history\": lambda _: \"\\n\".join(\n",
    "                    [f\"{r}: {m}\" for r, m in st.session_state.history]\n",
    "                )\n",
    "            }\n",
    "            | prompt\n",
    "            | llm\n",
    "            | StrOutputParser()\n",
    "        )\n",
    "\n",
    "        answer = rag_chain.invoke(query)\n",
    "\n",
    "        st.session_state.history.append((\"You\", query))\n",
    "        st.session_state.history.append((\"Bot\", answer))\n",
    "\n",
    "# -------------------------------\n",
    "# Display chat\n",
    "# -------------------------------\n",
    "if st.session_state.history:\n",
    "    st.markdown(\"### 💬 Conversation\")\n",
    "    for role, msg in st.session_state.history:\n",
    "        st.markdown(f\"**{role}:** {msg}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": none,
   "id": "b96bb811-c314-4205-bde8-18fcfb5f32de",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (rag_env)",
   "language": "python",
   "name": "rag_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
