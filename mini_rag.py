import re
import json
from faiss import  read_index
import numpy as np


from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
import torch
from LLM import LLM
from langchain_core.language_models.llms import LLM as LangChainLLM
from typing import Any, Dict, List, Optional

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.load_local("faiss_gearbox_index", embeddings=embedding, allow_dangerous_deserialization=True)

# ================================================ Просто фаис поиск ================================================
query = "Как проверить сальник?"
results = faiss_index.similarity_search(query, k=3)
for r in results:
    print('Название секции:', r.metadata['section'], '\n')
    print('Содержимое секции:', r.page_content, '\n')
    print('-'*100)

# ================================================ ПОЛНЫЙ РАГ ================================================
# llm = LLM()
# llm_pipe = HuggingFacePipeline(pipeline=llm.pipeline)
# with open("prompt.txt", "r", encoding="utf-8") as f:
#     template = f.read() 

# prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm_pipe,
#     chain_type="stuff",  # или map_reduce, refine, etc.
#     retriever=faiss_index.as_retriever(search_kwargs={"k": 2}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt}
# )
# # === Запрос ===
# query = "Как проверить сальник?"
# result = qa_chain(query)

# # === Выводим результат ===
# print("Ответ:", result['result'])

# # Источник (по желанию)
# for doc in result['source_documents']:
    print('Название секции:', r.metadata['section'], '\n')
    print('Содержимое секции:', r.page_content, '\n')
    print('-'*100)