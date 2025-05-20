from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import re
import os

# Загрузка текста (или напрямую вставь строку)
with open("Техническое обслуживание коробки переменных передач.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Разбивка по секциям ==Заголовок==
sections = re.findall(r"==(.+?)==\n(.*?)(?=\n==|$)", text, flags=re.DOTALL)

# Создание документов с метаданными
docs = []
for title, content in sections:
    content = content.strip()
    if content:
        docs.append(Document(
            page_content=content,
            metadata={"section": title.strip()}
        ))

# Эмбеддинги
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Создание FAISS индекса
db = FAISS.from_documents(docs, embedding)

# Сохраняем локально
db.save_local("faiss_gearbox_index")
