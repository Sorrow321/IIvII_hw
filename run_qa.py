import pandas as pd
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings, SentenceTransformerEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema import Document


input_docs = ['documents/doc1.pdf', 'documents/doc2.pdf']

chunks = []
loaders = [UnstructuredPDFLoader(path) for path in input_docs]
for loader in loaders:
    pages = loader.load()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    chunks_i = text_splitter.transform_documents(pages)
    chunks += chunks_i


df = pd.read_csv('documents/cards.csv')
result_docs = []
for service, cond, val in df.values:
    line = f'Сервис: {service}. Условие: {cond}. Тариф: {val}'
    result_docs.append(Document(page_content=line, metadata={'source': 'cards.csv'}))

chunks += result_docs



store = LocalFileStore("./cache/")
embedding_model = SentenceTransformerEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L12-v2')
embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding_model,
    store,
    namespace='llama_cpp'
)
faiss_index = FAISS.from_documents(chunks, embedder)

template = """
Ваша роль - консультант клиентов банка "Тинькофф Банк".
Ваша задача - отвечать на вопросы клиентов, руководствуясь предложенными фрагментами документов.
Никогда не выдумывайте ответ, если его нет в предложенных фрагментах документов.
Используйте от 3 до 5 предложений. Ваш ответ должен быть краткий и содержательный.
Всегда говори "Спасибо за Ваш вопрос!" в начале ответа.
Используйте только русские слова в своей ответе.
Не придумывайте ответ, если его нет в предложенном фрагменте документов.
Предложенных фрагменты документов: {context}
Вопрос клиента: {question}
Ответ на русском языке:"""

rag_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = LlamaCpp(
    model_path="weights/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.0,
    max_tokens=5000,
    top_p=1,
    n_ctx=2048,
    n_batch=8,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True
)

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_index.as_retriever(search_kwargs={'k': 10}),
    callbacks=[StdOutCallbackHandler()],
    return_source_documents=True,
    chain_type_kwargs={"prompt": rag_prompt}
)


def get_qa(question: str) -> str:
    return qa_with_sources_chain.invoke(question)


def run_infinite():
    while True:
        inp = input()
        ans = get_qa(inp)['result']
        print(f'=========INPUT: {inp} ===========\n\n===========ANSWER: ===========\n{ans}')

# run_infinite()