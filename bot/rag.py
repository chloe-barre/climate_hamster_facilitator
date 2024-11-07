import os
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
import pinecone

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone index
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "factcheck-local"
pinecone_index = pinecone.Index(index_name)

model_name = "BAAI/bge-m3"
encode_kwargs = {'normalize_embeddings': True,"show_progress_bar":False} # set True to compute cosine similarity
print("Loading embeddings model: ", model_name)
embeddings_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction=query_instruction,
)

# Create a Retrieval QA chain using Pinecone as the vector store
retriever = Pinecone(
    pinecone_index=pinecone_index,
    embedding_function=embeddings_function,
)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)

async def generate_factcheck(query: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain.invoke({"input": query})
