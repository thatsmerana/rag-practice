import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent

# Load environment variables from .env file if present
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key.")

# Iterate through all PDF files from all subdirectories in /Users/manojrana/Documents/Cursor Coworker /Qwary
pdf_directory = "/Users/manojrana/Documents/Cursor Coworker /Qwary"
documents = []
for root, dirs, files in os.walk(pdf_directory):
    for file in files:
        if file.endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            print(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)

# Use Tiktoken encoder for splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)

# Print the split documents to verify the output
print(f"Total chunks: {len(split_docs)}")

# Create embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Store the documents in the Chroma vector database
vector_store = Chroma(
    collection_name="sales_calls_chunks",
    persist_directory="/Users/manojrana/Documents/Cursor Coworker /Qwary/chroma_db",
    embedding_function=embeddings,
)

doc_ids = vector_store.add_documents(split_docs)
print(f"Document IDs: {doc_ids}")


# Create a simple agent that can query the vector store
@tool
def query_vector_store(query: str) -> str:
    """
    Query the vector store for similar chunks of text based on the input query.

    Args:
        query (str): The query string to search for similar chunks in the vector store.

    Returns:
        str: A string representation of the top 3 most similar chunks from the vector store.
    """
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])


model = init_chat_model(model="gpt-4o")

agent = create_agent(
    tools=[query_vector_store],
    model=model,
)

query = "What were the pain points of the customers?"

for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
):
    event["messages"][-1].pretty_print()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("PDF loading and splitting completed.")
