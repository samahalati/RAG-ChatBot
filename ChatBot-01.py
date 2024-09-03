from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings
from huggingface_hub import login
from langchain.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
import chromadb  # Import Chroma
from langchain_chroma import Chroma




login('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', add_to_git_credential=True)




#Token and LLM
HF_TOKEN = "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"

llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=HF_TOKEN,
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature= 0.1,
    max_new_tokens= 500
)



#Make Loader:
loader=PyPDFLoader('PDF/Prompt-Breeding.pdf')
PDF = loader.load()
print(len(PDF))

#Chunk-Splitting: Defining chunk size and chunk overlap
chunk_size =1500
chunk_overlap = 150

#Using of RecursiveCharacterSplitter,CharacterSplitter:

def extract_text_from_pdf(documents):
    if documents is not None:
        text = ""
        for document in documents:
            text += document.page_content 
    return text


extracted_text = extract_text_from_pdf(PDF)

   
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""],
    length_function=len  
)


chunks = r_splitter.split_text(extracted_text)


print(len(chunks))


for i, chunk in enumerate(chunks[:]):  
   print(f"Chunk {i+1}: {chunk}\n")


#Embeddings:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", clean_up_tokenization_spaces=True)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Example model
chunk_embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]


##VECTORSTORE:
persist_directory = "chroma_db"  # Define the directory for persistence

# Initialize Chroma for vector storage
vectorstore = Chroma(persist_directory=persist_directory)

# Create the vector store from texts
vectordb = Chroma.from_texts(
    texts=chunks,  # Use the list of text chunks directly
    embedding=embedding_model,
    persist_directory=persist_directory
)

question = "tell about first order prompt generation"
docs = vectordb.similarity_search(question,k=3)
print(docs[0])


#creating chat templates: