from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader


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
chunk_size =15000
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





#creating chat templates: