import streamlit as st
from PyPDF2 import PdfReader  # used to extract text from pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter  # split text in smaller snippets
from langchain_community.llms import Ollama  # interact with ollama local server
from langchain_ollama import OllamaEmbeddings

import chromadb
import numpy
import uuid
import requests

res = requests.post('http://localhost:11434/api/embeddings',
                    json={
                        'model': 'nomic-embed-text',
                        'prompt': 'Hello world'
                    })


st.title("Chat BOT!!!")
uploaded_file = st.file_uploader("Upload a PDF file")

if uploaded_file is not None:
    prompt = st.text_area("Enter your prompt")
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating Response..."):

                # Parameters
                DIMENSION = len(res.json()['embedding'])  # 768 dimensions for nomic-embed-text embedding model
                EXTRACTED_TEXT_FILE_PATH = "pdf_txt.txt"  # text extracted from pdf
                PDF_FILE_PATH = uploaded_file
                CHUNK_SIZE = 150  # chunk size to create snippets
                CHUNK_OVERLAP = 20  # check size to create overlap between snippets
                OUTPUT_RESULT_COUNT = 5  # results of chunk from vector database.

                # initialize chroma DB
                chroma_client = chromadb.Client()
                collection = chroma_client.get_or_create_collection(name="test_chat_nomic")

                # Initialize Ollama-embeddings and test:
                embeddings = (
                    OllamaEmbeddings(model="nomic-embed-text")
                )

                embedding = embeddings.embed_query("Hello World")
                dimension = len(embedding)


                # Exract text from PDF
                def extract_text_from_pdf(file_path: str):
                    # Open the PDF file using the specified file_path
                    reader = PdfReader(file_path)
                    # Get the total number of pages in the PDF
                    number_of_pages = len(reader.pages)

                    # Initialize an empty string to store extracted text
                    pdf_text = ""

                    # Loop through each page of the PDF
                    for i in range(number_of_pages):
                        # Get the i-th page
                        page = reader.pages[i]
                        # Extract text from the page and append it to pdf_text
                        pdf_text += page.extract_text()
                        # Add a newline after each page's text for readability
                        pdf_text += "\n"

                    # Specify the file path for the new text file
                    file_path = EXTRACTED_TEXT_FILE_PATH

                    # Write the content to the text file
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(pdf_text)


                # Create Embeddings and Save it to chromaDB collection

                def create_embeddings(file_path: str):
                    # Initialize a list to store text snippets
                    snippets = []
                    # Initialize a CharacterTextSplitter with specified settings
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

                    # Read the content of the file specified by file_path
                    with open(file_path, "r", encoding="utf-8") as file:
                        file_text = file.read()

                    # Split the text into snippets using the specified settings
                    snippets = text_splitter.split_text(file_text)
                    # print(len(snippets))

                    x = numpy.zeros((len(snippets), dimension), dtype='float32')
                    ids = []

                    for i, snippet in enumerate(snippets):
                        # print(snippet)
                        embedding = embeddings.embed_query(snippet)
                        ids.append(get_uuid())
                        x[i] = numpy.array(embedding)

                    collection.add(embeddings=x,
                                   documents=snippets,
                                   ids=ids)


                def get_uuid():
                    return str(uuid.uuid4())


                extract_text_from_pdf(PDF_FILE_PATH)
                create_embeddings(EXTRACTED_TEXT_FILE_PATH)


                def answer_users_question(user_question):
                    embedding_arr = embeddings.embed_query(user_question)
                    result = collection.query(
                        query_embeddings=embedding_arr,
                        n_results=OUTPUT_RESULT_COUNT
                    )

                    return frame_response(result['documents'][0], user_question)


                def frame_response(results, ques):
                    joined_string = "\n".join(results)
                    prompt = joined_string + "\n Given this information, " + ques
                    llm = Ollama(
                        model="llama3"
                    )
                    return llm.invoke(prompt)

                st.write(answer_users_question(user_question=prompt))

                # Print end of processing
                print("----------------------")
