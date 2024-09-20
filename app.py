import time
import json
import requests
from PyPDF2 import PdfReader
import streamlit as st
import os
import validators
from datetime import datetime
import faiss
import pickle
import numpy as np
import easyocr
from pdf2image import convert_from_path
#from open_source_t5_model import open_source_modal

# Import embedding functions
# from cohere import cohere_embed
# from gemni import gemni_embed
from sentence_transformers import SentenceTransformer
my_secret = os.getenv('GemniKey')
# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])
# Directory to save uploaded PDFs
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
#######################################################
# Ensure the directory exists
save_dir = "faiss"
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
############################################################
def load_faiss_index_and_metadata(filename):
    # Path to the saved files
    index_path = os.path.join(save_dir, f"{filename}.index")
    metadata_path = os.path.join(save_dir, f"{filename}.pkl")

    # Load the FAISS index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        raise FileNotFoundError(f"FAISS index file '{index_path}' not found.")

    # Load the metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        raise FileNotFoundError(f"Metadata file '{metadata_path}' not found.")

    return index, metadata
###############################################################
# Get list of PDF files
def list_uploaded_pdfs(upload_folder):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return [f for f in os.listdir(upload_folder) if f.endswith('.pdf')]

# Split string into chunks
def split_into_chunks(input_string, chunk_size=200):
    return [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]

#read_pdf_with_ocr
def read_pdf_with_ocr(file_path,file_name):
    filename = os.path.splitext(file_name)[0]
    images = convert_from_path(file_path)
    pdftext=''
    output = "img"
    imgpath = []
    os.makedirs(output,exist_ok=True)

    for i , image in enumerate(images):
        imagepath = os.path.join(output,f'page_{i+1}.png')
        image.save(imagepath,"PNG")
        imgpath.append(imagepath)

    for image_path in imgpath:
        results = reader.readtext(image_path)
    # Print extracted text
        for result in results:
            text = result[1]
            pdftext +=text

    return pdftext

################################################################################
# Read PDF without OCR
def read_pdf_without_ocr(file_path, file_name):
    filename = os.path.splitext(file_name)[0]
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        # Split the text into chunks
        chunks = split_into_chunks(text)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)        
        embeddings_np = np.array(chunk_embeddings.cpu())  # Use .cpu() if using GPU, otherwise skip .cpu()
        
        # Store metadata like chunk ID and original text
        metadata = {i: {'chunk': chunks[i], 'text': chunks[i]} for i in range(len(chunks))}

        # Initialize FAISS index for L2 distance
        d = embeddings_np.shape[1]  # Dimensionality of embeddings
        index = faiss.IndexFlatL2(d)
        # Add the embeddings to the FAISS index
        index.add(embeddings_np)
        try:
        # Save the FAISS index
            index_file_path = os.path.join(save_dir, f"{filename}.index")
            faiss.write_index(index, index_file_path)
            print(f"FAISS index saved to {index_file_path}")

        # Save the metadata
            metadata_file_path = os.path.join(save_dir, f"{filename}.pkl")
            with open(metadata_file_path, "wb") as f:
                pickle.dump(metadata, f)
            print(f"Metadata saved to {metadata_file_path}")

        except Exception as e:
            print(f"Error saving FAISS index and metadata: {str(e)}")

    except Exception as e:
        text = f"Error reading PDF: {str(e)}"
    
    # Return the embeddings for further processing if needed
    return "embeddings_np"


#######################################################
# Sidebar for URL input and PDF upload
st.sidebar.subheader("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
st.sidebar.write("### Choose how to read the PDF")
ocr_option = st.sidebar.radio("Select the method to read the PDF:", ["Without OCR", "With OCR (for scanned documents)"])

if ocr_option == "Without OCR":
    if uploaded_file:
        start_time = time.time()  # Record start time
        # Show loader while uploading and processing the file
        with st.spinner('Uploading and processing...'):
            # Simulate processing time (optional)
            time.sleep(2)

            # Save the uploaded PDF file to the directory
            st.sidebar.success("Reading PDF without OCR...")
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            filename = uploaded_file.name
            extracted_text = read_pdf_without_ocr(file_path, filename)
            st.text_area("Extracted Text", "Let's Start Chat", height=20)

        end_time = time.time()  # Record end time
        time_taken = end_time - start_time  # Calculate time taken
        # Success message with time taken
        st.sidebar.success(f"Saved file: {uploaded_file.name}")
        st.sidebar.info(f"Time taken: {time_taken:.2f} seconds")

elif ocr_option == "With OCR (for scanned documents)":
    if uploaded_file:
        start_time = time.time()  # Record start time
        # Show loader while uploading and processing the file
        with st.spinner('Uploading and processing...'):
            # Simulate processing time (optional)
            time.sleep(2)

            # Save the uploaded PDF file to the directory
            st.sidebar.success("Reading PDF with OCR...")
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_name = uploaded_file.name
            filename = os.path.splitext(file_name)[0]
            extracted_ocrtext = read_pdf_with_ocr(file_path, filename)

            chunks = split_into_chunks(extracted_ocrtext)
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True)        
            embeddings_np = np.array(chunk_embeddings.cpu()) 
                    # Store metadata like chunk ID and original text
            metadata = {i: {'chunk': chunks[i], 'text': chunks[i]} for i in range(len(chunks))}

            # Initialize FAISS index for L2 distance
            d = embeddings_np.shape[1]  # Dimensionality of embeddings
            index = faiss.IndexFlatL2(d)
            # Add the embeddings to the FAISS index
            index.add(embeddings_np)
            try:
            # Save the FAISS index
                index_file_path = os.path.join(save_dir, f"{filename}.index")
                faiss.write_index(index, index_file_path)
                print(f"FAISS index saved to {index_file_path}")

            # Save the metadata
                metadata_file_path = os.path.join(save_dir, f"{filename}.pkl")
                with open(metadata_file_path, "wb") as f:
                    pickle.dump(metadata, f)
                print(f"Metadata saved to {metadata_file_path}")

            except Exception as e:
                print(f"Error saving FAISS index and metadata: {str(e)}")


            
        st.text_area("Extracted Text", "Let's Start Chat", height=20)

        end_time = time.time()  # Record end time
        time_taken = end_time - start_time  # Calculate time taken
        # Success message with time taken
        st.sidebar.success(f"Saved file: {uploaded_file.name}")
        st.sidebar.info(f"Time taken: {time_taken:.2f} seconds")



########################################################################
# Show list of uploaded files
st.sidebar.subheader("Uploaded Files")
pdf_files = list_uploaded_pdfs(UPLOAD_DIR)
selected_pdf = st.sidebar.selectbox("Select a PDF to chat with", pdf_files)

###########################################################
# Response generator function
def response_generator(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={my_secret}"
    payload = json.dumps({
      "contents": [{
        "parts": [{
          "text": "You are a helpful assistant who can provide detailed information on a wide range of topics. When a user asks a question, you should first retrieve relevant information from the passage and then generate a response based on that information. If you don't find relevant information, you should clearly state that and try to offer a general response or ask for more details.If the user asks irrelevant questions such as hi or how Question -" + prompt}]
      }]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    
    # Ensure that we correctly parse the JSON response
    data = response.json()
    text = data['candidates'][0]['content']['parts'][0]['text']
    
    #text = open_source_modal(prompt)
    
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

###########################################################
# Main chat code
st.title("Rag Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if selected_pdf:
    # Load existing FAISS index and metadata
    pdfname = os.path.splitext(selected_pdf)[0]
    index, metadata = load_faiss_index_and_metadata(pdfname)

    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display user message in chat
        # user_query_embed = gemni_embed(prompt)
        query_embedding = model.encode([prompt], convert_to_tensor=True).cpu().numpy()

        with st.chat_message("user"):
            st.markdown(selected_pdf)
            st.markdown(prompt)

        k = 1  # Retrieve top 3 results
        distances, indices = index.search(np.array(query_embedding), k)

        # Retrieve the metadata for the nearest neighbors
        relevant_chunks = ''
        for idx in indices[0]:
            if idx != -1:  # Check if valid index
                relevant_chunks += metadata[idx]['text']

        makePrompt = f"PASSAGE - {relevant_chunks} \n USER QUESTION - {prompt}"
        
        # Generate and display assistant response
        assistant_response = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for word in response_generator(makePrompt):
                assistant_response += word
                message_placeholder.markdown(assistant_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

else:
    st.markdown("Please select a PDF from the sidebar.")
