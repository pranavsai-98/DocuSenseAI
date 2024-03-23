from nltk.tokenize import sent_tokenize
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import config
import boto3
import time
import uuid
import openai
from openai import OpenAI
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed


nltk.download('punkt')

s3 = boto3.resource(
    service_name='s3',
    region_name=config.AWS_DEFAULT_REGION,
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
)

dynamodb = boto3.resource('dynamodb',
                          aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                          region_name=config.AWS_DEFAULT_REGION)


os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
client = OpenAI()
OpenAI.api_key = os.environ["OPENAI_API_KEY"]


def summarize_chunk(chunk, model="gpt-3.5-turbo-instruct", max_tokens=1024):
    """
    Function to summarize a single chunk of text.
    """
    prompt = f"Summarize this document:\n\n{chunk}"
    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=0.5,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()


def summarize_with_context_parallel(chunks, model="gpt-3.5-turbo-instruct", max_tokens=1024, workers=5):
    """
    Summarizes text chunks in parallel.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all chunk summarization tasks at once and use as_completed to retrieve their results as they come.
        futures = [executor.submit(
            summarize_chunk, chunk, model, max_tokens) for chunk in chunks]
        summaries = [future.result() for future in as_completed(futures)]

    # Combine summaries of all chunks into a final summary
    final_summary = ' '.join(summaries)
    return final_summary


def create_overlapping_chunks(text, chunk_size=3000, overlap=500):
    """
    Create overlapping chunks from the input text.
    """
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunks.append(text[i:end])
        i += (chunk_size - overlap)
    return chunks


def concise_final_summarization(summary_text, model="gpt-3.5-turbo", max_tokens=300):
    """
    Perform a final summarization pass to condense the combined chunk summaries
    into a more concise summary using the chat-based model interaction.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following document: {summary_text}"}
        ],
        max_tokens=max_tokens
    )

    # Correctly access the 'content' of the response
    final_summary = response.choices[0].message.content
    return final_summary


def handle_document_query(query, document_text):

    relevant_sections = document_text

    # Generate prompt with document context. Here, 'relevant_sections' should be a string
    # containing the information from the document that is relevant to the user's query.
    prompt = f"Given the document context: {relevant_sections}\n\nHow would you answer the user's question: '{query}'?"

    # Use OpenAI Chat Completion with the constructed prompt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant familiar with the document."},
            {"role": "user", "content": prompt}
        ]
    )

    # Return the assistant's message
    return response.choices[0].message.content


def upload_file_to_s3(s3, file):
    """
    Upload a file to an S3 bucket

    :param file: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """

    object_name = file.name

    try:
        s3.Bucket("chatbot-filestorage").put_object(Key=object_name, Body=file)
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True


def save_chat_to_dynamodb(dynamodb, user_id,  chat_history):
    table = dynamodb.Table('chatbot-chat-history')
    for message in chat_history:
        timestamp = int(time.time() * 1000)
        # Generate a unique key using a UUID
        history_key = user_id
        try:
            table.put_item(
                Item={
                    'history': history_key,  # Unique identifier for each item
                    'timestamp': timestamp,
                    'user_id': user_id,  # Replace with actual user ID if available
                    'role': message['role'],
                    'content': message['content']
                }
            )
        except Exception as e:
            print(f"Error saving to DynamoDB: {e}")


# Sidebar contents
with st.sidebar:
    st.title('Quick Start Guide')
    st.markdown("""
    **Intelligent PDF Assistant** leverages advanced NLP to interactively answer questions from your PDFs and provide concise summaries.

    **Features:**
    - **Interactive Q&A:** Directly ask questions about the contents of your uploaded PDF.
    - **Document Summarization:** Get swift, comprehensive summaries of lengthy documents.

    **How to Use:**
    1. **Upload PDF:** Click 'Upload your PDF' and select a document.
    2. **Ask or Summarize:** Type questions related to the document or use the 'Summarize Document' button for a quick overview.
    3. **Reset:** Use 'End Session' to clear the current session and start over.

    Powered by OpenAI's GPT and proprietary NLP technology for accurate, real-time insights.
    """)

    st.markdown(
        "[Learn More](https://github.com/pranavsai-98/DocuSenseAI)", unsafe_allow_html=True)


def main():
    st.header("Chat with PDF 💬")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(
            uuid.uuid4()) + "_" + str(int(time.time()))

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:

        uploaded_to_s3 = upload_file_to_s3(s3, pdf)
        if uploaded_to_s3:
            st.success("File uploaded to S3")
        else:
            st.error("Failed to upload file to S3")

        pdfReader = PdfReader(pdf)
        raw_text = ''
        for page in pdfReader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Load or create FAISS index
        embeddings = OpenAIEmbeddings()
        # st.write(texts)
        docsearch = FAISS.from_texts(texts, embeddings)

    if st.button('Summarize Document'):
        if pdf is not None:
            with st.spinner('Summarizing...'):
                chunks = create_overlapping_chunks(
                    raw_text, chunk_size=3000, overlap=500)
                intermediate_summary = summarize_with_context_parallel(chunks)
                # Generate a concise summary from the intermediate summary
                final_concise_summary = concise_final_summarization(
                    intermediate_summary)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Document Summary:\n" + final_concise_summary})

        else:
            st.error("Please upload a PDF document.")

    # Chat input
    query = st.chat_input("Ask questions about your PDF file:")
    if query:
        # Process the query
        docs = docsearch.similarity_search(query=query, k=3)
        response = handle_document_query(query, docs)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append(
            {"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            with st.chat_message("User", avatar="👩‍💻"):
                st.write(content)
        else:
            with st.chat_message("Assistant"):
                st.write(content)

    if st.button('End Session'):
        # Save the current chat history
        save_chat_to_dynamodb(
            dynamodb, st.session_state.user_id, st.session_state.messages)

        # Clear the session state
        st.session_state.messages = []

        # Reload the page to start a new session
        st.rerun()


if __name__ == '__main__':
    main()
