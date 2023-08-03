import os
import tempfile
import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# Initializing the models and configurations
LLM_MODEL="gpt-4"
LLM_TEMPERATURE=0
EMBEDDING_MODEL="text-embedding-ada-002"
CHAIN_TYPE="stuff"
DOCUMENT_TYPE="pdf"

# Variables for storing API keys and other configurations
OPEN_API_KEY = None
PINECONE_API_KEY = None
PINECONE_ENV = None
PINECONE_INDEX = None
DOCUMENT_SRC = None

def setup():
    # Check if 'generated' and 'past' keys exist in the session state, if not, initialize them.
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

def get_prompt():
    # Define a text input field in the interface for user prompts.
    input_prompt = st.text_input("Your prompt: ", placeholder="Ask me anything about the document...", key="input_prompt")
    return input_prompt

def user_prompt_interface(user_form_container):
    # Interface for the user prompt input
    with user_form_container:
        with st.form(key='qa_form', clear_on_submit=True):
            user_prompt = get_prompt()
            submit_button = st.form_submit_button(label='Send')
            if submit_button and user_prompt:
                return user_prompt
            else:
                return None

def process_pdf(file):
    # Process a PDF document and split it into pages.
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file.seek(0)
        loader = PyPDFLoader(tmp_file.name)
        pages = loader.load_and_split()
        os.remove(tmp_file.name)
        return pages
    except Exception as e:
        raise Exception(f"Failed to process the PDF file: {e}")

def initialize_retriever(pages=None):
    # Initialize the vector retriever
    try:
        # model: str = "text-embedding-ada-002", chunk_size: int = 1000
        embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY, model=EMBEDDING_MODEL)
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        if pages:
            vectordb = Pinecone.from_documents(pages, embeddings, index_name=PINECONE_INDEX)
        else:
            vectordb = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)
        return vectordb.as_retriever(), embeddings
    except Exception as e:
        raise Exception(f"Failed to initialize retriever: {e}")

def generate_response(user_prompt_input, pages=None):
    # Generate a response for the user's prompt
    try:
        retriever, embeddings = initialize_retriever(pages)
        llm = ChatOpenAI(model_name=LLM_MODEL, temperature=LLM_TEMPERATURE, openai_api_key=OPEN_API_KEY)
        qa = RetrievalQA.from_chain_type(llm, chain_type=CHAIN_TYPE, retriever=retriever)
        response = qa.run(user_prompt_input)
        return response
    except Exception as e:
        raise Exception(f"Failed to generate response: {e}")

def validate_inputs():
    # Validate whether the required inputs are provided
    if not OPEN_API_KEY or not PINECONE_API_KEY or not PINECONE_ENV or not PINECONE_INDEX:
        st.warning(f"Please provide the missing fields.")
        return False
    return True

def get_env_variable(var_name):
    # Get the value of the given environment variable
    var_value = os.getenv(var_name)
    if not var_value:
        raise ValueError(f"{var_name} environment variable is not set.")
    return var_value

def main():
    # Main function to run the app
    global CHAIN_TYPE, DOCUMENT_TYPE, OPEN_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX, DOCUMENT_SRC

    setup()

    with st.sidebar:
        st.title('🚀 Streamlit Chat App')
        add_vertical_space(2)
        st.markdown('''
        ## Specs
        - OpenAI LLM
        - GPT 4 Model
        - Text Embedding Ada 002
        - PyPDFLoader Document Loader
        - Pinecone Vector Store
        - RetrievalQA Chain / Stuff Chain
        ''')
        add_vertical_space(2)
        if st.button('Clear Chat History'):
            # Clear the chat history lists
            st.session_state['generated'] = []
            st.session_state['past'] = []
        add_vertical_space(2)
        OPEN_API_KEY = get_env_variable("OPENAI_API_KEY") or st.text_input("OpenAI API key", type="password", key="open_api_key")
        PINECONE_API_KEY = get_env_variable("PINECONE_API_KEY") or st.text_input("Pinecone API key", type="password", key="pinecone_api_key")
        PINECONE_ENV = get_env_variable("PINECONE_ENV") or st.text_input("Pinecone environment", key="pinecone_env")
        PINECONE_INDEX = get_env_variable("PINECONE_INDEX") or st.text_input("Pinecone index name", key="pinecone_index")
        add_vertical_space(2)
        st.write('Made with love by Jos❤️roma')

    enable_upload = st.checkbox('Enable file upload')

    if enable_upload:
        DOCUMENT_SRC = st.file_uploader("Upload your document", type=f"{DOCUMENT_TYPE}")
    else:
        DOCUMENT_SRC = None
    user_form_container = st.container()
    user_prompt_input = user_prompt_interface(user_form_container)
    if user_prompt_input is not None and user_prompt_input.strip() != '' and validate_inputs():
        try:
            if DOCUMENT_SRC:
                # pinecone.delete_index(PINECONE_INDEX)
                # pinecone.create_index(PINECONE_INDEX, dimension=1536, metric="cosine", pods=1, pod_type="s1.x1")
                pages = process_pdf(DOCUMENT_SRC)
                response = generate_response(user_prompt_input, pages)
            else:
                response = generate_response(user_prompt_input)
            st.session_state.past.append(user_prompt_input)
            st.session_state.generated.append(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        exit(1)
