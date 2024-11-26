# import streamlit as st
# import os
# import pickle
# from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings  import OpenAIEmbeddings

# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback # helps us to know how much it costs us for each query

# from dotenv import load_dotenv


# #side bar contents
# with st.sidebar:
#     st.title(' New Chat ➕')
#     # st.markdown("""
#     # ## About
#     # This app is an LLM-powered chatbot built using:
#     # - [Streamlit](https://streamlit.io/)
#     # - [Langchain](https://python.langchian.com/)
#     # - [OpenAI](https://platform.openai.com/docs/models) LLM model
#     # - [Github](https://github.com/praj2408/Langchain-PDF-App-GUI) Repository
                
#     # """)
#     add_vertical_space(5)
#     st.write("")
    
    
# load_dotenv()
    
# def main():
#     st.header("Welcome to Pinpoint 💬")
    
#     # upload a PDF file
#     pdf = st.file_uploader("Contribute more accurate medical data:", type="pdf")
    
#     #st.write(pdf)
    
#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
        
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
            
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, # it will divide the text into 800 chunk size each (800 tokens)
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text=text)
        
#         #st.write(chunks)
        
        
#         ## embeddings
        
#         store_name = pdf.name[:-4]
        
        
#         vector_store_path = os.path.join('vector_store', store_name)
        
        
#         if os.path.exists(f"{vector_store_path}.pkl"):
#             with open(f"{vector_store_path}.pkl", 'rb') as f:
#                 VectorStore = pickle.load(f)
#             # st.write("Embeddings loaded from the Disk")
                
#         else:
#             embeddings = OpenAIEmbeddings()
#             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#             with open(f"{vector_store_path}.pkl", "wb") as f:
#                 pickle.dump(VectorStore, f)
            
#             # st.write("Embeddings Computation Completed ")
            
        
#         # Accept user questions/query
#     add_vertical_space(27)
#     query = st.text_input("Ask your emergency medical questions:")
#         #st.write(query)
        
#     if query:
        
#         docs = VectorStore.similarity_search(query=query, k=3) # k return the most relevent information
        
#         llm = OpenAI(model_name='gpt-3.5-turbo')
#         chain = load_qa_chain(llm=llm, chain_type='stuff')
#         with get_openai_callback() as cb:
            
#             response = chain.run(input_documents=docs, question=query)
#             print(cb)
#         st.write(response)
        
            
            

    
    
    
# if __name__ == "__main__":
#     main()
    

import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings  import OpenAIEmbeddings

from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback # helps us to know how much it costs us for each query

from dotenv import load_dotenv




#side bar contents
with st.sidebar:
    st.title(' New Chat ➕')
    # st.markdown("""
    # ## About
    # This app is an LLM-powered chatbot built using:
    # - [Streamlit](https://streamlit.io/)
    # - [Langchain](https://python.langchian.com/)
    # - [OpenAI](https://platform.openai.com/docs/models) LLM model
    # - [Github](https://github.com/praj2408/Langchain-PDF-App-GUI) Repository
                
    # """)
    #add_vertical_space(5)
    st.write("")
    
    
load_dotenv()
    
def main():
    st.header("Welcome to Pinpoint ⚕️")
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://pixabay.com/photos/milky-way-stars-night-sky-2695569/")
    background-size: cover;
    }
    [data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
    }
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # upload a PDF file
    pdf = st.file_uploader("Contribute more accurate medical data:", type="pdf")
    
    #st.write(pdf)
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # it will divide the text into 800 chunk size each (800 tokens)
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        #st.write(chunks)
        
        
        ## embeddings
        
        store_name = pdf.name[:-4]
        
        
        vector_store_path = os.path.join('vector_store', store_name)
        
        
        if os.path.exists(f"{vector_store_path}.pkl"):
            with open(f"{vector_store_path}.pkl", 'rb') as f:
                VectorStore = pickle.load(f)
            # st.write("Embeddings loaded from the Disk")
                
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{vector_store_path}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            
            # st.write("Embeddings Computation Completed ")
            
        
        # Accept user questions/query
    #add_vertical_space(27)
    query = st.text_input("Ask your emergency medical questions:")
        #st.write(query)
        
    if query:
        
        docs = VectorStore.similarity_search(query=query, k=3) # k return the most relevent information
        
        llm = OpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        with get_openai_callback() as cb:
            
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)
        
            
        
if __name__ == "__main__":
    main()
    
