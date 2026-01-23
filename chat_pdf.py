import streamlit as st
import os
from ingestion import indexing
from retrieval_augmentation import retrieval


def save_uploaded_file(uploaded_file):
    
    save_path = os.path.join(os.getcwd(), "data")
    
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_location = os.path.join(save_path, uploaded_file.name)
    
    # 4. Write the file bytes
    with open(file_location, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_location
if "filestate" not in st.session_state:
    st.session_state.filestate=False
with st.sidebar:

        uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_pdf is not None and not st.session_state.filestate:
            if st.button('Process & Index PDF'):
                with st.spinner("Saving and Indexing..."):
                    path = save_uploaded_file(uploaded_pdf)
                    try:
                        indexing() # Run your ingestion logic here
                        st.session_state.filestate = True
                        st.success(f"File indexed successfully!")
                    except Exception as e:
                        st.error(f"Indexing error: {e}")
                    st.session_state.filestate = True
st.title("🤖 RAG based QA chatbot")
st.markdown('Ask any question about pdf')
if "message" not in st.session_state:
    st.session_state.messages=[]
    
for message in st.session_state.messages:
        role=message["role"]
        content=message["content"]
        st.chat_message(role).markdown(content)   
if  st.session_state.filestate==True:

    query=st.chat_input("Ask any question")

    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message('user').markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Performing retrieval..."):
                try:
                    response=retrieval(query)
                    st.session_state.messages.append({"role":"ai","content":response})
                    st.write(response)
                except:
                    st.write("error occurred while performing retrieval!")
else:
    st.write("Please upload PDF file")                   
   
        
            
              