from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()
client=OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
# embedding_model=OpenAIEmbeddings(model="text-embedding-3-large")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def retrieval(userquery):
    vector_db=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="Qacollection",
    embedding=embeddings)
    # userquery=input("Ask something:")
    search_result=vector_db.similarity_search(query=userquery)
    context="\n\n\n".join([f"Page Content:{result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location:{result.metadata['source']}" for result in search_result])
    system_prompt=f"""You are helpful assistant with who answers user query based on available context retrieved from a pdf file along with page number and page_contents.
    you should only ans the user based on the following context and navigate the user to open the right page number to know more.
    Context:
    {context}
    """
    response=client.chat.completions.create(
        model="gemini-3-flash-preview",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":userquery}
        ]
    )
    return response.choices[0].message.content
    # print("bot: ",response.choices[0].message.content)


    