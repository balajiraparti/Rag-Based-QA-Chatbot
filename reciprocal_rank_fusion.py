from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from collections import defaultdict
import json
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
client=OpenAI()
from typing import List
class ParallelQuerySchema(BaseModel):
    queries: List[str]
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


context_array=[]
def parallel_query(query: str,k:int):
        user_prompt_parallel_query=f"""
       generate  {k} different  variations  this query that would help us to retrieve relevant documents.
       original question:{query}
        the question should help us to get additional information about the original question
        """
        response=client.beta.chat.completions.parse(
            model="gpt-4o",
           response_format=ParallelQuerySchema,
            messages=[
                {"role":"user","content":query},
                {"role":"system","content":user_prompt_parallel_query}
            ]
            )
       
        return response.choices[0].message.parsed.queries


def search_chunks(questions: list):
    vector_db=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="Qacollection",
    embedding=embedding_model,
)
    rankings=[]
    for query in questions:
        search_result=vector_db.similarity_search(query=query)
        rankings.append([doc for doc in search_result])
        context_array.append([doc for doc in search_result])
    return rankings
def reciprocal_rank_fusion(results_list, k: int = 60):
    rrf_scores = defaultdict(float)  
    all_unique_chunks = {}  
    
    chunk_id_map = {}
    chunk_counter = 1
    chunk_counter = 1
    scores = defaultdict(float)
 
    for result in results_list:
        
        for rank, chunk in enumerate(result):
            chunk_content=chunk.page_content
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            chunk_id = chunk_id_map[chunk_content]
            
       
            all_unique_chunks[chunk_content] = chunk
            
            position_score = 1 / (k + rank)
            rrf_scores[chunk_content] += position_score
    
            
            scores[chunk_content] += 1 / (k + rank + 1)  

    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    return sorted_chunks
def build_chunks(docs:list):
     print(docs)
     lines = []
     for rank, (doc, rrf_score) in enumerate(docs[:5], 1):
         lines.append(doc.page_content)

     return "\n\n".join(lines)
 

def generate_response(userquery:str,q:int=3):
    parallel_queries=parallel_query(userquery,q)
    parallel_chunks=search_chunks(parallel_queries)
    ranked_docs=reciprocal_rank_fusion(parallel_chunks)
    combined_context=build_chunks(ranked_docs)
    system_prompt=f"""You are helpful assistant with who answers above given user query based on available context retrieved from a pdf file.
    you should only ans the user based on the following context.
    Context:
    {combined_context}
    """

    response=client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":userquery}
    ]
   )   
    
    return response.choices[0].message.content
