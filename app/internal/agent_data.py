import openai.error
from app.internal import chunking

from app import settings


from sklearn.metrics.pairwise import cosine_similarity as cs
import openai
import numpy as np
from typing import Dict, List, Optional, Literal
import pydantic
import traceback
import spacy
from concurrent.futures import ThreadPoolExecutor
from app.api_clients.mclapsrl import mclapsrlClient
from app.internal.embedding_request import embed


nlp = spacy.load("en_core_web_sm")
rate_limiter = mclapsrlClient()

openai.api_key = settings.OPEN_AI_KEY




class Chunk(pydantic.BaseModel):
    # parent_DataStr_id: uuid.UUID
    index: int 
    # DataStr_index: int 
    string: str 
    embedding_vector: np.ndarray 
    conjugate_vector: Optional[np.ndarray] = None
    
    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator('embedding_vector', 'conjugate_vector', pre=True)
    def check_numpy_array(cls, v):
        if not isinstance(v, np.ndarray):
            return np.array(v)
        return v
    
    #computation include self similarity as chunk is added to DataChunks before this function is called
    def compute_conjugate_vector(self, chunk_embeddings: List[np.ndarray]) -> None:
        self.conjugate_vector = np.array([])
        if len(chunk_embeddings) == 0:

            self.conjugate_vector =  np.array([0])

        else:
            embedding_vector_reshaped = self.embedding_vector.reshape(1, -1)
            chunk_embeddings_stacked = np.vstack(chunk_embeddings)
            similarities = cs(embedding_vector_reshaped, chunk_embeddings_stacked)
            conjugate_vector = similarities.flatten()
            rescaled_conjugate_vector = (conjugate_vector + 1) / 2
            rescaled_conjugate_vector[self.index] = 0 ####sets self similarity to 0

            self.conjugate_vector = rescaled_conjugate_vector


# #####################
# class DataStr(pydantic.BaseModel):
#     DataStr_id: uuid.UUID
#     index: int
#     string: str
#     chunks: List[Chunk]
#     embedding_vector: np.ndarray

#     class Config:
#         arbitrary_types_allowed = True

#     @pydantic.validator('embedding_vector')
#     def check_numpy_array(cls, v):
#         assert isinstance(v, np.ndarray), 'must be a numpy array'
#         return v

#     @pydantic.validator('DataStr_id')
#     def check_uuid(cls, v):
#         assert isinstance(v, uuid.UUID), 'must be a UUID'
#         return v
    


class AgentData:

    def __init__(self, 
                 memory_limit: int, 
                 chunk_size: int, 
                 sampling_top_n: int, 
                 reconstruction_top_n: int, 
                 reconstruction_trigger_factor: float, 
                 memory_loss_factor: Optional[float] = 0.1, 
                 embedding_dim: Optional[int] = 512,
                 embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-small"):  
        
        # self.DataStrings: List[DataStr] = []  
        self.DataChunks: List[Chunk] = [] 
        self.embedding_model: str = embedding_model 


        self.memory_size: int = memory_limit       #####INCEASE FOR PRODUCTION, unit is number of chunks
        self.chunk_size: int = chunk_size    ####unit in number of tokens
        self.query_sampling_n: int = sampling_top_n
        self.reconstruction_sampling_n: int = reconstruction_top_n
        self.reconstruction_trigger_length: int = round(reconstruction_trigger_factor * memory_limit)  #in n. of chunks
        self.loss_factor: float = memory_loss_factor 
        self.dimension: int = embedding_dim
        
    def isotropic_rescaler(self, value: float) -> float:
        rescaled_value = (value + 1)/2
        return rescaled_value
    
    def compute_conjugate_vector(self, embedding_vector: np.ndarray) -> np.ndarray:
        embedding_vector_reshaped = embedding_vector.reshape(1, -1)
        chunk_embeddings = np.vstack([chunk.embedding_vector for chunk in self.DataChunks])
        similarities = cs(embedding_vector_reshaped, chunk_embeddings)
        conjugate_vector = similarities.flatten()
        rescaled_conjugate_vector = np.array([self.isotropic_rescaler(value) for value in conjugate_vector])
        return rescaled_conjugate_vector

    def update_conjugate_vectors(self, new_chunk: Chunk) -> None:
        
        
        if len(self.DataChunks) <= 1:
            pass
        else:
            for chunk in self.DataChunks[:-1]:
                chunk.conjugate_vector = np.append(chunk.conjugate_vector, new_chunk.conjugate_vector[chunk.index])
            
            # #all chunks except the last one, i = 0,1,2 ... 
            # for i in range(len(self.DataChunks)-1):
            #     ith_chunk = self.DataChunks[i]
            #     ith_chunk.conjugate_vector = np.append(ith_chunk.conjugate_vector, new_chunk.conjugate_vector[i])

    def embed_string(self, input_string: str) -> np.ndarray:
        embedding = embed(
            input_string, 
            dimension = self.dimension, 
            embedding_model = self.embedding_model
            )
        return embedding

    #quick check to see if there is anything more related in database
    def L0_query(self, query_string:str) -> Optional[List[int]]:
        if len(self.DataChunks) == 0:
            return [0]
        else:
            try:
                query_embedding = self.embed_string(query_string)
                query_conjugate_vector = self.compute_conjugate_vector(query_embedding)
                top_1: List[int] = sorted(query_conjugate_vector.tolist(),reverse=True)[0:1]
                return top_1
            except Exception as e:
                print(f"Error in L0 query: {e}")
                return [0]
            
    #returns 5 strings, each string has six chunks 6 chunks, each chunk 20 tokens
    def fast_query(self, query_string: str) -> list[str]:
        #returns 5 most related chunks to the target chunk, sorting through conjugate vector
        def chunk_group_reconstruct(chunk: Chunk) -> str:
            top_n = sorted(enumerate(chunk.conjugate_vector.tolist()), key=lambda x: x[1], reverse=True)[0:self.reconstruction_sampling_n]
            target_chunk_list = [self.DataChunks[index] for index, _ in top_n]
            string_group = "\n".join([chunk.string for chunk in target_chunk_list])
            return string_group
        
        if len(self.DataChunks) == 0:
            return []
        
        else:
            try:
                query_embedding = self.embed_string(query_string)
                query_conjugate_vector = self.compute_conjugate_vector(query_embedding)
                top_n = sorted(enumerate(query_conjugate_vector.tolist()), key=lambda x: x[1], reverse=True)[0:self.query_sampling_n]    ###top 5 chunks identified through query similarity
                target_chunk_list = [self.DataChunks[index] for index, _ in top_n]
                if len(self.DataChunks) > self.reconstruction_trigger_length:
                    reconstructed_strings = []
                    for target_chunk in target_chunk_list:
                        reconstructed_strings.append(chunk_group_reconstruct(target_chunk))
                    return reconstructed_strings
                else: 
                    return [chunk.string for chunk in target_chunk_list]    
                
            except Exception as e:
                print(f"Error in fast query: {e}")
                return []

    def resturcture_memory(self):
        try:
            if len(self.DataChunks) >= self.memory_size: 
                # Calculate average similarities for each chunk
                chunk_average_similarities = [np.mean(chunk.conjugate_vector) for chunk in self.DataChunks]
                # Find least relevant chunks by sorting indices based on similarity
                least_relevant_chunk_indices = sorted(enumerate(chunk_average_similarities), key=lambda x: x[1])[0:round(self.loss_factor * len(self.DataChunks))]
                # Extract the indices only
                least_relevant_chunk_indices = [index for index, _ in least_relevant_chunk_indices]
                # Delete conjugate vector entries BEFORE modifying self.DataChunks
                for chunk in self.DataChunks:
                    # Deleting from conjugate vectors using the original indices
                    chunk.conjugate_vector = np.delete(chunk.conjugate_vector, least_relevant_chunk_indices)
                # Sort indices in reverse order and delete from DataChunks to avoid index shifting issues
                least_relevant_chunk_indices = sorted(least_relevant_chunk_indices, reverse=True)
                for index in least_relevant_chunk_indices:
                    del self.DataChunks[index]
                # Reindex the remaining chunks in DataChunks
                for i, chunk in enumerate(self.DataChunks):
                    chunk.index = i  # Update the chunk index after deletions
            else:
                pass
        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Error in restructuring memory in agent_data: {e}")


    ####add in logic for deleting old chunks, restruturing existing chunks and restructuring of relational matrix once max chunk size is hit
    def add_data_str(self, input_string: str):

        def add_chunk(input_string: str, input_string_embedding: np.ndarray) -> None:
        
            chunk = Chunk(
                index=len(self.DataChunks), ## 0th indexing same as python list indexing
                string=input_string,
                embedding_vector=input_string_embedding
            )
            self.DataChunks.append(chunk)
            chunk.compute_conjugate_vector([current_chunk.embedding_vector for current_chunk in self.DataChunks])
            self.update_conjugate_vectors(chunk)

        
        
        try:
            list_of_chunked_str: List[str] = chunking.chunk_string(input_string, chunk_size = self.chunk_size)
            with ThreadPoolExecutor(max_workers=len(list_of_chunked_str)) as executor:
                list_of_chunk_embeddings: List[np.ndarray] = list(executor.map(self.embed_string, list_of_chunked_str))
            for string, embedding in zip(list_of_chunked_str, list_of_chunk_embeddings):
                add_chunk(string, embedding)
            self.resturcture_memory()
            
        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Error in add_str in agent_data: {e}")
        
        
    def query(self, input_string: str, evalutator_k: Optional[float] = 0):
        if len(self.DataChunks) == 0:
            return []
        else:
            relatedness = self.L0_query(input_string)[0]
            if relatedness >= evalutator_k:
                return self.fast_query(input_string)
            else:
                return []

        
    

            
            
            

 # datastr = input_string
            # datastr_embedding = self.embed_string(input_string)
            # datastr_uuid = uuid.uuid4()
            # datastr_chunked = chunking.chunk_string(input_string, chunk_size = self.chunk_size)
            
            # list_of_chunk_embeddings: List[np.ndarray] = []
            # with ThreadPoolExecutor(max_workers=len(datastr_chunked)) as executor:
            #     list_of_chunk_embeddings = list(executor.map(self.embed_string, datastr_chunked))

            # list_of_chunks: List[Chunk] = []    
            # for string, embedding in zip(datastr_chunked, list_of_chunk_embeddings):
            #     data_chunk = add_chunk(datastr_index= len(list_of_chunks) , parent_str_id = datastr_uuid, input_string = string, input_string_embedding = embedding)
            #     list_of_chunks.append(data_chunk)
            
            # data_str = DataStr(
            #     DataStr_id = datastr_uuid,
            #     index = len(self.DataStrings),
            #     chunks = list_of_chunks,
            #     string = datastr,
            #     embedding_vector = datastr_embedding
            # )
            # self.DataStrings.append(data_str)



    
 






    

    

