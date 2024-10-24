import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Literal
import pydantic
from app.api_clients.mclapsrl import mclapsrlClient
import time
import openai
from openai.error import OpenAIError, Timeout, ServiceUnavailableError, RateLimitError
from sklearn.metrics.pairwise import cosine_similarity as cs
from app import settings


rate_limiter = mclapsrlClient()
openai.api_key = settings.OPEN_AI_KEY

class Chunk(pydantic.BaseModel):
    string: str
    embedding_vector: np.ndarray
    relational_vector: Optional[np.ndarray] = None
    decay_factor: Optional[float] = 0.9

    class Config:
        arbitrary_types_allowed = True

    #parallelized in ChunksArray Class when computing
    def compute_relational_vector(self, principal_components: np.ndarray):
        n_components = principal_components.shape[0]
        self.relational_vector = np.zeros(n_components)
        for i, e_i in enumerate(principal_components):
            projection = np.dot(self.embedding_vector, e_i)
            self.relational_vector[i] = projection

    #depends on number of new chunks added in a single batch
    def decay_relational_vector(self, decay_strength: int):
        self.relational_vector = self.relational_vector * self.decay_factor**decay_strength 

#k_similarity search for chunks
class ChunksList(pydantic.BaseModel):
    chunks: list[Chunk] = []
    n_top: Optional[int] = 5

    def length(self):
        return len(self.chunks) 
    
    def add_chunks(self, chunks: list[Chunk]):
        self.chunks.extend(chunks)

    def query(self, query_chunks: list[Chunk]) -> list[Chunk]:
        query_embeddings = [chunk.embedding_vector for chunk in query_chunks]
        for embedding in query_embeddings:
            embedding_reshaped = embedding.reshape(1, -1)
            chunk_embeddings = np.vstack([chunk.embedding_vector for chunk in self.chunks])
            similarities = cs(embedding_reshaped, chunk_embeddings)
        top_n_similarities = sorted(enumerate(similarities.tolist()), key=lambda x: x[1], reverse=True)[0:self.n_top]
        top_indices = [index for index, _ in top_n_similarities]
        return [self.chunks[index] for index in top_indices]
        

class ChunksArray(pydantic.BaseModel):

    chunks: list[Chunk] = []
    new_chunks: list[Chunk] = 0 
    
    target_explained_variance: Optional[float] = 0.9     #higher is more compute 
    num_components: int = 0
    dataset_explained_variance: float = 0
       
    principal_components: np.ndarray = None
    ipca: IncrementalPCA = None      # Will hold Incremental PCA object
   
    class Config:
        arbitrary_types_allowed = True

    def initialize(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.initialize_components()
        self.initialize_chunk_weights()

    def initialize_components(self):
        
        V = np.vstack([chunk.embedding_vector for chunk in self.chunks])
        self.ipca = IncrementalPCA(n_components=min(V.shape[0], V.shape[1]))  # Initialize IPCA object
        self.ipca.fit(V)  
        #
        self.num_components = np.argmax(np.cumsum(self.ipca.explained_variance_ratio_) >= self.target_explained_variance) + 1
        # Use the principal components from IPCA
        self.principal_components = self.ipca.components_[:self.num_components]
        # Set the explained variance
        self.dataset_explained_variance = np.sum(self.ipca.explained_variance_ratio_[:self.num_components])
            
    def initialize_chunk_weights(self):
        
        with ThreadPoolExecutor(max_workers=min(8, len(self.chunks))) as executor:       
            for chunk in self.chunks:
                executor.submit(chunk.compute_relational_vector, self.principal_components)
            executor.shutdown(wait=True)

    ##########workflow for adding new chunks
    def add_chunks(self, chunks: list[Chunk]):
        self.new_chunks.extend(chunks)
        N_threshold = self.calculate_recompute_threshold()
        if len(self.new_chunks) >= N_threshold:
            self.update_principal_components()
            self.update_chunk_weights()
        else: 
            self.calculate_chunk_weights()
    
    def calculate_chunk_weights(self):
        with ThreadPoolExecutor(max_workers=min(8, len(self.new_chunks))) as executor:       
            for chunk in self.new_chunks:
                executor.submit(chunk.compute_relational_vector, self.principal_components)
            executor.shutdown(wait=True)
            #update logic here for         
    
    def update_principal_components(self):
        V_new = np.vstack([chunk.embedding_vector for chunk in self.new_chunks])
        self.ipca.partial_fit(V_new)
        # Update principal components with the incrementally learned components
        self.num_components = np.argmax(np.cumsum(self.ipca.explained_variance_ratio_) >= self.target_explained_variance) + 1
        self.principal_components = self.ipca.components_
        
        self.dataset_explained_variance = np.sum(self.ipca.explained_variance_ratio_)


    def update_chunk_weights(self):
        with ThreadPoolExecutor(max_workers=min(8, len(self.chunks))) as executor:       
            for chunk in self.chunks:
                executor.submit(chunk.compute_relational_vector, self.principal_components)
            executor.shutdown(wait=True)
        self.chunks.extend(self.new_chunks)
        self.new_chunks = []
        
    # number of unaccounted for chunks which would trigger a PCA recalculation
    def calculate_recompute_threshold(self) -> int:
        N = len(self.chunks)
        if N == 0:
            return 1  # No chunks, no need for recomputation
        
        k = 0.1  # You can adjust this constant to control the decay rate
        recompute_threshold = round((0.2 + (0.8 / (1 + k * np.log(N + 1)))) * N)
        return recompute_threshold

    def query(self, query_chunks: list[Chunk], top_n: Optional[int] = 5, attention_length: Optional[int] = 2) -> list[Chunk]:
        #step 1: compute relational vectors for query chunks
        with ThreadPoolExecutor(max_workers=min(8, len(query_chunks))) as executor:
            for query_chunk in query_chunks:
                executor.submit(query_chunk.compute_relational_vector, self.principal_components)
            executor.shutdown(wait=True)

        #step 2: combine new_chunks and chunks to construct a list of all chunks for query
        search_space = self.chunks + self.new_chunks

        #step 3: compute cosine similarity of relational vectors between query chunks and existing chunks
        similarities = cs(np.vstack([query_chunk.relational_vector for query_chunk in query_chunks]), np.vstack([chunk.relational_vector for chunk in search_space]))

        #step 4: rank each similarity array in terms of average similarity
        average_similarities = np.mean(similarities, axis=0)
        average_similarities_sorted = sorted(enumerate(average_similarities.tolist()), key=lambda x: x[1], reverse=True)
        most_important_query_chunk_indexes = [index for index, _ in average_similarities_sorted][0:attention_length]

        #step 5: look at the top_n similar relational vectors in most important query chunks and return identified chunks
        top_n_chunks = []
        for index in most_important_query_chunk_indexes:
            top_n_similarities = sorted(enumerate(similarities[index].tolist()), key=lambda x: x[1], reverse=True)[0:top_n]
            top_indices = [index for index, _ in top_n_similarities]
            top_n_chunks.extend([search_space[index] for index in top_indices])

        return top_n_chunks








        







class AgentData:

    def __init__(self, 
                 pca_threshold: int,  
                 chunks: list[Chunk], 
                 n_dim: int,
                 agent_id: str,
                 max_memory: int,
                 chunk_size: int,
                 top_n_sampling: int):
        
        
        

        self.dimension = n_dim
        
        self.pca_array_threshold: int = pca_threshold
        self.agent_id: str = agent_id 


        self.memory_size: int = max_memory       #####INCEASE FOR PRODUCTION, unit is number of chunks
        self.chunk_size: int = chunk_size    ####unit in number of tokens
        self.query_sampling_n: int = top_n_sampling


        
    def embed_text(self, string: str, embedding_model: Optional[Literal["text-embbedding-3-small", "text-embedding-3-large"]] = "text-embedding-3-small") -> np.ndarray:
        
        def normalize_l2(x):
            x = np.array(x)
            if x.ndim == 1:
                norm = np.linalg.norm(x)
                if norm == 0:
                    return x
                return x / norm
            else:
                norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
                return np.where(norm == 0, x, x / norm)
            

        while rate_limiter.model_status(embedding_model) == False:
            time.sleep(2)
            continue
        retries = 5
        while retries > 0:
            try:
                response=openai.Embedding.create(
                    model = embedding_model,
                    input=str(string)
                    )
                rate_limiter.new_response(response)
                embedding = np.array(response['data'][0]['embedding'][:self.dimension])
                return embedding
            except (openai.error.OpenAIError, openai.error.Timeout, openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
                print(f"Error while embedding in agent data: {e}")
                retries -= 1
                time.sleep(5)
                continue



    def calculate_embeddings(self, strings: list[str]) -> list[np.ndarray]:
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=min(8, len(strings))) as executor:
            embeddings = list(executor.map(self.embed_text, strings))
        




#old code

# class Chunk(pydantic.BaseModel):
#     string: str
#     embedding_vector: np.ndarray
#     relational_vector: np.ndarray
#     decay_constant: Optional[float] = 0.95

#     class Config:
#         arbitrary_types_allowed = True

#     def compute_relational_vector(self, principal_components: np.ndarray):
#         n_components = principal_components.shape[0]
#         self.relational_vector = np.zeros(n_components)          
#         for i, e_i in enumerate(principal_components):
#             projection = np.dot(self.embedding_vector, e_i)
#             self.relational_vector[i] = projection
        
#     def decay_relational_vector(self, decay_strength: int):
#         self.relational_vector = self.relational_vector * self.decay_constant**decay_strength   ##decay_strength depends on number of new chunks added


# class PCADataChunks(pydantic.BaseModel):
#     target_explained_variance: Optional[float] = 0.9   #higher is more compute generally
#     chunks: List[Chunk] = [] 
    
#     n_new_chunks: int = 0   #counter for new chunks
#     num_components: int = 0
#     principal_components: np.ndarray = None
#     dataset_explained_variance: float = 0
   
#     class Config:
#         arbitrary_types_allowed = True

#     def calculate_principal_components(self):
#         V = np.vstack([chunk.embedding_vector for chunk in self.chunks]) #stack of chunk embeddings
#         if self.dataset_explained_variance < self.target_explained_variance:
#             pca = PCA(n_components=min(V.shape))
#             pca.fit(V)
#             self.num_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= self.target_explained_variance) + 1
#             self.principal_components = pca.components_[:self.num_components]
#             self.dataset_explained_variance = np.sum(pca.explained_variance_ratio_[:self.num_components])
#         else:
#             pca = PCA(n_components=self.num_components)
#             pca.fit(V)
#             self.dataset_explained_variance = np.sum(pca.explained_variance_ratio_)
#             self.principal_components = pca.components_


#     def compute_relational_vectors(self):
#         with ThreadPoolExecutor(max_workers=min(8, len(self.chunks))) as executor:       
#             for chunk in self.chunks:
#                 executor.submit(chunk.compute_relational_vector, self.principal_components)
#             executor.shutdown(wait=True)
    
#     def calculate_recompute_threshold(self) -> int:
#         N = len(self.chunks)

#         n = self.num_components if self.num_components else 1  
#         if N == 0:
#             return 1  # No chunks, no need for recomputation
#         k = 0.1  # You can adjust this constant to control the decay rate
#         recompute_threshold = round((0.2 + (0.8 / (1 + k * (n / N)))) * N)
#         return recompute_threshold
    
#     def add_chunks(self, chunks: list[Chunk]):

#         recompute_threshold = self.calculate_recompute_threshold()  
        
#         if len(self.chunks) == 0 or len(chunks)+self.n_new_chunks >= recompute_threshold:
#             self.chunks.extend(chunks)
#             self.calculate_principal_components()
#             self.compute_relational_vectors()
#             self.n_new_chunks = 0
        
#         else: 
#             with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:       
#                 for chunk in chunks:
#                     executor.submit(chunk.compute_relational_vector(), self.principal_components)
#                 executor.shutdown(wait=True)
#             self.chunks.extend(chunks)
#             self.n_new_chunks += len(chunks)

        
#     def query(self, chunks: list[Chunk]):
#         ## compute relational vectors for new chunks
#         ## compute cosine similarity of relational vectors between new chunks and existing chunks
#         pass