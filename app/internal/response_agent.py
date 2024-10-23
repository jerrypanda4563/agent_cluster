from app.internal import agent_data
from app.internal.tokenizer import count_tokens
from app.api_clients.mclapsrl import mclapsrlClient
from app import settings
from app.internal import chunking
from app import mongo_config
import random

import numpy as np
import spacy
from typing import List, Optional
import time
import openai
from openai.error import OpenAIError, Timeout, ServiceUnavailableError, RateLimitError
from sklearn.metrics.pairwise import cosine_similarity as cs
from concurrent.futures import ThreadPoolExecutor
from app.data_models import AgentParameters
import gc
from app.internal.embedding_request import embed
from app.internal.model_request import model_response
import warnings
import logging

openai.api_key = settings.OPEN_AI_KEY


nlp = spacy.load("en_core_web_sm")      
rate_limiter = mclapsrlClient()
logger = logging.getLogger(__name__)


#### note to self: new data str is only added when injecting under satisfied conditions or when restructuring memory

class Agent:

    def __init__(self, agent_id:str, instruction:str, params: AgentParameters):

        self.agent_id = agent_id #generated within simulation instance

        self.lt_memory = agent_data.AgentData(
            memory_limit = params.memory_limit, 
            chunk_size = round(params.chunk_size/(params.reconstruction_top_n + 1)), 
            sampling_top_n = params.sampling_top_n, 
            reconstruction_top_n = params.reconstruction_top_n, 
            reconstruction_trigger_factor = params.reconstruction_factor,
            embedding_dim = params.embedding_dimension,
            memory_loss_factor = params.memory_loss_factor
            )
        
        self.st_memory: list[str] = []
        self.instruction:str = instruction
        self.lt_memory_chunk_size = round(params.chunk_size/(params.reconstruction_top_n + 1))

        self.st_memory_capacity: int = params.memory_context_length
        self.max_output_length: int = params.max_output_length
        self.lt_memory_trigger_length: int = params.lt_memory_trigger_length   # n. of tokens in string required to force trigger lt_memory storage rather than st_memory
        self.memory_chunk_size: int = params.chunk_size
        self.memory_limit: int = params.memory_limit
        self.embedding_dimension: int = params.embedding_dimension

        self.llm_model = params.agent_model
        self.embedding_model = params.embedding_model
        self.model_temperature = params.llm_temperature
        self.agent_temperature = params.agent_temperature    # for query randomness
        self.json_mode = params.json_mode
        self.existence_date = params.existance_date


        self.initialize_instance_object()

    def initialize_instance_object(self) -> None:
        instance_object = {
            "_id": self.agent_id,
            "qrr_iterations": [],           #qrr = query response reflection
            "st_memories":[]
        }
        db = mongo_config.database["agent_instances"]
        db.insert_one(instance_object)
        logger.info(f"Agent {self.agent_id} has been initialized")

    #add limiter
    def embed_string(self, string:str) -> np.ndarray:
        embedding = embed(string, embedding_model = self.embedding_model, dimension = self.embedding_dimension)
        return embedding

        
    def evaluator(self, string1:str, string2:str) -> float:
        string_1_embedding = self.embed_string(string1)
        string_2_embedding = self.embed_string(string2)
        k = cs(string_1_embedding.reshape(1,-1),string_2_embedding.reshape(1,-1))[0][0]
        if k: 
            return k
        else:
            return 1  # if fails, always stick with the current memory since max k is 1
        
    def st_memory_length(self) -> int:            
        return count_tokens(' '.join(self.st_memory))
    
    ################
    #can potentially be added with random generation of memory based on initialization data
    def random_memory(self) -> list[str]:
        
        trigger_value = random.random()
        if trigger_value < self.agent_temperature:
            return []
        else:
            if len(self.lt_memory.DataChunks) == 0:
                return []

            else:
                sampled_chunk = random.choice(self.lt_memory.DataChunks)
                related_chunk_indices = sorted(enumerate(sampled_chunk.conjugate_vector.tolist()), key=lambda x: x[1], reverse=True)[0:min(round(self.memory_chunk_size/self.lt_memory_chunk_size), len(self.lt_memory.DataChunks))]
                related_chunks = [self.lt_memory.DataChunks[index] for index, _ in related_chunk_indices]
                related_strings = [chunk.string for chunk in related_chunks]
                related_strings.extend([sampled_chunk.string])
                return related_strings
        
    ###################


    def construct_st_memory(self, query_str: str) -> None:
        try:
            #generate random memory based on agent temperature
            if random.random() < self.agent_temperature:
                random_memory = self.random_memory()
            else:
                random_memory = []
            

            current_memory = self.st_memory.copy()
            if len(self.st_memory) == 0:
                queried_memory = self.lt_memory.query(query_str)
            else:
                #1-agent_temp/2 chance of querying memory
                if random.random() > self.agent_temperature/2:
                    with ThreadPoolExecutor(max_workers=len(self.st_memory)) as executor:
                        similarity_scores = list(executor.map(lambda x: self.evaluator(query_str, x), self.st_memory))
                    k = np.average(similarity_scores) #mean similarity to query
                    queried_memory = self.lt_memory.query(query_str, k)
                else: 
                    #generates random memory along with current memory, ignores query
                    queried_memory = []
            
            self.st_memory = current_memory + queried_memory + random_memory

 
            if self.st_memory_length() > self.st_memory_capacity:
                self.restructure_memory(string = query_str)

        except Exception as e:
            logger.error(f"Error in constructing st memory: {e}")
            self.st_memory = self.st_memory
        
    
    #pop out strings with lowest similarity to query and add popped memory to lt_memory
    def restructure_memory(self, string:str) -> None:
        
        with ThreadPoolExecutor(max_workers=len(self.st_memory)) as executor:
            similarity_scores = list(executor.map(lambda x: self.evaluator(string, x), self.st_memory))

        new_lt_memory: list[str] = []
        while self.st_memory_length() > self.st_memory_capacity:
            index = similarity_scores.index(min(similarity_scores))
            forgotten_memory = self.st_memory.pop(index)
            similarity_scores.pop(index)
            new_lt_memory.append(forgotten_memory)
        new_lt_memory_joined = '\n'.join(new_lt_memory)
        self.lt_memory.add_data_str(new_lt_memory_joined)
            

    def update_instance(self, qrr: dict):
        retries = 3
        while retries > 0:
            try:
                db = mongo_config.database["agent_instances"]
                db.update_one({"_id": self.agent_id}, {"$push": {"qrr_iterations": qrr}})
                db.update_one({"_id": self.agent_id}, {"$push": {"st_memories": self.st_memory}})
                break
            except Exception as e:
                logger.error(f"Error in updating agent instance: {e} for agent {self.agent_id}")
                retries -= 1
                time.sleep(5)
                continue
        else:
            logger.error(f"Maximum retries reached for updating agent instance: {self.agent_id}")
            pass

    def llm_request(self, query: str):

        qrr_object = {
            "query": query,
            "response": " ",
            "reflection": " "
        }
        
        memory_prompt = "You recall the following pieces of information from memory: \n" + '\n'.join(self.st_memory) + "Consider these memories when responding to query if the memories are relevant." 
        system_prompt = self.instruction + f"The current timestamp is {self.existence_date}"
        response = model_response(
            query_message = query, 
            assistant_message = memory_prompt, 
            system_message = system_prompt,
            model_name = self.llm_model,
            json_mode = self.json_mode,
            temperature = self.model_temperature,
            response_length = self.max_output_length 
            )
        qrr_object["response"] = response
        

        reflection_prompt: str = f"Give a reason for why you responded the way you did to the most recent message."
        reflection_memory: str = f"Your responded: {response} to the most recent message: {query}. You also recall the following pieces of information:\n" + '\n'.join(self.st_memory)
        reflection_statement =  model_response(
            query_message = reflection_prompt, 
            assistant_message = reflection_memory, 
            system_message = system_prompt,
            model_name = self.llm_model,
            json_mode = False,
            temperature = self.model_temperature,
            response_length = round(self.max_output_length/2) 
            )
        qrr_object["reflection"] = reflection_statement
        

        self.update_instance(qrr_object)
    

        qr_pair = f"Query message:{query} \n Response:{response}"
        self.st_memory.append(qr_pair)
        reflection_chunked = chunking.chunk_string(reflection_statement, chunk_size = self.memory_chunk_size)
        self.st_memory.extend(reflection_chunked)

        return response
    

    
    
    
    #interacted functions
    def chat(self, query:str) -> str:
        self.construct_st_memory(query)   #changes system message 
        response: str = self.llm_request(query)
        if self.st_memory_length() > self.st_memory_capacity:
            self.restructure_memory(query)
        return response
    

    ####************ 
    def inject_memory(self, string: str) -> None:
        string_length: int = count_tokens(string)

        if string_length >= self.lt_memory_trigger_length:
            self.lt_memory.add_data_str(string)

        else:
            available_st_memory: int = self.st_memory_capacity - self.st_memory_length()
            if string_length > available_st_memory:
                self.lt_memory.add_data_str(string) ###directly addes the string chunked in AgentData
            else: 
                string_chunks: list[str] = chunking.chunk_string(string, chunk_size = self.memory_chunk_size) #chunk size to be larger for strings within st memory
                self.st_memory.extend(string_chunks)
                if self.st_memory_length() > self.st_memory_capacity:
                    self.restructure_memory(string)
                

                
            
        
        
    



