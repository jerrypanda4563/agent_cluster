from app.internal import response_agent 
from app.data_models import SurveyModel, DemographicModel, AgentParameters
import traceback
from typing import Dict, List, Optional
from app.internal.prompt_payloads import initialization_prompt, Iterator



import app.settings as settings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import json
import openai.error
import time
import uuid
import logging

logger = logging.getLogger(__name__)


#when an instance is ran, return a response_data json object containing responses and demographic data
#this is the part that can be setup as a celery task, so json as input and output




class Simulator():
    #unwraps agent parameters and performs initialization of agent 
    def __init__(self, request_id: str, survey: Dict, demographic: Dict, agent_params: AgentParameters, retries: Optional[int] = 3):
        
        self.request_id = request_id
        self.simulator_id = str(uuid.uuid4())

        

        self.survey_context: list[str] = survey["context"]
        self.json_mode: bool = agent_params.json_mode
        
        self.iterator = Iterator(json_mode = agent_params.json_mode, iteration_questions = survey["questions"])
        self.demographic: Dict = demographic["demographic"]
        self.persona: Dict = demographic["persona"]

        self.simulator_instructions = initialization_prompt(self.demographic, self.persona)
        self.simulator_params = agent_params
        
        self.retry_policy = retries
        
    def initialize_database(self):
        mongo=MongoClient(settings.MONGO_URI, server_api=ServerApi('1'))
        db = mongo["simulations"]
        return db
        
    def simulate(self) -> None:
        
        database = self.initialize_database()
        
        simulator = response_agent.Agent(
            agent_id = self.simulator_id,
            instruction = self.simulator_instructions,
            params = self.simulator_params
        )

        database["results"].insert_one({"_id": self.simulator_id,
                                  "request_id": self.request_id, 
                                  "demographic": self.demographic, 
                                  "persona": self.persona, 
                                  "response_data": []})
        result_object_query = {"_id": self.simulator_id}
        request_object_query = {"_id": self.request_id}

        for context in self.survey_context:
            simulator.inject_memory(context)
        
        #iterating though generator object
        for _ in range(self.iterator.n_of_iter):
            current_iteration = self.iterator.iter()
            schema = self.iterator.iterations[_]
            for i in range(self.retry_policy):
                try: 
                    result = simulator.chat(current_iteration)
                    if self.json_mode:
                        try:
                            response_json = json.loads(result)
                            answer = response_json["answer"]
                            schema["answer"] = answer
                        except Exception as e: 
                            print(f"Warning: JSON error in simulation run for simulator {self.simulator_id}: {e}")
                            schema["answer"] = result
                    else:
                        schema["answer"] = result
                    break

                except (openai.error.ServiceUnavailableError, openai.error.Timeout, openai.error.RateLimitError) as e:
                    logger.error(f'OpenAI error in simulation run for simulator {self.simulator_id} (Attempt {i + 1}): for question {_+1}. {e}')
            else:
                logger.error(f"Maximum retries reached for question: {json.dumps(schema)}. Skipping to next question.")
                schema["answer"] = None
                break
            #pushing to result object based on json mode
            database["results"].update_one(result_object_query, {"$push": {"response_data": schema}})
            database["requests"].update_one(request_object_query, {"$inc": {"completed_timesteps": 1}})
            

        #updating request object once all iterations are completed
        database["requests"].update_one(request_object_query, {"$inc":{"completed_runs": 1}})
        database["requests"].update_one(request_object_query, {"$push": {"result_ids": self.simulator_id}})
            

        
        
async def worker_function(sim_id: str, iterations: dict, agent_profile: dict, agent_params: dict) -> str:
    instance =  Simulator(
        request_id = sim_id,
        survey = iterations,
        demographic = agent_profile,
        agent_params = agent_params
    )
    instance_id = instance.simulator_id
    instance.simulate()
    return instance_id

        

        







    



