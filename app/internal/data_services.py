import traceback
import csv

from typing import Dict, List
from tests.test import mongo_connection_test
import app.mongo_config as mongo_db
from fastapi import HTTPException

def load_simulation_json(sim_id: str) -> Dict:
    if mongo_connection_test():
        db = mongo_db.database
    else:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error connecting to database.")
        
    request_object_query = {"_id": sim_id}
    request_object: dict = db["requests"].find_one(request_object_query)
    result_ids: list[str] = request_object["result_ids"]

    result_object_queries = [{"_id": result_id} for result_id in result_ids]
    result_objects: list[dict] = [db["results"].find_one(result_object_query) for result_object_query in result_object_queries]

    return {
        "simulation_id": sim_id,
        "name": request_object["name"],
        "context": request_object["context"],
        "demographic_conditions": request_object["demographic_sampling_conditions"],
        "iterations": request_object["iterations"],
        "n_of_runs": request_object["n_of_runs"],
        "results": result_objects
    }




def load_simulation_csv(sim_id: str, file_path: str) -> str:
    data = load_simulation_json(sim_id)
    simulation_id: str = data["simulation_id"]
    simulation_name: str = data["name"]
    iteration_questions: list[dict] = data["iterations"]
    simulation_results: list[dict] = data["results"]

    file_name = f"{file_path}/{simulation_id}_{simulation_name}_results.csv"
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:

        column_names = []

        demographic_fields: list[str] = list(data["demographic_conditions"].keys())
        question_fields: list[str] = [iteration_question["question"] for iteration_question in iteration_questions]

        column_names.append("agent_id")
        column_names.extend(demographic_fields)
        column_names.append("persona")
        column_names.extend(question_fields)

        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()


        #result object schema
        # {"_id": self.simulator_id,
        # "request_id": self.request_id, 
        # "demographic": self.demographic, 
        # "persona": self.persona, 
        # "response_data": []}
        for result in simulation_results:
            result_dict: dict = {}
            agent_id_dict = {"agent_id": result["_id"]}
            result_dict.update(agent_id_dict)
            demographic_dict: dict = result["demographic"]
            result_dict.update(demographic_dict)
            persona_dict:dict = {"persona": result["persona"]}
            result_dict.update(persona_dict)
            response_dict: dict = {response["question"]:response["answer"] for response in result["response_data"]}
            result_dict.update(response_dict)
            writer.writerow(result_dict)


    return file_name
