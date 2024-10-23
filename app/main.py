import fastapi 
from app.internal.simulation import worker_function
from app.data_models import SimulationInstance
import logging
import multiprocessing


app = fastapi.FastAPI()




@app.get("/")
async def root():
    return {"worker_status": True}


@app.post("/simulate")
async def initialize_simulation_instance(instance_params: SimulationInstance):
    #unwrapping params
    sim_id = instance_params.simulation_id
    agent_params = instance_params.agent_params
    agent_profile = instance_params.agent_profile
    iterations = instance_params.iterations


    params_dict = agent_params.dict()
    profile_dict = agent_profile.dict()
    iterations_dict = iterations.dict()

    #starting the worker function
    p = multiprocessing.Process(target = worker_function, args=(sim_id, iterations_dict, profile_dict, params_dict))
    p.start()

    return 