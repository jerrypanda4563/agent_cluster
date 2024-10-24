import fastapi 
from app.internal.simulation import Simulator
from app.data_models import SimulationInstance
import logging
import multiprocessing
from typing import Optional
from app.mongo_config import database


app = fastapi.FastAPI()
db = database["results"]
logger = logging.getLogger(__name__)



@app.get("/")
async def root():
    return {"worker_status": True}


@app.post("/simulate")
async def initialize_simulation_instance(instance_params: SimulationInstance):
    #unwrapping params
    sim_id = instance_params.simulation_id
    agent_params = instance_params.agent_params.dict()
    agent_profile = instance_params.agent_profile.dict()
    iterations = instance_params.iterations.dict()

    #starting the worker function
    instance =  Simulator(
        request_id = sim_id,
        survey = iterations,
        demographic = agent_profile,
        agent_params = agent_params
    )
    instance_id = instance.simulator_id

    process = multiprocessing.Process(target = instance.simulate())
    process.start()
    return instance_id


@app.get("/instance/result")
async def instance_result(instance_id: str):
    result_query = {"_id": instance_id}
    result_object = db.find_one(result_query)
    return result_object

    

# @app.post("/kill")
# async def kill_simulation_instance(instance_id: Optional[str] = None):
#     if instance_id is None:





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
