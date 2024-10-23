import requests
from app import settings
from app.data_models import OpenAIModels
import time

openai_models = OpenAIModels()


#parses the response body from openai generator object to a dictionary
def parse_response(response) -> dict:
    try:
        model = response.model
    except AttributeError:
        model = None
    try:
        input_tokens = response.usage.prompt_tokens
    except AttributeError:
        input_tokens = 0
    try:
        output_tokens = response.usage.completion_tokens
    except AttributeError:
        output_tokens = 0
    try:    
        total_tokens = response.usage.total_tokens
    except AttributeError:
        total_tokens = 0
    
    parsed_json = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }
    return parsed_json


class mclapsrlClient:
    def __init__(self):
        self.base_url = settings.MCLAPSRL_API
        self.headers = {'accept': 'application/json', 'Content-Type': 'application/json'}

    #redundant not used in here
    def get_counter_status(self, model: str) -> dict:
        response = requests.get(f"{self.base_url}/counter_status", params={'model': model})
        return response.json()
    def create_counter(self, model: str) -> bool:
        attempts = 10
        while attempts > 0:
            try:
                response = (requests.post(f"{self.base_url}/create_counter", headers = self.headers, json = {"model":model})).json()  # returns true false for whether counter created
                if response == True:
                    return True
                else:
                    print(f"Counter creation failed for model {model}, retrying  ({attempts} attempts remaining).")
                    attempts -= 1
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                print(f"Error in mclapsrl connection: {e}, retrying  ({attempts} attempts remaining).")
                attempts -= 1
        return False
    


    def check_service_status(self) -> dict:
        response_root = (requests.get(f"{self.base_url}/")).json()
        response_redis = (requests.get(f"{self.base_url}/redis_connection")).json()
        mclapsrl_status = {
            "root_status": response_root,
            "redis_status": response_redis
        }
        return mclapsrl_status

    def reinitialize_counters(self) -> bool:
        response = requests.get(f"{self.base_url}/clear_cache", headers = self.headers)
        return response.json()
    #post
    def new_response(self, response) -> bool:
        #openai generator object parsed to dictionary
        response_body = parse_response(response)

        attempts = 10
        while attempts > 0:
            try:
                response = requests.post(f"{self.base_url}/new_response", headers = self.headers, json = response_body)
                if response.json() == True:
                    return True
                else:
                    print(f"Response logging failed, retrying  ({attempts} attempts remaining).")
                    attempts -= 1
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"Error in mclapsrl connection: {e}, retrying  ({attempts} attempts remaining).")
                attempts -= 1
            except requests.exceptions.RequestException as e:
                print(f"Request exception: {e}")
                break #breaks loop if request exception occurs

        return False
    
    

    def model_break(self, model: str, break_time: int) -> bool:
        attempts = 10
        while attempts > 0:
            try:
                response = (requests.post(f"{self.base_url}/model_break", headers = self.headers, json = {'model':model, 'break_time': break_time})).json()  # returns true false for whether counter updated
                if response == True:
                    return True
                else:
                    print(f"Counter update failed for model {model}, retrying  ({attempts} attempts remaining).")
                    attempts -= 1
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                print(f"Error in mclapsrl connection: {e}, retrying  ({attempts} attempts remaining).")
                attempts -= 1
                time.sleep(2)
        return False


    #returns boolean status of model, or false if client is down, so if client is down the simulation halts indefinitely
    def model_status(self, model: str) -> bool:      
        attempts = 10
        while attempts > 0:
            try:
                response = requests.get(f"{self.base_url}/model_status", params={'model': model})
                return response.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"Error in mclapsrl connection: {e}, retrying ({attempts} attempts remaining).")
                time.sleep(5)
                attempts -= 1
        
        return False
    




