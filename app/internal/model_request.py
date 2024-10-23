from app import settings
import openai
from openai.error import OpenAIError, Timeout, ServiceUnavailableError, RateLimitError, APIError
from typing import Optional, Literal
from app.api_clients import mclapsrl
import numpy as np
import time
import traceback
import warnings
import logging


rate_limiter = mclapsrl.mclapsrlClient()
openai.api_key = settings.OPEN_AI_KEY
logger = logging.getLogger(__name__)


def model_response(query_message: str, assistant_message: str, system_message: str , model_name: str, json_mode: bool, temperature: float, response_length: int ) -> str:
    
    
    #json mode
    if json_mode == True:
        response_type = {"type": "json_object"}
    else:
        response_type = {"type": "text"}
    
    retries = 3

    while retries > 0:

        max_sleep_time = 70
        while rate_limiter.model_status(model_name) == False:
            if max_sleep_time >= 0: 
                time.sleep(10)
                max_sleep_time -= 10
                continue
            else: 
                logger.error(f"model counter {model_name}is fucked, resetting")
                rate_limiter.reinitialize_counters()
                break

        try:
            completion = openai.ChatCompletion.create(
                    model = model_name,
                    response_format = response_type,
                    messages=[
                            {"role": "system", "content": system_message},
                            {"role": "assistant", "content": assistant_message},
                            {"role": "user", "content": query_message},
                        ],
                    temperature = temperature,
                    max_tokens = response_length,
                    n=1  
                    )
            rate_limiter.new_response(completion)
            completion_string: str = completion.choices[0].message.content
            return completion_string
        except (OpenAIError, Timeout, ServiceUnavailableError, APIError) as e:
            logger.error(f"server returned an error while processing query: {query_message}. {e}")           
            rate_limiter.model_break(model_name, 10)
            retries -= 1
            continue
        except RateLimitError as e:
            logger.error(f"rate limit exceeded while processing query: {query_message}. {e}")
            rate_limiter.model_break(model_name, 60)
            retries -= 1
            continue
        except Exception as e:
            logger.error(f"an exception occurred while processing query: {query_message}. {e}")
            rate_limiter.model_break(model_name, 10)
            retries -= 1
            continue
            
    else:
        logger.error(f"a default response was returned for model query: {query_message}")
        default_response = "\n".join([system_message, assistant_message, query_message])
        return default_response
    


