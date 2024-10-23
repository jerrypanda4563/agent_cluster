from app import settings
import openai
from openai import Embedding
from openai.error import OpenAIError, Timeout, ServiceUnavailableError, RateLimitError, APIError
from typing import Optional
from app.api_clients import mclapsrl
import numpy as np
import time
import traceback
import warnings
import logging

rate_limiter = mclapsrl.mclapsrlClient()
logger = logging.getLogger(__name__)

openai.api_key = settings.OPEN_AI_KEY

def embed(text: str, embedding_model: Optional[str] = "text-embedding-3-small", dimension: Optional[int] = 512) -> np.ndarray:
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
            

    retries = 5
    while retries > 0:

        max_sleep_time = 70
        while rate_limiter.model_status(embedding_model) == False:
            if max_sleep_time >= 0: 
                time.sleep(10)
                max_sleep_time -= 10
                continue
            else: 
                logger.error(f"model counter {embedding_model} is fucked, resetting counter")
                rate_limiter.reinitialize_counters()
                break

        try:
            
            response=Embedding.create(
                model = embedding_model,
                input=str(text)
                )
            rate_limiter.new_response(response)
            embedding = np.array(normalize_l2(response['data'][0]['embedding'][:dimension]))
            return embedding
        
        except (OpenAIError, Timeout, ServiceUnavailableError, APIError) as e:
            logger.error(f"server returned an error while embedding the text: {text}. {e}")
            rate_limiter.model_break(embedding_model, 10)
            retries -= 1
            continue
        except RateLimitError as e:
            logger.error(f"rate limit exceeded while embedding the text: {text}. {e}")
            rate_limiter.model_break(embedding_model, 60)
            retries -= 1
            continue
            
        except Exception as e:
            logger.error(f"an exception occurred while embedding the text: {text}. {e}")
            rate_limiter.model_break(embedding_model, 10)
            retries -= 1
            continue

    else: 
        logger.error(f"a zero vector was returned for the text: {text}")
        return np.zeros(dimension) 