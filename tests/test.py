import sys
from app import mongo_config, redis_config
import openai
import app.settings as settings


def redis_connection_test() -> bool:
    redis_status=redis_config.cache_connection_test() #return boolean
    return redis_status

def mongo_connection_test() -> bool:
    mongo_status=mongo_config.db_connection_test() #return boolean
    return mongo_status
    

def openai_connection_test() -> bool:
    openai.api_key=settings.OPEN_AI_KEY
    try:
        # Make a test call to the API, for example, list available models
        response = openai.Engine.list()
        print("OpenAI Connection successful.")
        return True
    except Exception as e:
        print(f"An error occurred connecting to OpenAI: {e}")
        return False

# if __name__ == "__main__":
#     openai_status=openai_connection_test()
#     if not openai_status:
#         print("OpenAI connection failed")
#         sys.exit(1)
#     mongo_status=mongo_connection_test()
#     if not mongo_status:
#         print("MongoDB connection failed")
#         sys.exit(1)
#     redis_status=redis_connection_test()
#     if not redis_status:
#         print("Redis connection failed")
#         sys.exit(1)




