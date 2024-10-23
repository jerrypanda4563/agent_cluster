from dotenv import load_dotenv
import os

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_API_KEY")
MONGO_URI= os.getenv("MONGO_URI")
REDIS_URI = os.getenv("REDIS_URI")
MCLAPSRL_API = os.getenv("MCLAPSRL_URI")
MCLAPS_DEMGEN_API = os.getenv("MCLAPS_DEMGEN_URI")
DEBUG = os.getenv("DEBUG") == 'True'
