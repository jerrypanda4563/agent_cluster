import app.settings as settings

from pymongo.errors import PyMongoError
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


mongo=MongoClient(settings.MONGO_URI, server_api=ServerApi('1'))
database=mongo["simulations"]

def db_connection_test() -> bool:
    try:
        mongo.admin.command('ping')
        return True
    except Exception as e:
        print(f'Mongo connection failed: {e}')
        return False
