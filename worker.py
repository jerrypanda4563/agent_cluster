from app.redis_config import cache
from rq import Worker, Queue, Connection
import logging

# listen = ['sim_requests']

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# redis_conn = cache

# if __name__ == '__main__':
#     logger.info('Worker starting...')
#     try:
#         with Connection(redis_conn):
#             worker = Worker(map(Queue, listen))
#             worker.work()
#     except Exception as e:
#         logger.error(f'Worker failed: {e}')
#         raise e
    

listen = ['sim_requests']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_conn = cache

class CustomWorker(Worker):
    DEFAULT_WORKER_TTL = 7200  # Set default timeout to 7200 seconds (2 hours)

if __name__ == '__main__':
    logger.info('Worker starting...')
    try:
        with Connection(redis_conn):
            worker = CustomWorker(map(Queue, listen))
            worker.work()
    except Exception as e:
        logger.error(f'Worker failed: {e}')
        raise e