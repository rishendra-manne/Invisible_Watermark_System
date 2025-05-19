import logging
import os
from datetime import datetime

log_dir = os.path.join('artifacts', 'Logs')
log_file = f'{datetime.now().strftime("%m_%d_%y_%H_%M_%S")}.log'
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO

)
logging.info("logging has started")
