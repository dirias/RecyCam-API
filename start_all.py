from utils import img, utils
import subprocess

import logging 

# Create a custom logger and set its level
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# Create a formatter with color codes
formatter = logging.Formatter("\x1b[31m%(levelname)s:\x1b[0m %(message)s")

# Create a console handler and set the formatter
ch = logging.StreamHandler()
ch.setFormatter(formatter)

# Add the handler to the logger
log.addHandler(ch)

PORT = '8080'
SERVER_IP = '192.168.0.6'

def start_app():

    log.warning('Resizing images to create the working table')

    img.main()

    log.warning('Creating the CSV with the resized images information for the prediction')

    utils.create_csv()

    try:
        subprocess.run(["python3", "cnn.py"], check=True)
        log.warning('Finished running the model.')
        log.warning('Starting server')
        command = ["uvicorn", "app.app:app", "--host", SERVER_IP, "--port", str(PORT)]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f'Error running the Python script: {e}')

start_app()

