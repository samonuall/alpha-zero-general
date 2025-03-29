from huggingface_hub import HfApi
import os
from huggingface_hub import login
from huggingface_hub import hf_hub_download
import logging
import coloredlogs
import time

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]



def upload_file_to_hf(path):
    """
    Upload a model file to Hugging Face Hub with error handling and logging.
    
    Args:
        model_path: Path to the model file to upload
        repo_id: Hugging Face repository ID
        path_in_repo: Path where to store the file in the repository
        repo_type: Repository type (model, dataset, etc.)
    
    Returns:
        bool: True if upload successful, False otherwise
    """
    
    repo_id = "samonuall/alpha-poker"
    repo_type = "model"
    
    # Configure logger
    logger = logging.getLogger("upload_model")
    coloredlogs.install(level='INFO', logger=logger, 
                        fmt='%(asctime)s %(levelname)s %(message)s')
    
    try:
        # Login to Hugging Face Hub
        # Try to get token from environment variable first
        token = os.environ.get("HF_TOKEN")
        if token:
            logger.info("Logging in to Hugging Face using token from environment")
            login(token=token)
        else:
            # If not available in environment, you can login interactively
            logger.info("Token not found in environment. Attempting interactive login")
            login()
        
        # Check if model file exists
        if not os.path.exists(path):
            logger.error(f"Model file not found at {path}")
            return False
            
        logger.info(f"Uploading model from {path} to {repo_id}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path.split("/")[-1],
            repo_id=repo_id,
            repo_type=repo_type
        )
        logger.info("✅ Model uploaded to Hugging Face successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading model to Hugging Face: {str(e)}")
        return False
    


def _get_new_model(logger, model_path, hub_path):
    # Download new_model.pth.tar from hub and store in model_path
    # Returns true/false based on success
    
    repo_id = "samonuall/alpha-poker"
    repo_type = "model"
    
    # Login to Hugging Face Hub
    token = os.environ.get("HF_TOKEN")
    if token:
        logger.info("Logging in to Hugging Face using token from environment")
        login(token=token)
    else:
        logger.info("Token not found in environment. Attempting interactive login")
        login()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        # Download the model
        logger.info(f"Downloading model from {repo_id}/new_model to {model_path}")
        api = HfApi()
        hf_hub_download(
            repo_id=repo_id,
            filename=hub_path,
            repo_type=repo_type,
            local_dir=os.path.dirname(model_path)
        )
            
        logger.info("✅ Model downloaded successfully!")
    except Exception as e:
        logger.error(f"Error in get_new_model: {str(e)}")
        return False
    
    # Delete the model from the repository
    logger.info(f"Deleting model from repository {repo_id}")
    api.delete_file(
        path_in_repo=hub_path,
        repo_id=repo_id,
        repo_type=repo_type
    )
    logger.info("✅ Model deleted from repository successfully!")
    
    return True
    

def get_new_model(model_path, hub_path):
    # Configure logger
    logger = logging.getLogger(f"get_new_model")
    coloredlogs.install(level='INFO', logger=logger, 
                        fmt='%(asctime)s %(levelname)s %(message)s')
    
    
    logger.info(f"Waiting for new model")
    counter = 0
    while not _get_new_model(logger, model_path, hub_path):
        logger.info("Retrying in 10 seconds...")
        time.sleep(10)
        counter += 10
        if counter > 1800:
            logger.error("Failed to get new model after 30 minutes")
            return False

    return True
    

if __name__ == "__main__":
    upload_file_to_hf("new_model.pth.tar")