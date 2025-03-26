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



def upload_model_to_hf(path_in_repo, model_path="temp/iter_model.pth.tar"):
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
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
            
        logger.info(f"Uploading model from {model_path} to {repo_id}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type
        )
        logger.info("✅ Model uploaded to Hugging Face successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading model to Hugging Face: {str(e)}")
        return False
    


def _get_new_model(id, logger, model_path="temp/iter_model.pth.tar"):
    """
    Download a model file from Hugging Face Hub with error handling and logging.
    
    Args:
        id: Identifier for logging
        model_path: Path where to save the downloaded model
        repo_id: Hugging Face repository ID
        path_in_repo: Path to the file in the repository
        repo_type: Repository type (model, dataset, etc.)
    
    Returns:
        bool: True if download successful, False otherwise
    """
    
    repo_id = "samonuall/alpha-poker"
    repo_type = "model"
    
    try:
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
        
        # Download the model
        new_model_path =  f"samonuall/alpha-poker{id+1}.pth.tar"
        logger.info(f"Downloading model from {repo_id}/{new_model_path} to {model_path}")
        api = HfApi()
        hf_hub_download(
            repo_id=repo_id,
            filename=new_model_path,
            repo_type=repo_type,
            local_dir=os.path.dirname(model_path)
        )
        
        # Rename downloaded file if needed
        downloaded_path = os.path.join(os.path.dirname(model_path), new_model_path)
        if downloaded_path != model_path and os.path.exists(downloaded_path):
            os.rename(downloaded_path, model_path)
            
        logger.info("✅ Model downloaded successfully!")
        
        # Delete the model from the repository
        old_model_path = f"samonuall/alpha-poker{id}.pth.tar"
        logger.info(f"Deleting model {old_model_path} from repository {repo_id}")
        api.delete_file(
            path_in_repo=old_model_path,
            repo_id=repo_id,
            repo_type=repo_type
        )
        logger.info("✅ Model deleted from repository successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in get_new_model: {str(e)}")
        return False
    

def get_new_model(id, model_path="temp/iter_model.pth.tar"):
    # Configure logger
    logger = logging.getLogger(f"get_new_model_{id}")
    coloredlogs.install(level='INFO', logger=logger, 
                        fmt='%(asctime)s %(levelname)s %(message)s')
    
    
    logger.info(f"Waiting for new model with id {id}")
    while not _get_new_model(id, logger, model_path=model_path):
        logger.info("Retrying in 10 seconds...")
        time.sleep(10)
    

if __name__ == "__main__":
    upload_model_to_hf("temp/thing.npy")
    get_new_model(0)