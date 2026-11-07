import os

class Config:
    # Reproducibility - Critical for production!
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2  
    CV_FOLDS = 5
    N_JOBS = -1  
    
    # Model directories - Organized project structure
    MODEL_DIR = "models"
    
    
    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    
config = Config()