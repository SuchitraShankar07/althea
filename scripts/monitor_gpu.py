import torch
import time
import psutil
from loguru import logger

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if not torch.cuda.is_available():
        logger.warning("No GPU available")
        return
    
    while True:
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            logger.info(f"GPU: {allocated:.2f}/{total:.2f}GB ({allocated/total*100:.1f}%) | "
                       f"Reserved: {reserved:.2f}GB | CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}%")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            break

if __name__ == "__main__":
    monitor_gpu_memory()