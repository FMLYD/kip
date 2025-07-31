import torch
import numpy as np
from loguru import logger

test_tensor = torch.tensor([6.0, np.nan, 3.0])
nan_tensor = torch.isnan(test_tensor)
logger.info(f"the test tensor: {test_tensor.shape}, nan tensor: {nan_tensor.bool()}")
new_tensor = torch.masked_fill(test_tensor, mask=nan_tensor, value=0.0)
logger.info(f"the new tensor is: {new_tensor}")


