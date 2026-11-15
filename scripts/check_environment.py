import sys
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("SciPy:", scipy.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Selected device:", device)

x = torch.randn(3, 3, device=device)

print("Tensor test:")
print(x)