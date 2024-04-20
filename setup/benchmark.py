
import torch
import numpy as np
import zmq
import transformers
import accelerate
import optimum
import auto_gptq
import awq

print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("zmq:", zmq.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("auto-gptq:", auto_gptq.__version__)
print("autoawq:", awq.__version__)