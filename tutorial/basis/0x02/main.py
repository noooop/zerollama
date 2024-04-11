
import numpy as np

tenser = np.random.rand(3, 224, 224).astype(np.float32)

buffer = tenser.tobytes()

tenser2 = np.frombuffer(buffer, dtype=np.float32)

print(tenser2.shape)

tenser3 = tenser2.reshape((3, 224, 224))
print(tenser3.shape)
