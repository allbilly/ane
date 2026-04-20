import sys, numpy as np
from ane import Model  

model = Model(sys.argv[1])
a = np.ones((1, 64, 1, 1), dtype=np.float16)
b = np.full((1, 64, 1, 1), 2, dtype=np.float16)

out = model.predict([a, b])[0]
print(out.shape, out.dtype)
print(out)