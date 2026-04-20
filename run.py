import sys, numpy as np
from ane import model  

model = model(sys.argv[1])
# Only need (1, 64, 1, 1) from model.src_nchw (1, 64, 1, 1, 64, 64) 
a = np.full(model.src_nchw[0][:4], 2, dtype=np.float16)
b = np.full(model.src_nchw[1][:4], 2, dtype=np.float16)

out = model.predict([a, b])[0]
print(out.shape, out.dtype)
print(out)