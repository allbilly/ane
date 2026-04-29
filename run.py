import sys, numpy as np
from ane import model  

m = model(sys.argv[1])
a = np.full(m.src_nchw[0][:4], 3, dtype=np.float16)
if m.src_count > 1:
    b = np.full(m.src_nchw[1][:4], 2, dtype=np.float16)
    out = m.predict([a, b])[0]
else:
    out = m.predict([a])[0]
print(out.shape, out.dtype)
print(out[0])