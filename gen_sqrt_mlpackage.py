import torch
import torch.nn as nn
import coremltools as ct

class Sqrt(nn.Module):
    def __init__(self):
        super(Sqrt, self).__init__()
    def forward(self, x):
        return torch.sqrt(x)

model = Sqrt().eval()
input = torch.rand(1024, 2048)
trace = torch.jit.trace(model, input)
mlmodel = ct.convert(trace, inputs=[ct.TensorType(name="x", shape=input.shape)])
mlmodel.save('sqrt.mlpackage')
