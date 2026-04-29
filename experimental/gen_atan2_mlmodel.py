import torch
import torch.nn as nn
import coremltools as ct

class Atan2(nn.Module):
    def __init__(self):
        super(Atan2, self).__init__()
    def forward(self, x, y):
        x = torch.atan2(x, y)
        return x

model = Atan2().eval()

input = [torch.rand(1024, 2048), torch.rand(1024, 2048)]
trace = torch.jit.trace(model, input)
mlmodel = ct.convert(trace, convert_to="neuralnetwork",
                    inputs=[ct.TensorType(name="x", shape=input[0].shape),
                            ct.TensorType(name="y", shape=input[1].shape)])
mlmodel.save('atan2.mlmodel')
