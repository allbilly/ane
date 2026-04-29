import coremltools as ct
from coremltools.converters.mil import Builder as mb

@mb.program(input_specs=[mb.TensorSpec(shape=(1024, 2048))])
def sqrt(x):
    x = mb.sqrt(x=x)
    return x

mlmodel = ct.convert(sqrt, convert_to="neuralnetwork")
mlmodel.save('sqrt.mlmodel')
