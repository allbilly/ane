import coremltools as ct
from coremltools.converters.mil import Builder as mb

@mb.program(input_specs=[mb.TensorSpec(shape=(1024, 2048)),
                         mb.TensorSpec(shape=(1024, 2048))])
def atan2(x, y):
    return mb.atan2(x=x, y=y)

mlmodel = ct.convert(atan2, convert_to="neuralnetwork")
mlmodel.save('atan2_mb.mlmodel')
