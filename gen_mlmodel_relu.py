import numpy as np
import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
import sys

K = 64

input_features = [('image', datatypes.Array(K))]
output_features = [('probs', datatypes.Array(K))]

builder = NeuralNetworkBuilder(input_features, output_features)

builder.add_activation(name='act_layer', non_linearity='RELU', input_name='image', output_name='probs')

# compile the spec
mlmodel = ct.models.MLModel(builder.spec)

# trigger the ANE!
out = mlmodel.predict({"image": np.zeros(K, dtype=np.float32)+1})
print(out)
mlmodel.save(sys.argv[1])