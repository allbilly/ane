# Apple ANE running on Asahi

Tested on Linux fedora 6.14.8-400.asahi.fc42.aarch64+16k

## Generate hwx

sum.hwx is from https://github.com/tinygrad/tinygrad/tree/v0.10.3/extra/accel/ane/ops

mul.hwx if from MacOS Monterey VM (v12.4 21F79) running on M4 macbook air 

```
import numpy as np
import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder

# KxK GEMM with bias
K = 64

input_features = [('image', datatypes.Array(K))]
input_features2 = [('image2', datatypes.Array(K))]
output_features = [('probs', datatypes.Array(K))]

weights = np.zeros((K, K)) + 3
bias = np.ones(K)

builder = NeuralNetworkBuilder(input_features+input_features2, output_features)

#builder.add_inner_product(name='ip_layer', W=weights, b=None, input_channels=K, output_channels=K, has_bias=False, input_name='image', output_name='med')
#builder.add_inner_product(name='ip_layer_2', W=weights, b=None, input_channels=3, output_channels=3, has_bias=False, input_name='med', output_name='probs')
builder.add_elementwise(name='element', input_names=['image', 'image2'], output_name='probs', mode='MULTIPLY')
#builder.add_bias(name='bias', b=bias, input_name='med', output_name='probs', shape_bias=(K,))
#builder.add_activation(name='act_layer', non_linearity='SIGMOID', input_name='med', output_name='probs')

# compile the spec
mlmodel = ct.models.MLModel(builder.spec)

# trigger the ANE!
out = mlmodel.predict({"image": np.zeros(K, dtype=np.float32)+1, "image2": np.zeros(K, dtype=np.float32)+2})
print(out)
mlmodel.save('test.mlmodel')
```

```bash
./coreml2hwx /Users/mac/tinygrad/extra/accel/ane/1_build/mul.mlmodel
cp /tmp/hwx_output/mul/model.hwx ./mul.hwx
```

## Run hwx

```bash
compile https://github.com/eiln/ane/blob/main/bindings/python/dylib/Makefile and the cp libane_python.so to /usr/lib/

uv venv --python=3.11 && source .venv/bin/activate
uv pip install https://github.com/eiln/anecc.git#subdirectory=anecc https://github.com/eiln/ane.git#subdirectory=bindings/python/python
anecc sum.hwx -o sum.ane
python run.py ./sum.ane 
```

## Parse hwx

python hwx_parsing.py sum.hwx -s 4

# Add vs Mul

| Offset | sum.cmd | mul.cmd | Register | Field | Description |
|--------|---------|---------|---------|-------|-------------|
| 0x20   | 0x66    | 0xa5    | Common @ 0x20 | pad2 | Unused padding field |
| 0x218  | 0x00    | 0x10    | TileDMA Src | RowStride | Operand stride config |
| 0x228  | 0x00    | 0x10    | TileDMA Src | PlaneStride | Operand stride config |
| 0x220  | 0x00    | 0x04    | L2 ResultBase | - | Result buffer address |
| 0x270  | 0x00    | 0x30    | NE MACCfg | OpMode | **Operation: ADD → MUL** |

**Main difference**: At offset 0x270, the NE MACCfg OpMode changes from `0x00` (ADD/sum) to `0x30` (MUL/mul). This switches the operation from accumulation to multiplication.


# Reference
https://github.com/freedomtan/coreml_to_ane_hwx
https://github.com/tinygrad/tinygrad/tree/v0.10.3/extra/accel/ane/