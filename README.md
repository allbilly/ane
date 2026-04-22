# Apple ANE running on Asahi

Tested on Linux fedora 6.14.8-400.asahi.fc42.aarch64+16k

## 1. Generate hwx

### option 1 on MacOS 12.4

sum.hwx is from https://github.com/tinygrad/tinygrad/tree/v0.10.3/extra/accel/ane/ops

mul.hwx if from MacOS Monterey VM (v12.4 21F79) running on M4 macbook air 

```bash
python gen_mlmodel.py
git clone https://github.com/freedomtan/coreml_to_ane_hwx && cd coreml_to_ane_hwx && make && mv ./coreml2hwx ../ && cd ../
./coreml2hwx ./test.mlmodel
cp /tmp/hwx_output/mul/model.hwx ./mul.hwx
```

more ops on https://github.com/eiln/ane-ex/blob/main/sources.md

### option 2 on Github action
Not working now
Note: sadly macos14 is the oldest macos version on gh action, which is not yet supported by anecc 
https://docs.github.com/en/actions/reference/runners/github-hosted-runners

Check the gihub action config at .github/workflows/ane-generation.yml
- go to branch "macos_buildhwx" and modify mode in builder.add_elementwise
- go to actions/runs/_runid_/ -> Artifacts -> download and unzip

## 2. Run hwx

```bash
compile https://github.com/eiln/ane/blob/main/bindings/python/dylib/Makefile and the cp libane_python.so to /usr/lib/

uv venv --python=3.11 && source .venv/bin/activate
uv pip install https://github.com/eiln/anecc.git#subdirectory=anecc https://github.com/eiln/ane.git#subdirectory=bindings/python/python
anecc hwx/sum.hwx -o hwx/sum.ane
python run.py ./hwx/sum.ane 
```

## 3. Parse hwx

python parse.py hwx/sum.hwx 

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