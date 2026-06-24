# Steps
```
python gen_mlmodel.py test.mlmodel
git clone https://github.com/freedomtan/coreml_to_ane_hwx && cd coreml_to_ane_hwx && make && mv ./coreml2hwx ../ && cd ../
./coreml2hwx ./test.mlmodel
cp /tmp/hwx_output/test/model.hwx ./mul.hwx
```

# ENV
$ sw_vers
ProductName:            macOS
ProductVersion:         15.4.1
BuildVersion:           24E263

$ uv pip freeze
numpy==2.2.6
coremltools==9.0