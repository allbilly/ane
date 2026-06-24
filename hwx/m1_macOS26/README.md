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
ProductVersion:         26.5.1
BuildVersion:           25F80

$ uv pip freeze
numpy==2.4.6
coremltools==9.0