# rlv2onnx

This repository provides a simple way to convert `.rlv` file to `.onnx` format, and `.in` file to `.vnnlib` format.

## Usage

```bash
git clone --recursive git@github.com:ytsao/rlv2onnx.git
pip install onnx numpy cvxpy

# Convert rl to onnx
python convert_rlv_to_onnx.py --rlv /path/to/network.rlv --onnx /path/to/output.onnx

# Convert in to vnnlib
python convert_in_to_vnnlib.py --in /path/to/input.in --vnnlib /path/to/output.vnnlib
```
