# rlv2onnx

This implementation handles the basic structure of RLV files as interpreted by your load_rlv method.

The converter properly represents the affine layers (with weights and biases) and ReLU activation functions in ONNX format.

If your RLV files contain more complex structures not covered by this implementation, you might need to extend the converter.

After conversion, you may want to verify that the ONNX model produces the same outputs as your original RLV model for the same inputs.

## Dependencies

```bash
pip install onnx numpy cvxpy
```

## Usage

```bash
git clone --recursive git@github.com:ytsao/rlv2onnx.git
python convert_rlv_to_onnx.py --rlv /path/to/network.rlv --onnx /path/to/output.onnx
```
