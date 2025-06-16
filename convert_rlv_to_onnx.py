import numpy as np
import onnx
from onnx import helper, TensorProto
import os
from dep.RefineRobustness.network import network, layer


def rlv_to_onnx(rlv_file, onnx_file):
    """
    Convert a network in .rlv format to ONNX format

    Args:
        rlv_file (str): Path to the .rlv file
        onnx_file (str): Path to save the .onnx file
    """
    # Load the network using the existing method
    net = network()
    net.load_rlv(rlv_file)

    # Initialize ONNX graph components
    nodes = []
    inputs = []
    outputs = []
    initializers = []

    # Create input with (1, 1, 28, 28) shape
    input_name = "input"
    inputs.append(
        helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 1, 28, 28])
    )

    # Add a Flatten node instead of Reshape
    flatten_output_name = "flattened_input"
    nodes.append(
        helper.make_node(
            "Flatten",
            inputs=[input_name],
            outputs=[flatten_output_name],
            name="flatten_input",
            axis=1,  # Keep batch dimension (1) and flatten the rest (1x28x28 = 784)
        )
    )

    last_output = flatten_output_name
    layer_idx = 0

    # Process each layer
    for i in range(len(net.layers)):
        current_layer = net.layers[i]

        if current_layer.layer_type == layer.INPUT_LAYER:
            # Skip input layer, already handled
            continue

        elif current_layer.layer_type == layer.AFFINE_LAYER:
            # Create weight tensor
            weight_name = f"weight_{layer_idx}"
            weight_data = np.zeros((current_layer.size, net.layers[i - 1].size))

            # Create bias tensor
            bias_name = f"bias_{layer_idx}"
            bias_data = np.zeros(current_layer.size)

            # Fill weight and bias values
            for j, neuron in enumerate(current_layer.neurons):
                weight_data[j, :] = neuron.weight
                bias_data[j] = neuron.bias

            # Add weight initializer
            initializers.append(
                helper.make_tensor(
                    name=weight_name,
                    data_type=TensorProto.FLOAT,
                    dims=weight_data.shape,
                    vals=weight_data.flatten().tolist(),
                )
            )

            # Add bias initializer
            initializers.append(
                helper.make_tensor(
                    name=bias_name,
                    data_type=TensorProto.FLOAT,
                    dims=bias_data.shape,
                    vals=bias_data.tolist(),
                )
            )

            # Create Gemm node (matrix multiplication + bias)
            gemm_output = f"gemm_output_{layer_idx}"
            nodes.append(
                helper.make_node(
                    "Gemm",
                    inputs=[last_output, weight_name, bias_name],
                    outputs=[gemm_output],
                    name=f"gemm_{layer_idx}",
                    alpha=1.0,
                    beta=1.0,
                    transB=1,  # Transpose weight matrix
                )
            )

            last_output = gemm_output
            layer_idx += 1

        elif current_layer.layer_type == layer.RELU_LAYER:
            # Create ReLU node
            relu_output = f"relu_output_{layer_idx}"
            nodes.append(
                helper.make_node(
                    "Relu",
                    inputs=[last_output],
                    outputs=[relu_output],
                    name=f"relu_{layer_idx}",
                )
            )

            last_output = relu_output
            layer_idx += 1

    # Set the final output
    outputs.append(
        helper.make_tensor_value_info(
            last_output, TensorProto.FLOAT, [1, net.layers[-1].size]
        )
    )

    # Create the graph
    graph = helper.make_graph(
        nodes=nodes,
        name=os.path.basename(rlv_file).replace(".rlv", ""),
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    # Create the model
    model = helper.make_model(graph, producer_name="RLV_to_ONNX_Converter")
    model.opset_import[0].version = 13  # Use ONNX opset 13

    # Check the model
    onnx.checker.check_model(model)

    # Save the model
    onnx.save(model, onnx_file)
    print(f"Successfully converted {rlv_file} to {onnx_file}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert RLV network to ONNX format")
    parser.add_argument("--rlv", type=str, required=True, help="Path to the .rlv file")
    parser.add_argument("--onnx", type=str, help="Path to save the .onnx file")

    args = parser.parse_args()

    if args.onnx is None:
        args.onnx = args.rlv.replace(".rlv", ".onnx")

    rlv_to_onnx(args.rlv, args.onnx)
