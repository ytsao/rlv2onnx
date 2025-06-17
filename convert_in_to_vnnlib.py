import os
import numpy as np
import argparse


def convert_in_to_vnnlib(
    in_file, vnnlib_file, epsilon=0.01, input_size=784, output_size=10, clip_bounds=True
):
    """
    Convert a property in .in format to .vnnlib format
    Each input line contains a single value, and bounds are created using epsilon (L_∞ norm)

    Args:
        in_file (str): Path to the .in file
        vnnlib_file (str): Path to save the .vnnlib file
        epsilon (float): Epsilon value for L_infinity norm
        input_size (int): Size of the input vector
        output_size (int): Size of the output vector
        clip_bounds (bool): Whether to clip bounds to [0,1] range
    """
    # Read the .in file
    input_values = []
    output_constraints = []

    with open(in_file, "r") as f:
        # Read input values (first input_size lines)
        for i in range(input_size):
            line = f.readline().strip()
            if not line:
                break

            # Check if the line has one or two values
            parts = line.split()
            if len(parts) == 1:
                # Single value - apply epsilon
                value = float(parts[0])
                lower = value - epsilon
                upper = value + epsilon
                if clip_bounds:
                    lower = max(0.0, lower)
                    upper = min(1.0, upper)
                input_values.append([lower, upper])
            else:
                # Already has bounds
                bounds = [float(x) for x in parts]
                input_values.append(bounds)

        # Read output constraints (remaining lines)
        line = f.readline()
        while line:
            data = [float(x) for x in line.strip().split(" ")]
            if len(data) != output_size + 1:
                break
            output_constraints.append(data)
            line = f.readline()

    # Write the .vnnlib file
    with open(vnnlib_file, "w") as f:
        # Declare input and output variables
        for i in range(input_size):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(output_size):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Input constraints
        f.write("\n; Input constraints\n")
        for i, (lb, ub) in enumerate(input_values):
            f.write(f"(assert (<= X_{i} {ub}))\n")
            f.write(f"(assert (>= X_{i} {lb}))\n")

        # Output constraints
        f.write("\n; Output constraints\n")
        # For robustness properties, we typically want to check if any output class
        # other than the correct class has a higher value
        # Format: (assert (or (and (>= Y_0 Y_c)) (and (>= Y_1 Y_c)) ...))

        # Group constraints by their structure
        # Assuming constraints represent: "verify that output class X is NOT the highest output"
        target_classes = set()
        for constraint in output_constraints:
            # Find the negative entry which usually indicates the target class
            weights = constraint[:-1]
            negatives = [i for i, w in enumerate(weights) if w < 0]
            if len(negatives) == 1:
                target_classes.add(negatives[0])

        for target_class in target_classes:
            # Start a new disjunction for each target class
            f.write(f"(assert (or\n")

            # For each possible output class
            for i in range(output_size):
                if i != target_class:
                    # Assert that this class has a higher value than target_class
                    f.write(f"    (and (>= Y_{i} Y_{target_class}))\n")

            f.write("))\n")

        # If we don't have standard robustness properties, fall back to directly translating constraints
        if not target_classes:
            for constraint in output_constraints:
                weights = constraint[:-1]  # All but the last element
                bias = constraint[-1]  # Last element

                # Build the expression: w_0*y_0 + w_1*y_1 + ... + w_n*y_n + bias >= 0
                terms = []
                for i, w in enumerate(weights):
                    if w != 0:
                        terms.append(f"{w} Y_{i}")

                if terms:
                    expression = " ".join(terms)
                    if len(terms) > 1:
                        expression = f"(+ {expression})"

                    # Add the bias if necessary
                    if bias != 0:
                        if len(terms) > 0:
                            expression = f"(+ {expression} {bias})"
                        else:
                            expression = str(bias)

                    f.write(f"(assert (>= {expression} 0))\n")


def batch_convert(
    in_dir, vnnlib_dir, epsilon=0.01, input_size=784, output_size=10, clip_bounds=True
):
    """
    Convert all .in files in a directory to .vnnlib format

    Args:
        in_dir (str): Directory containing .in files
        vnnlib_dir (str): Directory to save .vnnlib files
        epsilon (float): Epsilon value for L_infinity norm
        input_size (int): Size of the input vector
        output_size (int): Size of the output vector
        clip_bounds (bool): Whether to clip bounds to [0,1] range
    """
    os.makedirs(vnnlib_dir, exist_ok=True)

    for filename in os.listdir(in_dir):
        if filename.endswith(".in") and filename != "smallexample.in":
            in_file = os.path.join(in_dir, filename)
            vnnlib_file = os.path.join(vnnlib_dir, filename.replace(".in", ".vnnlib"))

            print(f"Converting {in_file} to {vnnlib_file}")
            convert_in_to_vnnlib(
                in_file, vnnlib_file, epsilon, input_size, output_size, clip_bounds
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .in property files to .vnnlib format"
    )
    parser.add_argument("--in_file", type=str, help="Path to the .in file")
    parser.add_argument("--vnnlib_file", type=str, help="Path to save the .vnnlib file")
    parser.add_argument("--in_dir", type=str, help="Directory containing .in files")
    parser.add_argument(
        "--vnnlib_dir", type=str, help="Directory to save .vnnlib files"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01, help="Epsilon value for L_infinity norm"
    )
    parser.add_argument(
        "--input_size", type=int, default=784, help="Size of the input vector"
    )
    parser.add_argument(
        "--output_size", type=int, default=10, help="Size of the output vector"
    )
    parser.add_argument(
        "--no_clip",
        action="store_false",
        dest="clip_bounds",
        help="Disable clipping bounds to [0,1] range",
    )

    args = parser.parse_args()

    if args.in_file and args.vnnlib_file:
        # Single file conversion
        convert_in_to_vnnlib(
            args.in_file,
            args.vnnlib_file,
            args.epsilon,
            args.input_size,
            args.output_size,
            args.clip_bounds,
        )
    elif args.in_dir and args.vnnlib_dir:
        # Batch conversion
        batch_convert(
            args.in_dir,
            args.vnnlib_dir,
            args.epsilon,
            args.input_size,
            args.output_size,
            args.clip_bounds,
        )
    else:
        parser.print_help()
