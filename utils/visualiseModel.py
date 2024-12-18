import re
from graphviz import Digraph

def parse_log_file(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    encoder_layers = []
    decoder_layers = []
    channel_info = None
    input_shape = None
    layer_output_shapes = []  # Store all layer output shapes
    channel_output_shape = None  # Explicitly track the channel output shape

    for line in lines:
        if "Building Encoder Layer" in line:
            match = re.search(r"Config: (.*?), GDN: (.*?)$", line)
            if match:
                config = eval(match.group(1))  # Convert string to dict
                gdn = match.group(2)
                config["gdn"] = gdn
                encoder_layers.append(config)
        elif "Building Decoder Layer" in line:
            match = re.search(r"Config: (.*?), GDN: (.*?)$", line)
            if match:
                config = eval(match.group(1))  # Convert string to dict
                gdn = match.group(2)
                config["gdn"] = gdn
                decoder_layers.append(config)
        elif "Building Channel Layer" in line:
            channel_info = re.search(r"Type: (.*?), snrdB: (\d+)", line).groups()
        elif "Input shape" in line:
            input_shape = eval(line.split("=")[1].strip())
        elif "Channel output shape" in line:
            # Parse the specific channel output shape
            shape_match = re.search(r"Channel output shape = (\(.*?\))", line)
            if shape_match:
                channel_output_shape = eval(shape_match.group(1))  # Convert to tuple
        elif "layer output shape" in line or "output shape" in line:
            # Parse general layer output shapes
            shape_match = re.search(r"output shape: (\(.*?\))", line)
            if shape_match:
                shape = eval(shape_match.group(1))  # Convert to tuple
                layer_output_shapes.append(shape)

    return {
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "channel_info": channel_info,
        "input_shape": input_shape,
        "layer_output_shapes": layer_output_shapes,
        "channel_output_shape": channel_output_shape,
    }

def visualize_model(parsed_info, output_file="model_architecture"):
    graph = Digraph("Model Architecture", format="png")
    graph.attr(rankdir="TB")  # Top to Bottom layout

    # Add Input Node
    input_shape = parsed_info["input_shape"]
    graph.node("Input", f"Input\nShape: {input_shape}", shape="box")

    # Process Encoder Nodes
    current_shape_idx = 0
    for idx, layer in enumerate(parsed_info["encoder_layers"]):
        filters = layer["filters"]
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        gdn = layer["gdn"]

        # Use the actual output shape
        output_shape = parsed_info["layer_output_shapes"][current_shape_idx]
        current_shape_idx += 1

        label = (
            f"Encoder Layer {idx + 1}\n"
            f"Filters: {filters}\n"
            f"Kernel: {kernel_size}\n"
            f"Stride: {stride}\n"
            f"Output: {output_shape}\n"
            f"GDN: {gdn}"
        )
        graph.node(f"Encoder_{idx + 1}", label, shape="box")
        if idx == 0:
            graph.edge("Input", f"Encoder_{idx + 1}")
        else:
            graph.edge(f"Encoder_{idx}", f"Encoder_{idx + 1}")

    # Add Channel Node
    if parsed_info["channel_info"]:
        channel_type, snrdB = parsed_info["channel_info"]
        # Use the explicitly parsed channel output shape
        channel_output_shape = parsed_info["channel_output_shape"]
        label = f"Channel\nType: {channel_type}\nsnrdB: {snrdB}\nShape: {channel_output_shape}"
        graph.node("Channel", label, shape="ellipse")
        graph.edge(f"Encoder_{len(parsed_info['encoder_layers'])}", "Channel")

    # Process Decoder Nodes
    for idx, layer in enumerate(parsed_info["decoder_layers"]):
        filters = layer["filters"]
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        upsample_size = layer.get("upsample_size", None)
        gdn = layer["gdn"]

        # Use the actual output shape
        output_shape = parsed_info["layer_output_shapes"][current_shape_idx]
        current_shape_idx += 1

        label = (
            f"Decoder Layer {idx + 1}\n"
            f"Filters: {filters}\n"
            f"Kernel: {kernel_size}\n"
            f"Stride: {stride}\n"
            f"Upsample: {upsample_size}\n"
            f"Output: {output_shape}\n"
            f"GDN: {gdn}"
        )
        graph.node(f"Decoder_{idx + 1}", label, shape="box")
        if idx == 0:
            graph.edge("Channel", f"Decoder_{idx + 1}")
        else:
            graph.edge(f"Decoder_{idx}", f"Decoder_{idx + 1}")

    # Add Output Node
    output_shape = parsed_info["layer_output_shapes"][-1]
    graph.node("Output", f"Output\nShape: {output_shape}", shape="box")
    graph.edge(f"Decoder_{len(parsed_info['decoder_layers'])}", "Output")

    # Render the graph
    graph.render(output_file, view=True)

if __name__ == "__main__":
    log_file = "../logs/Testing_print_modelShape.txt"  # Path to your log file
    parsed_info = parse_log_file(log_file)
    visualize_model(parsed_info, output_file="../outputs/model_architecture")
