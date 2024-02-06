import onnx

# Load the ONNX model
model = onnx.load('eval_model.onnx')

# Names from your model
sigmoid_node_name = '/roi_heads/mask_head/Sigmoid'
sigmoid_output_name = '/roi_heads/mask_head/Sigmoid_output_0'
final_output_name = 'pred_masks'
split_node_name = '/roi_heads/mask_head/Split'

# Remove the Split node and change the output name of the Sigmoid node
nodes_to_remove = []
for node in model.graph.node:
    if node.name == split_node_name:
        nodes_to_remove.append(node)
    if node.name == sigmoid_node_name:
        # Change the output name of the sigmoid node to 'pred_masks'
        node.output[0] = final_output_name

# Remove nodes that are marked for removal (Split node)
for node in nodes_to_remove:
    model.graph.node.remove(node)

# Additionally, update the name in the graph outputs if necessary
for output in model.graph.output:
    if output.name == sigmoid_output_name:
        output.name = final_output_name

# Save the modified model
onnx.save(model, 'modified_model.onnx')
