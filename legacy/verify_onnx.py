import onnx

# Load dan check model
model = onnx.load("phishing_rf.onnx")

# Verify model valid
onnx.checker.check_model(model)
print("✅ Model valid!")

# Print info
print(f"Opset version: {model.opset_import[0].version}")
print(f"IR version: {model.ir_version}")

# Print input/output info
for input in model.graph.input:
    print(f"Input: {input.name}, Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")

for output in model.graph.output:
    print(f"Output: {output.name}")