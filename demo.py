import torch
import torchvision
import trtorch

# Get a model
model = torchvision.models.alexnet(pretrained=False).eval().cuda()

# Create some example data
data = torch.randn((1, 3, 224, 224)).to("cuda")

# Trace the module with example data
traced_model = torch.jit.trace(model, [data])

# Compile module
compiled_trt_model = trtorch.compile(traced_model, {
    "input_shapes": [data.shape],
    "op_precision": torch.half, # Run in FP16
})

results = compiled_trt_model(data.half())

print(compiled_trt_model.graph)


