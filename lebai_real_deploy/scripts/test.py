import torch

policy = torch.jit.load(
    "checkpoints/lm3_reach_policy.pt",
    map_location="cpu",
)

print("Mean:", policy.mean)
print("Variance:", policy.variance)