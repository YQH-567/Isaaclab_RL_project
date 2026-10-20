# import torch

# checkpoint_path = "//home/l/isaac_space_01/my_robot_rl/logs/skrl/reach_lebai_lm3/2026-10-18_23-25-12_ppo_torch/checkpoints/best_agent.pt"

# data = torch.load(
#     checkpoint_path,
#     map_location="cpu",
#     weights_only=True,
# )

# print("Checkpoint type:", type(data))

# if isinstance(data, dict):
#     print("Top-level keys:", list(data.keys()))
#     for key, value in data.items():
#         print(key, type(value))

# for module_name in ["policy", "state_preprocessor"]:
#     print(f"\n========== {module_name} ==========")

#     state_dict = data[module_name]

#     for key, value in state_dict.items():
#         if isinstance(value, torch.Tensor):
#             print(
#                 f"{key:45s}",
#                 f"shape={tuple(value.shape)}",
#                 f"dtype={value.dtype}",
#             )
#         else:
#             print(key, type(value))




import torch
import torch.nn as nn


class LM3Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.net_container = nn.Sequential(
            nn.Linear(25, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )

        self.policy_layer = nn.Linear(64, 6)

    def forward(self, normalized_obs):
        features = self.net_container(normalized_obs)
        return self.policy_layer(features)


checkpoint = torch.load(
    "/home/l/isaac_space_01/my_robot_rl/logs/skrl/reach_lebai_lm3/2026-10-18_23-25-12_ppo_torch/checkpoints/best_agent.pt",
    map_location="cpu",
    weights_only=True,
)

actor = LM3Actor()

# 只提取 Actor 所需参数
actor_state = {
    key: value
    for key, value in checkpoint["policy"].items()
    if key.startswith("net_container.")
    or key.startswith("policy_layer.")
}

actor.load_state_dict(actor_state, strict=True)
actor.eval()

# 示例：这里只输入已经完成标准化的 Observation
example_obs = torch.zeros(1, 25)

with torch.inference_mode():
    action = actor(example_obs)
    print("Action shape:", action.shape)
    print("Action:", action)

traced_actor = torch.jit.trace(actor, example_obs)
traced_actor.save("lm3_actor_normalized.pt")

print("Export completed.")


class LM3DeployPolicy(nn.Module):
    def __init__(self, actor, mean, variance):
        super().__init__()

        self.actor = actor

        self.register_buffer(
            "mean", mean.float().clone()
        )
        self.register_buffer(
            "variance", variance.float().clone()
        )

    def forward(self, obs):
        # 与 SKRL 1.4.3 的预处理规则保持一致
        normalized = (
            (obs - self.mean)
            / torch.sqrt(self.variance + 1e-8)
        )

        normalized = torch.clamp(
            normalized, -5.0, 5.0
        )

        return self.actor(normalized)


preprocessor_state = checkpoint["state_preprocessor"]

deploy_policy = LM3DeployPolicy(
    actor,
    preprocessor_state["running_mean"],
    preprocessor_state["running_variance"],
)

deploy_policy.eval()

example_obs = torch.zeros(1, 25)

with torch.inference_mode():
    example_action = deploy_policy(example_obs)
    print("Deploy action:", example_action)

traced_policy = torch.jit.trace(
    deploy_policy,
    example_obs,
)

traced_policy.save("lm3_reach_policy.pt")