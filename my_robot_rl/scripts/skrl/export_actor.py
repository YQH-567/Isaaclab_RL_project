
import torch
import torch.nn as nn

from skrl.resources.preprocessors.torch import RunningStandardScaler


# 1. 修改为你的 checkpoint 实际路径
CHECKPOINT = "/home/l/isaac_space_01/my_robot_rl/logs/skrl/reach_lebai_lm3/2026-10-18_23-25-12_ppo_torch/checkpoints/best_agent.pt"
OUTPUT = "lm3_reach_policy.pt"


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

    def forward(self, obs):
        return self.policy_layer(self.net_container(obs))


class DeployPolicy(nn.Module):
    def __init__(self, actor, mean, variance, epsilon, clip):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("variance", variance.float().clone())
        self.epsilon = float(epsilon)
        self.clip = float(clip)

    def forward(self, obs):
        obs = obs.float()
        normalized = torch.clamp(
            (obs - self.mean) /
            (torch.sqrt(self.variance) + self.epsilon),
            min=-self.clip,
            max=self.clip,
        )
        return self.actor(normalized)


# 2. 加载 checkpoint
ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

actor = LM3Actor()

actor_weights = {
    k: v for k, v in ckpt["policy"].items()
    if k.startswith("net_container.")
    or k.startswith("policy_layer.")
}

actor.load_state_dict(actor_weights, strict=True)
actor.eval()


# 3. 使用 SKRL 原生预处理器恢复统计量
scaler = RunningStandardScaler(size=25, device="cpu")
scaler.load_state_dict(ckpt["state_preprocessor"])
scaler.eval()

epsilon = scaler.epsilon
clip = scaler.clip_threshold

print("epsilon:", epsilon)
print("clip_threshold:", clip)


# 4. 包装完整推理模型
model = DeployPolicy(
    actor,
    ckpt["state_preprocessor"]["running_mean"],
    ckpt["state_preprocessor"]["running_variance"],
    epsilon,
    clip,
)
model.eval()


# 5. 用非零随机输入检验归一化与推理一致性
torch.manual_seed(42)
obs = torch.randn(100, 25)

with torch.inference_mode():
    # SKRL 原生 scaler + Actor
    reference_actions = actor(scaler(obs, train=False))

    # 自定义完整部署模型
    deploy_actions = model(obs)

    preprocess_error = torch.max(torch.abs(
        scaler(obs, train=False) -
        torch.clamp(
            (obs - model.mean) /
            (torch.sqrt(model.variance) + model.epsilon),
            -model.clip,
            model.clip,
        )
    )).item()

    error = torch.max(
        torch.abs(reference_actions - deploy_actions)
    ).item()

print("Preprocessor max error:", preprocess_error)
print("Action max error:", error)

assert preprocess_error < 1e-5
assert error < 1e-5


# 6. 导出 TorchScript
scripted = torch.jit.script(model)
scripted.save(OUTPUT)

# 7. 重新加载，验证序列化一致性
loaded = torch.jit.load(OUTPUT, map_location="cpu")
loaded.eval()

with torch.inference_mode():
    scripted_actions = loaded(obs)
    export_error = torch.max(
        torch.abs(reference_actions - scripted_actions)
    ).item()

print("TorchScript max error:", export_error)

assert export_error < 1e-5

print("Export successful:", OUTPUT)
