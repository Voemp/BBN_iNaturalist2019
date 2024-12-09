import torch
import yaml


# 加载配置文件
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# 保存模型
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


# 加载模型
def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    return model
