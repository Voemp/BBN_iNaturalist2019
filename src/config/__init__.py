from yacs.config import CfgNode  # 从yacs库导入CfgNode类，用于配置文件的读取和管理

# 创建一个空的配置对象cfg
cfg = CfgNode()
cfg.set_new_allowed(True)
# 从指定的YAML配置文件"config.yaml"加载配置并合并到cfg对象中
cfg.merge_from_file("config/config.yaml")
