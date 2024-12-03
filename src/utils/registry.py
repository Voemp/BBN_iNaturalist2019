def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    一个用于管理模块注册的辅助类，它扩展了字典并提供了注册功能。

    例如，创建一个注册表：
        some_registry = Registry({"default": default_module})

    有两种方式注册新模块：
    1): 普通方式是直接调用注册函数：
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): 声明模块时用作装饰器：
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...

    访问模块就像使用字典一样，例如：
        f = some_registry["foo_module"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # 用作函数调用
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # 用作装饰器
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn