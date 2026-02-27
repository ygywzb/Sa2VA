from .configuration_sa2va_chat import Sa2VAChatConfig


class Sa2VADevChatConfig(Sa2VAChatConfig):
    """Sa2VA dev的模型配置类，只加了预算参数，LIS输入维度与模型的LLM维度相等，初始化LIS时直接获取父类LLM维度信息"""

    model_type = "sa2va_dev_chat"

    def __init__(self, budgets: float = None, **kwargs):
        super().__init__(**kwargs)
        # assert budgets is not None, "budgets should be provided for Sa2VADevChatConfig"
        if budgets is None:
            budgets = 0.3  # default
        self.budgets = budgets

    def to_dict(self):
        output = super().to_dict()
        output["budgets"] = self.budgets
        return output
