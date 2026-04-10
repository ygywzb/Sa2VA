from .configuration_sa2va_chat import Sa2VAChatConfig


class Sa2VADevChatConfig(Sa2VAChatConfig):
    """Sa2VA Dev configuration with explicit LIS/Top-K inference parameters."""

    model_type = "sa2va_dev_chat"

    def __init__(
        self,
        budgets: float = 0.3,
        scorer_hidden_dim: int = 1792,
        scorer_init_scale: float = 0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if budgets is None:
            budgets = 0.3
        budgets = float(budgets)
        if not (0.0 < budgets <= 1.0):
            raise ValueError(f"budgets must be in (0, 1], but got {budgets}.")

        self.budgets = budgets
        self.scorer_hidden_dim = int(scorer_hidden_dim)
        self.scorer_init_scale = float(scorer_init_scale)

    def to_dict(self):
        output = super().to_dict()
        output["budgets"] = self.budgets
        output["scorer_hidden_dim"] = self.scorer_hidden_dim
        output["scorer_init_scale"] = self.scorer_init_scale
        return output
