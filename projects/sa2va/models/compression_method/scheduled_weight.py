# 原代码
# class ScheduledWeightTrainer(Trainer):
#     def __init__(self, *args, reg_weight_start=0.1, reg_weight_end=3.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.reg_weight_start = reg_weight_start
#         self.reg_weight_end = reg_weight_end

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         """
#         Overrides compute_loss to dynamically calculate regularization_weight.
#         """
#         total_steps = self.state.max_steps
#         current_step = self.state.global_step

#         if total_steps > 0:
#             # Use min to ensure progress does not exceed 1.0
#             progress = min(current_step / total_steps, 1.0)
#             current_weight = self.reg_weight_start + (self.reg_weight_end - self.reg_weight_start) * progress
#         else:
#             # Use the starting weight if total_steps is not yet computed (value is -1)
#             current_weight = self.reg_weight_start

#         # Set the calculated weight on the actual model
#         actual_model = model.module if hasattr(model, 'module') else model
#         actual_model.regularization_weight = current_weight

#         # Log the weight
#         if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
#             # Print only on the main process to avoid duplicates
#             if self.is_world_process_zero():
#                 print(f"\n[Step {self.state.global_step}] Set regularization_weight to: {current_weight:.4f}")

#         # Call the parent's compute_loss method
#         return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.runner import FlexibleRunner
from mmengine.dist import get_rank


@HOOKS.register_module()
class ScheduledWeightHook(Hook):
    def __init__(self, reg_weight_start=0.1, reg_weight_end=3.0, log_interval=10):
        self.reg_weight_start = reg_weight_start
        self.reg_weight_end = reg_weight_end
        self.log_interval = log_interval

    def before_train_iter(self, runner: FlexibleRunner, batch_idx, data_batch=None):
        total_steps = runner.max_iters
        current_step = runner.iter

        if total_steps > 0:
            # Use min to ensure progress does not exceed 1.0
            progress = min(current_step / total_steps, 1.0)
            current_weight = (
                self.reg_weight_start
                + (self.reg_weight_end - self.reg_weight_start) * progress
            )
        else:
            # Use the starting weight if total_steps is not yet computed (value is -1)
            current_weight = self.reg_weight_start

        # Set the calculated weight on the actual model
        actual_model = (
            runner.model.module if hasattr(runner.model, "module") else runner.model
        )
        actual_model.regularization_weight = current_weight

        # Log the weight
        if current_step > 0 and current_step % self.log_interval == 0:
            # Print only on the main process to avoid duplicates
            if get_rank() == 0:
                print(
                    f"\n[Step {current_step}] Set regularization_weight to: {current_weight:.4f}"
                )

    pass
