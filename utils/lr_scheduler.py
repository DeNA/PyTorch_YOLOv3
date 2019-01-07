from typing import Tuple


class MyLRScheduler:
    """Learning rate scheduler"""
    def __init__(self, optimizer, base_lr: float=0.01, burn_in: int=0,
                 decay_steps: Tuple[int]=(400000, 450000), batch_size: int=32,
                 subdivision: int=16, init_step_count: int=0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_count = init_step_count
        self.burn_in = burn_in
        self.decay_steps = decay_steps
        self.tmp_lr = base_lr
        self.batch_size = batch_size
        self.subdivision = subdivision

    def set_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.tmp_lr / self.batch_size / self.subdivision

    def step(self):
        # Add step
        self.step_count += 1

        # Update learning rate
        if self.step_count < self.burn_in:
            self.tmp_lr = self.base_lr * pow(self.step_count / self.burn_in, 4)
        elif self.step_count == self.burn_in:
            self.tmp_lr = self.base_lr
        elif self.step_count in self.decay_steps:
            self.tmp_lr *= 0.1

        # Set new learning rate to optimizer
        self.set_lr()


def compare_optimizer_params(optimizer1, optimizer2):
    for param_group1, param_group2 in zip(optimizer1.param_groups, optimizer2.param_groups):
        assert param_group1['lr'] == param_group2['lr']
