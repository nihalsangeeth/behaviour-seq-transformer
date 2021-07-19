import numpy as np

class GradientClipping:
    def __init__(self, clip_value):
        self.epoch_grads = []
        self.total_grads = []
        self.clip = clip_value

    def track_grads(self, x, grad_input, grad_output):
        self.epoch_grads.append(grad_input[0].norm().cpu().data.numpy())

    def register_hook(self, encoder):
        encoder.register_backward_hook(self.track_grads)

    def gradient_mean(self):
        return np.mean(self.epoch_grads)

    def gradient_std(self):
        return np.std(self.epoch_grads)

    def reset_gradients(self):
        self.total_grads.append(self.epoch_grads)
        self.epoch_grads = []

    def update_clip_value(self):
        self.clip = self.gradient_mean() + self.gradient_std()

    def update_clip_value_total(self):
        grads = [y for x in self.total_grads.append(self.epoch_grads) for y in x]
        self.clip = np.mean(grads)
