import wandb
import torch

class MyLogger:
    def __init__(self):
        self.log_dict = {}

    def log(self, log_dict):
        self.log_dict.update(log_dict)

    def send_log(self):
        # Perform logging using the values in self.log_dict
        wandb.log(self.log_dict)
        #print(self.log_dict)
        self.log_dict = {}

class LoggerHook(torch.nn.Module):
    def __init__(self, model, optimizer, log_loss=True, log_gradients=True, log_activations=True, model_name=None, mlog=None):
        super(LoggerHook, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.log_loss_flag = log_loss
        self.log_gradients_flag = log_gradients
        self.log_activations_flag = log_activations
        self.model_name = model_name or type(model).__name__
        self.mlog = mlog

    def forward(self, *inputs, y=None):
        activations = self.model(*inputs)
        self.log_weights_and_biases()
        self.log_learning_rate()
        if self.log_activations_flag:
            self.log_activations(activations)
        if y is not None:
            loss = self.loss_fn(activations, y)
            if self.log_loss_flag:
                if self.mlog is not None:
                    self.mlog.log({f"{self.model_name}_loss": loss.item()})
                else:
                    wandb.log({f"{self.model_name}_loss": loss.item()})
            if self.log_gradients_flag:
                loss.backward()
                self.log_gradients(self.model)
            self.optimizer.step()
        return activations

    def loss_fn(self, activations, y):
        return torch.nn.functional.cross_entropy(activations, y)

    def log_weights_and_biases(self):
        for name, param in self.model.named_parameters():
            if self.mlog is not None:
                self.mlog.log({f"{self.model_name}_{name}": param.data})
            else:
                wandb.log({f"{self.model_name}_{name}": param.data})

    def log_activations(self, activations):
        if self.mlog is not None:
            self.mlog.log({f"{self.model_name}_activations": activations.data})
        else:
            wandb.log({f"{self.model_name}_activations": activations.data})

    def log_gradients(self, model):
        for name, param in model.named_parameters():
            if self.mlog is not None:
                self.mlog.log({f"{self.model_name}_{name}_grad": param.grad.data})
            else:
                wandb.log({f"{self.model_name}_{name}_grad": param.grad.data})

    def log_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            if self.mlog is not None:
                self.mlog.log({f"{self.model_name}_learning_rate": param_group['lr']})
            else:
                wandb.log({f"{self.model_name}_learning_rate": param_group['lr']})
