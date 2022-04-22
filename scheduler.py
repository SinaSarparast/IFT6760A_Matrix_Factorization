class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling. This is exactly the scheduler used by attention is all you need repository.
    See https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
    '''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps, n_steps=0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = n_steps
        self.learning_rates = [lr_mul]


    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def get_last_lr(self):
        return self.learning_rates

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        print('>>> scheduler: learning rate updated to: ',lr)
        self.learning_rates.append(lr)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
