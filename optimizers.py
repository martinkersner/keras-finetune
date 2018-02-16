from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


"""
Usage:
Optimizers("SGD")

TODO log parameters of optimizers
"""


allowed_optimizers = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam",
                      "Adamax", "Nadam"]


class Optimizer(object):
    def __init__(self):
        """Light weight constructor.
        Called during initialization of `Finetune` object"""
        pass

    def build_optimizer(self, args):
        self.args = args
        return eval(f"self._{args.optimizer}")

    def _SGD(self):
        """Stochastic gradient descent optimizer.

        Includes support for momentum, learning rate decay, and Nesterov
        momentum."""
        if self.args.default_optimizer_value:
            return SGD()
        else:
            return SGD(
                lr=self.args.lr,
                momentum=self.args.momentum,
                decay=self.args.decay,
                nesterov=self.args.nesterov)

    def _RMSprop(self):
        """It is recommended to leave the parameters of this optimizer
        at their default values (except the learning rate, which can be
        freely tuned).

        This optimizer is usually a good choice for recurrent neural
        networks."""
        if self.args.default_optimizer_value:
            return RMSprop(lr=self.args.lr)
        else:
            return RMSprop(
                lr=self.args.lr,
                rho=self.args.rho,
                epsilon=self.args.epsilon,
                decay=self.args.decay)

    def _Adagrad(self):
        """It is recommended to leave the parameters of this optimizer
        at their default values."""
        if self.args.default_optimizer_value:
            return Adagrad()
        else:
            return Adagrad(
                lr=self.args.lr,
                epsilon=self.args.epsilon,
                decay=self.args.decay)

    def _Adadelta(self):
        """It is recommended to leave the parameters of this optimizer
        at their default values."""
        if self.args.default_optimizer_value:
            return Adadelta()
        else:
            return Adadelta(
                lr=self.args.lr,
                rho=self.args.rho,
                epsilon=self.args.epsilon,
                decay=self.args.decay)

    def _Adam(self):
        """Default parameters follow those provided in the original paper."""
        if self.args.default_optimizer_value:
            return Adam()
        else:
            return Adam(
                lr=self.args.lr,
                beta_1=self.args.beta_1,
                beta_2=self.args.beta_2,
                epsilon=self.args.epsilon,
                decay=self.args.decay,
                amsgrad=self.args.amsgrad)

    def _Adamax(self):
        """Adamax optimizer from Adam paper's Section 7.

        It is a variant of Adam based on the infinity norm.
        Default parameters follow those provided in the paper."""
        if self.args.default_optimizer_value:
            return Adamax()
        else:
            return Adamax(
                lr=self.args.lr,
                beta_1=self.args.beta_1,
                beta_2=self.args.beta_2,
                epsilon=self.args.epsilon,
                decay=self.args.decay)

    def _Nadam(self):
        """Nesterov Adam optimizer.

        Much like Adam is essentially RMSprop with momentum, Nadam is
        Adam RMSprop with Nesterov momentum.

        Default parameters follow those provided in the paper. It is
        recommended to leave the parameters of this optimizer at their
        default values."""
        if self.args.default_optimizer_value:
            return Nadam()
        else:
            return Nadam(
                lr=self.args.lr,
                beta_1=self.args.beta_1,
                beta_2=self.args.beta_2,
                epsilon=self.args.epsilon,
                schedule_decay=self.args.schedule_decay)

    def add_arguments(self, parser):
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--rho", type=float, default=0.9)
        parser.add_argument("--epsilon", type=float, default=None)
        parser.add_argument("--decay", type=float, default=0.0)
        parser.add_argument("--momentum", type=float, default=0.0)
        parser.add_argument("--nesterov", type=bool, default=False)
        parser.add_argument("--beta_1", type=float, default=0.9)
        parser.add_argument("--beta_2", type=float, default=0.999)
        parser.add_argument("--amsgrad", type=bool, default=False)
        parser.add_argument("--schedule_decay", type=float, default=0.004)
        parser.add_argument("--optimizer", type=str, choices=allowed_optimizers,
                            default="RMSprop")

        parser.add_argument("--default_optimizer_value", type=bool,
                            default=True)
