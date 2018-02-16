import logging

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

from utils import format_text


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
        logging.basicConfig(level=logging.INFO)
        self.kwargs = {}

        return eval(f"self._{args.optimizer}")

    def _log(self, optimizer_name, kwargs):
        with format_text("red") as fmt:
            logging.info(fmt(optimizer_name))
            for key, val in kwargs.items():
                logging.info(fmt(f"{key}: {val}"))

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
        if self.args.lr is not None:
            self.kwargs["lr"] = self.args.lr
        if self.args.epsilon is not None:
            self.kwargs["epsilon"] = self.args.epsilon
        if self.args.rho is not None:
            self.kwargs["rho"] = self.args.rho
        if self.args.decay is not None:
            self.kwargs["decay"] = self.args.decay

        self._log("RMSprop", self.kwargs)
        return RMSprop(**self.kwargs)

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
        """Default parameters follow those provided in the original
        paper."""
        if self.args.lr is not None:
            self.kwargs["lr"] = self.args.lr
        if self.args.beta_1 is not None:
            self.kwargs["beta_1"] = self.args.beta_1
        if self.args.beta_2 is not None:
            self.kwargs["beta_2"] = self.args.beta_2
        if self.args.epsilon is not None:
            self.kwargs["epsilon"] = self.args.epsilon
        if self.args.decay is not None:
            self.kwargs["decay"] = self.args.decay
        if self.args.amsgrad is not None:
            self.kwargs["amsgrad"] = self.args.amsgrad

        self._log("Adam", self.kwargs)
        return Adam(**self.kwargs)

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
        parser.add_argument("--lr", type=float, default=None)
        parser.add_argument("--rho", type=float, default=None)
        parser.add_argument("--epsilon", type=float, default=None)
        parser.add_argument("--decay", type=float, default=None)
        parser.add_argument("--momentum", type=float, default=None)
        parser.add_argument("--nesterov", type=bool, default=None,
                            help="boolean")
        parser.add_argument("--beta_1", type=float, default=None)
        parser.add_argument("--beta_2", type=float, default=None)
        parser.add_argument("--amsgrad", type=bool, default=None)
        parser.add_argument("--schedule_decay", type=float, default=None)
        parser.add_argument("--optimizer", type=str,
                            choices=allowed_optimizers, default="RMSprop")
