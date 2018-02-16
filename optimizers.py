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
        self.args_dir = vars(self.args)
        logging.basicConfig(level=logging.INFO)
        self.kwargs = {}

        return eval(f"self._{args.optimizer}")

    def _log(self, optimizer_name, kwargs):
        with format_text("red") as fmt:
            logging.info(fmt(optimizer_name))
            for key, val in kwargs.items():
                logging.info(fmt(f"{key}: {val}"))

    def _set_argument(self, arg_name):
        arg_val = self.args_dir.get(arg_name)
        if arg_val is not None:
            self.kwargs[arg_name] = arg_val

    def _SGD(self):
        """Stochastic gradient descent optimizer.

        Includes support for momentum, learning rate decay, and Nesterov
        momentum."""
        self._set_argument("lr")
        self._set_argument("momentum")
        self._set_argument("decay")
        self._set_argument("nesterov")

        self._log("SGD", self.kwargs)
        return SGD(**self.kwargs)

    def _RMSprop(self):
        """It is recommended to leave the parameters of this optimizer
        at their default values (except the learning rate, which can be
        freely tuned).

        This optimizer is usually a good choice for recurrent neural
        networks."""
        self._set_argument("lr")
        self._set_argument("epsilon")
        self._set_argument("rho")
        self._set_argument("decay")

        self._log("RMSprop", self.kwargs)
        return RMSprop(**self.kwargs)

    def _Adagrad(self):
        """It is recommended to leave the parameters of this optimizer
        at their default values."""
        self._set_argument("lr")
        self._set_argument("epsilon")
        self._set_argument("decay")

        self._log("Adagrad", self.kwargs)
        return Adagrad(**self.kwargs)

    def _Adadelta(self):
        """It is recommended to leave the parameters of this optimizer
        at their default values."""
        self._set_argument("lr")
        self._set_argument("rho")
        self._set_argument("epsilon")
        self._set_argument("decay")

        self._log("Adadelta", self.kwargs)
        return Adadelta(**self.kwargs)

    def _Adam(self):
        """Default parameters follow those provided in the original
        paper."""
        self._set_argument("lr")
        self._set_argument("beta_1")
        self._set_argument("beta_2")
        self._set_argument("decay")
        self._set_argument("amsgrad")

        self._log("Adam", self.kwargs)
        return Adam(**self.kwargs)

    def _Adamax(self):
        """Adamax optimizer from Adam paper's Section 7.

        It is a variant of Adam based on the infinity norm.
        Default parameters follow those provided in the paper."""
        self._set_argument("lr")
        self._set_argument("beta_1")
        self._set_argument("beta_2")
        self._set_argument("epsilon")
        self._set_argument("decay")

        self._log("Adamax", self.kwargs)
        return Adamax(**self.kwargs)

    def _Nadam(self):
        """Nesterov Adam optimizer.

        Much like Adam is essentially RMSprop with momentum, Nadam is
        Adam RMSprop with Nesterov momentum.

        Default parameters follow those provided in the paper. It is
        recommended to leave the parameters of this optimizer at their
        default values."""
        self._set_argument("lr")
        self._set_argument("beta_1")
        self._set_argument("beta_2")
        self._set_argument("epsilon")
        self._set_argument("schedule_decay")

        self._log("Nadam", self.kwargs)
        return Nadam(**self.kwargs)

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
