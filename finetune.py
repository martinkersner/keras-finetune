import argparse
from pathlib import Path
from datetime import datetime
import json

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201  # not working well

from keras.callbacks import TensorBoard, Callback, ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Dropout

from keras.regularizers import l2
from keras.optimizers import RMSprop
from optimizers import Optimizer

from utils import save_model, lr_schedule, exp_decay
from generator import DataGenerator
from utils import measure_time, load_model, make_dir


allowed_models = ["Xception", "InceptionResNetV2", "InceptionV3", "VGG19", "DenseNet201"]

# TODO logging
# TODO checkpoint to separate directories
# TODO stop training when validation error does not decrease
class Finetune(Optimizer):
    def __init__(self, parser):
        super().add_arguments(parser)

    def __call__(self, args):
        self.args = args
        self.categories = self._build_categories(args.categories)
        self.num_categories = len(self.categories)
        self.optimizer = self.build_optimizer(args)
        self.model_name = args.model

        self.metrics = self._build_metrics(args.metrics)
        self.stages = self._build_stages(self.args.stages)
        self.loss = args.loss

        self._init_logging()
        self.setup_data_generator()

        if 1 in self.stages:
            print("STAGE 1")
            self.build_model()
            with measure_time("first stage"):
                self.train_first_stage()

        if 2 in self.stages:
            if len(self.stages) == 1:  # train only second stage
                self.model = load_model(f"{args.model}_pretrain")

            with measure_time("second stage"):
                print("STAGE 2")
                self.train_second_stage()

    def _init_logging(self):
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_dir = Path(self.args.log_dir) / f"{self.model_name}_{timestamp}"
        self.log_dir = make_dir(log_dir)

        self.tensorboard = TensorBoard(str(log_dir))
        with open(self.log_dir / "config.json", "w") as args_log:
            json.dump(vars(self.args), args_log)

    def _split_strip_string(self, string, split_character=","):
        return [c.strip() for c in string.split(split_character)]

    def _build_categories(self, categories):
        return self._split_strip_string(categories)

    def _build_metrics(self, metrics):
        return self._split_strip_string(metrics)

    def _build_stages(self, stages: str):
        return [int(s) in s for self._split_strip_string(stages)]

    def setup_data_generator(self):
        dg = DataGenerator(self.args.train_dir, self.args.valid_dir)
        self.train_generator = dg.get_train_generator()
        self.val_generator = dg.get_valid_generator()

    def initiate_model(self):
        return eval(
            """{}(
                include_top=False,
                weights=\"imagenet\",
                pooling=\"avg\"
            )""".format(self.model_name))

    def build_model(self):
        base_model = self.initiate_model()

        for layer in base_model.layers:
            layer.trainable = False

        self.input_tensor = base_model.input

        net = base_model.output
        print(net.shape)
        net = Dropout(self.args.dropout_rate)(net)
        self.output_tensor = Dense(
            self.num_categories,
            activation="softmax"
        )(net)

        self.model = Model(inputs=self.input_tensor,
                           outputs=self.output_tensor)
        # print(self.model.summary())

    def train_first_stage(self):
        self.model.compile(
            optimizer=RMSprop(lr_schedule(1e-3)),
            loss=self.loss,
            metrics=self.metrics
        )

        callbacks = [self.tensorboard]
        self.model.fit_generator(
            self.train_generator,
            epochs=self.args.pretrain_num_epoch,
            validation_data=self.val_generator,
            callbacks=callbacks,
            workers=4,
            verbose=2
        )

        save_model(self.model, f"{self.model_name}_pretrain")

    def train_second_stage(self):
        for layer in self.model.layers:
            layer.W_regularizer = l2(self.args.l2_regularizer)
            layer.trainable = True

        self.model.compile(
            optimizer=self.optimizer(),
            # optimizer=RMSprop(lr_schedule(0)),
            loss=self.loss,
            metrics=self.metrics
        )

        checkpointer = ModelCheckpoint(
            filepath=str(self.log_dir / Path(f"{self.model_name}_{self.args.optimizer}_" + "{epoch:02d}.h5")),
            verbose=1,
            save_best_only=True
        )

        # lrs = LearningRateScheduler(lr_schedule)
        lrs = LearningRateScheduler(exp_decay)

        callbacks = [checkpointer, self.tensorboard, lrs]
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.args.steps_per_epoch,
            epochs=self.args.train_num_epoch,
            validation_data=self.val_generator,
            callbacks=callbacks,
            initial_epoch=self.args.pretrain_num_epoch,
            workers=4,
            verbose=2
        )

        save_model(self.model, f"{self.model_name}_{self.args.optimizer}_final")


class EarlyStoppingByLossVal(Callback):
    """https://github.com/keras-team/keras/issues/114"""
    def __init__(self, monitor='loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)
            exit()

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True


def on_epoch_end(self, epoch, logs=None):
    print(K.eval(self.model.optimizer.lr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=allowed_models, default="InceptionResNetV2")
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--categories", type=str,
                        default=("Black-grass, Charlock, Cleavers, Common Chickweed,"
                        "Common wheat, Fat Hen, Loose Silky-bent, Maize,"
                        "Scentless Mayweed, Shepherds Purse, Small-flowered Cranesbill,"
                        "Sugar beet"))
    parser.add_argument("--metrics", type=str, default="accuracy")
    parser.add_argument("--loss", type=str, default="categorical_crossentropy")
    parser.add_argument("--l2_regularizer", type=float, default=1e-2)
    parser.add_argument("--stages", type=str, default="1,2",
                        help="Comma separated number of stages that will be"
                        "performed.")

    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--valid_dir", type=str, default="data/valid")
    parser.add_argument("--log_dir", type=str, default="log")

    parser.add_argument("--pretrain_num_epoch", type=int, default=40)
    parser.add_argument("--steps_per_epoch", type=int, default=400)

    model = Finetune(parser)
    args = parser.parse_args()
    print(args)

    model(args)


if __name__ == "__main__":
    main()
