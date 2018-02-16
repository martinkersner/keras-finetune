import argparse
from pathlib import Path
from datetime import datetime
import json
import math

import tensorflow as tf
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201  # not working well

from keras.callbacks import EarlyStopping, LambdaCallback
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Dropout

from keras.regularizers import l2
from optimizers import Optimizer

from utils import save_model, lr_schedule, save_model_architecture, format_text, Decay
from generator import DataGenerator
from utils import timer, load_model, make_dir


allowed_models = ["Xception", "InceptionResNetV2", "InceptionV3", "VGG19", "DenseNet201"]


# TODO logging
# TODO  dir cleanup when fail
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
        self.tag = self._build_tag(self.args.tag)
        self.loss = args.loss

        self._init_logging()
        self.setup_data_generator()

        if 1 in self.stages:
            self.build_model()
            self.tensorboard = TensorboardKeras(self.model, str(self.log_dir))
            with timer("STAGE 1"):
                self.train_first_stage()

        if 2 in self.stages:
            if len(self.stages) == 1:  # train only second stage
                if self.args.checkpoint_dir is None:
                    model_name = f"{args.model}_pretrain"
                else:
                    model_name = self.args.checkpoint_dir

                self.model, self.initial_epoch = load_model(model_name)
                self.tensorboard = TensorboardKeras(self.model, str(self.log_dir))

            with timer("STAGE 2"):
                self.train_second_stage(self.initial_epoch)

    def _init_logging(self):
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_dir = Path(self.args.log_dir) / f"{self.model_name}_{timestamp}{self.tag}"
        self.log_dir = make_dir(log_dir)

        with format_text("yellow") as f:
            print(f(f"Logging to {self.log_dir}"))

        with open(self.log_dir / "config.json", "w") as args_log:
            json.dump(vars(self.args), args_log, indent=2)

        self.model_identificator = f"{self.model_name}_{self.args.optimizer}"
        self.saver = Saver(self.log_dir, self.model_identificator)

    def _split_strip_string(self, string, split_character=","):
        return [c.strip() for c in string.split(split_character)]

    def _build_categories(self, categories):
        return self._split_strip_string(categories)

    def _build_metrics(self, metrics):
        return self._split_strip_string(metrics)

    def _build_stages(self, stages: str):
        return [int(s) for s in self._split_strip_string(stages)]

    def _build_tag(self, tag: str):
        if len(tag) >= 1:
            return f"_{tag}"
        else:
            return ""

    def setup_data_generator(self):
        target_size = (self.args.target_size, self.args.target_size)
        final_size = (self.args.final_size, self.args.final_size)
        self.dg = DataGenerator(self.args.train_dir, self.args.valid_dir,
                                batch_size=self.args.batch_size,
                                target_size=target_size,
                                final_size=final_size)

        wrapper = self.args.data_wrapper
        self.train_generator = self.dg.get_train_generator(wrapper=wrapper)
        self.val_generator = self.dg.get_valid_generator(wrapper=wrapper)

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
        net = Dropout(self.args.dropout_rate)(net)
        self.output_tensor = Dense(
            self.num_categories,
            activation="softmax"
        )(net)

        self.model = Model(inputs=self.input_tensor,
                           outputs=self.output_tensor)

        save_model_architecture(self.model, self.log_dir / self.model_identificator)

        if self.args.print_summary:
            print(self.model.summary())

    def _get_steps_per_epoch(self):
        if self.args.steps_per_epoch is None:
            steps_per_epoch = math.ceil(self.dg.num_train_data / self.args.batch_size)
            validation_steps = math.ceil(self.dg.num_valid_data / self.args.batch_size)
            return steps_per_epoch, validation_steps
        else:
            return self.args.steps_per_epoch, self.args.validation_steps

    def train_first_stage(self):
        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.loss,
            metrics=self.metrics
        )

        early_stopping_cb = EarlyStopping(
            monitor="val_loss",
            min_delta=self.args.es_min_delta,
            patience=self.args.es_patience,
            verbose=1,
            mode="auto"
        )

        callbacks = [self.saver.checkpoint_callback,
                     early_stopping_cb,
                     self.tensorboard.on_epoch_end_cb()]

        if self.args.exp_decay_lr:
            decay = Decay(initial_lr=self.args.lr)
            callbacks.append(LearningRateScheduler(
                decay.exp(self.args.exp_decay_factor)
            ))

        steps_per_epoch, validation_steps = self._get_steps_per_epoch()
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.args.pretrain_num_epoch,
            validation_data=self.val_generator,
            callbacks=callbacks,
            workers=self.args.num_workers,
            verbose=2
        )

    def train_second_stage(self, initial_epoch):
        for layer in self.model.layers:
            layer.W_regularizer = l2(self.args.l2_regularizer)
            layer.trainable = True

        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.loss,
            metrics=self.metrics
        )

        # lrs = LearningRateScheduler(lr_schedule)
        lrs = LearningRateScheduler(exp_decay)

        callbacks = [self.saver.checkpoint_callback,
                     lrs,
                     self.tensorboard.on_epoch_end_cb()]

        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.args.train_num_epoch,
            validation_data=self.val_generator,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            workers=self.args.num_workers,
            verbose=2
        )

        save_model(self.model,
                   f"{self.model_name}_{self.args.optimizer}_final")


# TODO cleanup after training
class Saver(object):
    def __init__(self, log_dir, model_identificator, extension=".h5"):
        self.log_dir = log_dir
        self.extension = extension

        weights_filename = model_identificator + "_{epoch:02d}" + self.extension
        self.filepath = str(self.log_dir / weights_filename)

        self.checkpoint_callback = ModelCheckpoint(
            filepath=self.filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            period=1
        )

    def cleanup(self):
        """ Model file names are expected to be in format XXX_NN.h5
        where NN is epoch number and XXX is arbitrary long string without
        any numbers.
        """
        # all_models = self.log_dir.glob("*{self.extension}")
        # pattern = "\w+_([0-9]+)" + self.extension

        # last_epoch = max([int(re.match(pattern, name).group(1)) for name in all_models])

        # for name in all_models:
            # fullpath = self.log_dir / name
            # if last_epoch != jk

        pass


class TensorboardKeras(object):
    def __init__(self, model, log_dir):
        self.model = model
        self.log_dir = log_dir
        self.session = K.get_session()

        self.lr_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('lr', self.lr_ph)

        self.val_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('val/loss', self.val_loss_ph)

        self.val_acc_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('val/acc', self.val_acc_ph)

        self.train_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('train/loss', self.train_loss_ph)

        self.train_acc_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('train/acc', self.train_acc_ph)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)

    def get_lr(self):
        return K.eval(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs):
        summary = self.session.run(self.merged,
                                   feed_dict={
                                       self.lr_ph: self.get_lr(),
                                       self.val_loss_ph: logs["val_loss"],
                                       self.train_loss_ph: logs["loss"],
                                       self.val_acc_ph: logs["val_acc"],
                                       self.train_acc_ph: logs["acc"]
                                   })
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_epoch_end_cb(self):
        return LambdaCallback(on_epoch_end=lambda batch, logs:
                              self.on_epoch_end(batch, logs))


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

    parser_dir = parser.add_argument_group("Directory paths")
    parser_dir.add_argument("--train_dir", type=str, default="data/train")
    parser_dir.add_argument("--valid_dir", type=str, default="data/valid")
    parser_dir.add_argument("--log_dir", type=str, default="log")
    parser_dir.add_argument("--checkpoint_dir", type=str, default=None)

    parser_aug = parser.add_argument_group("Data augmentation techniques")
    parser_aug.add_argument("--data_wrapper", dest="data_wrapper",
                            action="store_true")
    parser_aug.add_argument("--no-data_wrapper", dest="data_wrapper",
                            action="store_false")
    parser_aug.add_argument("--target_size", type=int, default=400)
    parser_aug.add_argument("--final_size", type=int, default=350)
    parser_aug.set_defaults(data_wrapper=False)

    parser.add_argument("--train_num_epoch", type=int, default=1000)
    parser.add_argument("--pretrain_num_epoch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default="")

    parser.add_argument("--exp_decay_lr", action="store_true", default=False)
    parser.add_argument("--exp_decay_factor", type=float, default=0.1,
                        help=("Float number between 0 and 1."
                              "larger number -> larger decay"
                              "smaller number -> smaller_decay"))

    parser.add_argument("--print_summary", dest="print_summary", action="store_true")
    parser.add_argument("--no-print_summary", dest="print_summary", action="store_false")
    parser.set_defaults(print_summary=True)

    parser_es = parser.add_argument_group("Early Stopping")
    parser_es.add_argument("--es_min_delta", type=float, default=0.0001)
    parser_es.add_argument("--es_patience", type=int, default=20)

    model = Finetune(parser)
    args = parser.parse_args()
    print(args)

    model(args)


if __name__ == "__main__":
    main()
