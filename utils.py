import os
from pathlib import Path
import random
import shutil
import time
import re

from termcolor import colored
import humanfriendly as hf
import numpy as np
from keras.models import model_from_json
from contextlib import contextmanager


def make_dir(d: str) -> Path:
    if not isinstance(d, Path):
        d = Path(d)

    if not Path(d).exists():
        os.mkdir(str(d))

    return d.resolve()

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class Decay(object):
    def __init__(self, initial_lr=1e-3):
        self.initial_lr = initial_lr

    @property
    def exp(self, k=0.1):
        def exp_function(epoch):
            return self.initial_lr * np.exp(-k*epoch)
        return exp_function


def join_parts(parts):
    path = Path("")
    for p in parts:
        path = Path(path / p)
    return path


def split_train_val(train_dir, valid_dir, categories):
    # Run only once!
    # Split training data to train and validation parts.
    valid_ratio = 0.2
    if not valid_dir.exists:
        os.mkdir(valid_dir)

    for category in categories:
        if not Path(valid_dir / category).exists():
            os.mkdir(valid_dir / category)

        name = list(Path(train_dir / category).glob("*"))
        random.shuffle(name)
        valid_images = name[:int(len(name)*valid_ratio)]
        for src in valid_images:
            parts = list(src.parts)
            parts[parts.index("train")] = "valid"
            dst = join_parts(parts)
            shutil.move(src, dst)


def save_model_architecture(model, model_name, extension=".json"):
    if not isinstance(model_name, Path):
        model_name = Path(model_name)

    model_json = model.to_json()
    with open(model_name.with_suffix(extension), "w") as f:
        f.write(model_json)


def save_model(model, name: str,
               architecture_extension: str=".json",
               weights_extension: str=".h5",
               save_architecture: bool=True,
               save_weights: bool=True) -> None:
    if not isinstance(name, Path):
        name = Path(name)

    if save_architecture:
        model_arch = model.to_json()
        with open(name.with_suffix(architecture_extension), "w") as f:
            f.write(model_arch)

    if save_weights:
        model.save_weights(name.with_suffix(weights_extension))


def load_model(model_name: str,
               architecture_extension: str=".json",
               weights_extension: str=".h5"):
    if not isinstance(model_name, Path):
        model_name = Path(model_name)

    initial_epoch = 1

    if model_name.is_dir():
        # use the latest checkpoint file
        all_models = list(model_name.glob(f"*{weights_extension}"))
        pattern = "\w+_([0-9]+)" + weights_extension

        model_epoch = sorted([[int(re.match(pattern, str(path.name)).group(1)), path] for path in all_models],
                             reverse=True)
        model_name = model_epoch[0][1]
        initial_epoch = model_epoch[0][0]
        architecture_name = Path("_".join(str(model_name).split("_")[:-1]))

    with open(architecture_name.with_suffix(architecture_extension), "r") as f:
        model_arch = f.read()

    model = model_from_json(model_arch)
    model.load_weights(model_name.with_suffix(weights_extension))

    return model, initial_epoch


class TimeMeasure(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.start = time.time()

    def __exit__(self):
        self.end = time.time()
        print(hf.format_timespan(self.end-self.start))


@contextmanager
def timer(msg):
    with format_text("green") as f:
        print(f(f"{msg} starts"))
        start = time.time()
        yield
        end = time.time()
        print(f(f"{msg}: {hf.format_timespan(end-start)}"))


class TextFormatter(object):
    def __init__(self, color, attrs=None):
        self.color = color
        self.attrs = attrs

    def __call__(self, text):
        return colored(text, self.color, attrs=self.attrs)


@contextmanager
def format_text(color, attrs=None):
    yield TextFormatter(color, attrs=attrs)
