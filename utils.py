import os
from pathlib import Path
import random
import shutil
import time
import humanfriendly as hf
import numpy as np

from keras.models import model_from_json


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


def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate


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
               weights_extension: str=".h5") -> None:
    if not isinstance(name, Path):
        name = Path(name)

    model_arch = model.to_json()
    with open(name.with_suffix(architecture_extension), "w") as f:
        f.write(model_arch)

    model.save_weights(name.with_suffix(weights_extension))


def load_model(model_name: str,
               architecture_extension: str=".json",
               weights_extension: str=".h5"):
    if not isinstance(model_name, Path):
        model_name = Path(model_name)

    with open(model_name.with_suffix(architecture_extension), "r") as f:
        model_arch = f.read()

    model = model_from_json(model_arch)
    model.load_weights(model_name.with_suffix(weights_extension))

    return model


class TimeMeasure(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.start = time.time()

    def __exit__(self):
        self.end = time.time()
        print(hf.format_timespan(self.end-self.start))


from contextlib import contextmanager
@contextmanager
def measure_time(msg):
    start = time.time()
    yield
    end = time.time()
    print(f"{msg}: {hf.format_timespan(end-start)}")
