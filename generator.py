from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.transform import resize
from skimage.io import imread

# TODO additional augmentation method
# TODO add_arguments
class DataGenerator(object):
    def __init__(self, train_dir, valid_dir,
                 batch_size: int=None,
                 img_extension: str=".png",
                 target_size: tuple=(400, 400),
                 final_size: tuple=(350, 350),
                 augmentation_method: str):
        self.train_dir = train_dir
        self.valid_dir = valid_dir

        self.num_train_data = len(list(Path(self.train_dir)
                                       .glob(f"*/*{img_extension}")))
        self.num_valid_data = len(list(Path(self.valid_dir)
                                       .glob(f"*/*{img_extension}")))

        self.target_size = target_size
        self.final_size = final_size
        self.class_mode = "categorical"
        self.batch_size = batch_size
        self.augmentation_method = eval(f"self.{augmentation_method}")

    def get_train_generator(self,
                            fill_mode="wrap",
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            wrapper=False):
        train_datagen = ImageDataGenerator(
            rescale=1/255,
            rotation_range=360,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode=fill_mode)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=True)

        if wrapper is True:
            return self._data_generator_wrapper(train_generator,
                                                is_training=True)
        else:
            return train_generator

    def resize_image(self,
                     img: np.array,
                     target_size: int):
        """ Resize image and preserve aspect ratio.
        The smaller dimension is resized to `target_size`.
        """
        img_shape = img.shape[0:2]
        factor = min(img_shape) / target_size
        new_size = [int(size/factor) for size in img_shape]

        return resize(img, new_size,
                      mode='constant', cval=0,
                      clip=True, preserve_range=False)

    def resize_central_crop_aug(self,
                                img: np.array,
                                target_size: int):
        img_resized = resize_image(img, target_size)
        height, width, _ = img_resized.shape

        def compute_offset(current_size, target_size):
            return (current_size // 2) - target_size // 2

        y_offset = compute_offset(height, target_size)
        x_offset = compute_offset(width, target_size)

        return img_resized[y_offset:y_offset+target_size,
                           x_offset:x_offset+target_size]

    def resize_rand_crop_aug(self,
                             img: np.array,
                             target_size: int):
        def randint(max_val: int):
            if max_val != 0:
                return np.random.randint(0, max_val)
            else:
                return 0

        img_resized = resize_image(img, target_size)
        resized_shape = img_resized.shape[0:2]

        y_range, x_range = [size - target_size for size in resized_shape]
        y_offset, x_offset = 0, 0

        if y_range != 0:
            y_offset = randint(y_range)

        if x_range != 0:
            x_offset = randint(x_range)

        return img_resized[y_offset:y_offset+target_size,
                           x_offset:x_offset+target_size]

    # TODO test!
    # TODO apply on full batch!
    # use for make_submission.py
    def _data_generator_wrapper(self, generator, is_training=False):
        img_batch_modified = []
        for img_batch, onehot_batch in generator:
            for idx in range(img_batch.shape[0]):
                img = img_batch[idx]

                img_aug = self.augmentation_method(img)

                img_batch_modified.append(np.expand_dims(img_aug, axis=0))

            yield np.concatenate(img_batch_modified), onehot_batch
            img_batch_modified = []

    @staticmethod
    def normalize_data_generator():
        return ImageDataGenerator(rescale=1/255)

    def get_valid_generator(self, directory=None, wrapper=False):
        val_datagen = DataGenerator.normalize_data_generator()

        if directory is None:
            directory = self.valid_dir

        val_generator = val_datagen.flow_from_directory(
            directory,
            target_size=self.final_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=False)

        if wrapper is True:
            return self._data_generator_wrapper(val_generator,
                                                is_training=False)
        else:
            return val_generator
