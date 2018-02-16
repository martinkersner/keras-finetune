from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataGenerator(object):
    def __init__(self, train_dir, valid_dir,
                 batch_size=None,
                 img_extension=".png"):
        self.train_dir = train_dir
        self.valid_dir = valid_dir

        self.num_train_data = len(list(Path(self.train_dir).glob(f"*/*{img_extension}")))
        self.num_valid_data = len(list(Path(self.valid_dir).glob(f"*/*{img_extension}")))

        self.target_size = (400, 400)
        self.final_size = (350, 350)
        self.class_mode = "categorical"
        self.batch_size = batch_size

        # data wrapper randomly crops images

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
            return self._data_generator_wrapper(train_generator, is_training=True)
        else:
            return train_generator

    def _data_generator_wrapper(self, train_generator, is_training=False):
        img_batch_modified = []
        for img_batch, onehot_batch in train_generator:
            for idx in range(img_batch.shape[0]):
                img = img_batch[idx]
                if is_training:
                    h_offset = np.random.randint(img.shape[0] - self.final_size[0])
                    w_offset = np.random.randint(img.shape[1] - self.final_size[1])
                else:
                    h_offset = (img.shape[0] - self.final_size[0])//2
                    w_offset = (img.shape[1] - self.final_size[1])//2
                img = img[h_offset:h_offset+self.final_size[0], w_offset:w_offset+self.final_size[1]]
                img_batch_modified.append(np.expand_dims(img, axis=0))

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

        # if wrapper is True:
            # return self._data_generator_wrapper(val_generator, is_training=False)
        # else:
        return val_generator

    # @staticmethod
    # def get_test_generator(
        # directory,
        # target_size=(299, 299),
        # batch_size=64,
        # class_mode=None
    # ):
        # print(f"target size: {target_size}\n"
              # f"batch_size: {batch_size}\n"
              # f"class_mode: {class_mode}")
        # test_datagen = DataGenerator.normalize_data_generator()

        # test_generator = test_datagen.flow_from_directory(
            # directory,
            # target_size=target_size,
            # batch_size=batch_size,
            # class_mode=class_mode,
            # shuffle=False)

        # return test_generator
