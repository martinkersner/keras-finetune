from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataGenerator(object):
    def __init__(self, train_dir, valid_dir):
        self.train_dir = train_dir
        self.valid_dir = valid_dir

        self.target_size = (400, 400)
        self.class_mode = "categorical"
        self.batch_size = 16

    def get_train_generator(self,
                            fill_mode="wrap",
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            wrapper=True):
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

        # train_datagen = ImageDataGenerator(
            # rescale=1/255,
            # rotation_range=50,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True,
            # vertical_flip=True)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=True)

        if wrapper is True:
            return self._data_generator_wrapper(train_generator)
        else:
            return train_generator

    def _data_generator_wrapper(self, train_generator, target_height=350, target_width=350):
        img_batch_modified = []
        for img_batch, onehot_batch in train_generator:
            for idx in range(img_batch.shape[0]):
                img = img_batch[idx]
                h_offset = np.random.randint(img.shape[0] - target_height)
                w_offset = np.random.randint(img.shape[1] - target_width)
                img = img[h_offset:h_offset+target_height, w_offset:w_offset+target_width]
                img_batch_modified.append(np.expand_dims(img, axis=0))

            yield np.concatenate(img_batch_modified), onehot_batch

    @staticmethod
    def normalize_data_generator():
        return ImageDataGenerator(rescale=1/255)

    def get_valid_generator(self, directory=None):
        val_datagen = DataGenerator.normalize_data_generator()

        if directory is None:
            directory = self.valid_dir

        val_generator = val_datagen.flow_from_directory(
            directory,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=False)

        return val_generator

    @staticmethod
    def get_test_generator(
        directory,
        target_size=(299, 299),
        batch_size=64,
        class_mode=None
    ):
        print(f"target size: {target_size}\n"
              f"batch_size: {batch_size}\n"
              f"class_mode: {class_mode}")
        test_datagen = DataGenerator.normalize_data_generator()

        test_generator = test_datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False)

        return test_generator
