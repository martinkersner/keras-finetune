from keras.preprocessing.image import ImageDataGenerator


class DataGenerator(object):
    def __init__(self, train_dir, valid_dir):
        self.train_dir = train_dir
        self.valid_dir = valid_dir

        self.target_size = (299, 299)
        self.class_mode = "categorical"
        self.batch_size = 16

    def get_train_generator(self):
        train_datagen = ImageDataGenerator(
            rescale=1/255,
            rotation_range=60,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True)

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

        return train_generator

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
