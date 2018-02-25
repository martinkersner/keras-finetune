import random
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from skimage.transform import resize
from skimage.transform import rotate
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder

augmentation_methods = ["resize_random_crop_aug",
                        "resize_central_crop_aug"]

# TODO additional augmentation method
# TODO add_arguments
# TODO paralelize loading!, caching?
# TODO logging
class DataGenerator(object):
    """
    1. Loading
    2. Augmentation
    3. Preprocessing
    4. Providing
    """
    def __init__(
        self,
        train_dir=None,
        valid_dir=None,
        train_aug=None,
        val_aug=None,
        image_extension=".png",
        batch_size=None,
        target_size=400,
        rotation_range=None,
        fill_mode="constant",
        horizontal_flip=None,
        vertical_flip=None,
        rescale=None
    ):
        assert train_dir is not None
        assert valid_dir is not None
        assert train_aug is not None
        assert val_aug is not None

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.train_aug = train_aug
        self.val_aug = val_aug

        self.batch_size = batch_size
        self.image_extension = image_extension
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale

        self.num_train_data = self._count_image_files(self.train_dir)
        self.num_valid_data = self._count_image_files(self.valid_dir)

        self.class_mode = "categorical"

    def _count_image_files(self, path):
        """ Images are expected to be in subdirectories of given `path`.
        """
        return len(list(Path(path).glob(f"*/*{self.image_extension}")))

    def get_train_generator(
        self,
        fill_mode="wrap",
        width_shift_range=0.1,
        height_shift_range=0.1,
    ):
        train_datagen = ImageDataGenerator(
            self.target_size,
            rescale=self.rescale,
            rotation_range=self.rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            # shear_range=0.0,
            # zoom_range=0.0,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            fill_mode=self.fill_mode,
            augmentation_method=self.train_aug)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=True)

        return train_generator


    def get_valid_generator(self, directory=None):
        val_datagen = ImageDataGenerator(target_size=self.target_size,
                                         rescale=self.rescale,
                                         augmentation_method=self.val_aug)

        if directory is None:
            directory = self.valid_dir

        val_generator = val_datagen.flow_from_directory(
            directory,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=False)

        return val_generator


class OneHotEncoder(object):
    def __init__(self, n_values):
        self.n_values = n_values

    def transform(self, labels):
        if not isinstance(labels, list):
            labels = [labels]

        num_data = len(labels)
        one_hot = np.zeros((num_data, self.n_values), dtype=np.uint8)
        one_hot[np.arange(num_data), np.array(labels)] = 1
        return one_hot

# TODO iamge extension
# TODO width_shift_range
# TODO height_shift_range=0.1,
# TODO smooth edges after rotation
class ImageDataGenerator(object):
    def __init__(self,
                 target_size,
                 width_shift_range=None,
                 height_shift_range=None,
                 rescale=None,
                 rotation_range=None,
                 horizontal_flip=None,
                 vertical_flip=None,
                 fill_mode=None,
                 augmentation_method="resize_central_crop_aug"):

        self.rescale = rescale
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.operations = []
        self._add_operation("rescale", self.rescale)
        self._add_operation("horizontal_flip", self.horizontal_flip)
        self._add_operation("vertical_flip", self.vertical_flip)
        self._add_operation("rotation_range", self.rotation_range)

        self.augmentation_method = eval(f"self.{augmentation_method}")

    def _add_operation(self, operation_name, value):
        if value and value is not None:
            print(f"self._{operation_name}({value})")  # TODO delete
            self.operations.append(eval(f"self._{operation_name}({value})"))

    def _rescale(self, factor: float):
        def rescale_op(image_batch):
            return np.array(image_batch*factor, dtype=np.float32)
        return rescale_op

    def _random_flip(self, image_batch, axis):
        if np.random.uniform() > 0.5:
            return np.flip(image_batch, axis=axis)
        else:
            return image_batch

    def _horizontal_flip(self, none):
        def horizontal_flip_op(image_batch):
            return self._random_flip(image_batch, axis=1)
        return horizontal_flip_op

    def _vertical_flip(self, none):
        def vertical_flip_op(image_batch):
            return self._random_flip(image_batch, axis=0)
        return vertical_flip_op

    def _rotation_range(self, range: int):
        assert range != 0

        def rotation_range_op(image_batch):
            angle = np.random.randint(0, range)
            return rotate(image_batch, angle, mode=self.fill_mode)
        return rotation_range_op

    def _process_image(self, image_path):
        # TODO image preprocessing, e.g. inception, vgg
        image = imread(image_path)
        image = image[:, :, 0:3]  # remove alpha channel from RGBA
        image = self.augmentation_method(image, self.target_size)

        for op in self.operations:
            image = op(image)

        return image

    def _flow_from_directory_gen(self):
        while True:
            for idx in range(0, len(self.data)-self.batch_size, self.batch_size):
                batch = self.data[idx:idx+self.batch_size]
                image_batch, label_batch = [], []

                for path, label in batch:
                    image = self._process_image(path)
                    onehot_label = self.onehot_encoder.transform(label)

                    image_batch.append(image)
                    label_batch.append(onehot_label)

                image_batch = np.stack(image_batch)
                label_batch = np.concatenate(label_batch)

                yield image_batch, label_batch

            if self.shuffle:
                self.data = random.shuffle(self.data)

    def flow_from_directory(self,
                            path: Path,
                            target_size: int,
                            batch_size: int,
                            class_mode: str,
                            shuffle: bool=True):

        self.path = path
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.shuffle = shuffle

        # if class_mode == "categorical":  # TODO
        self.categories = sorted([p.name for p in list(path.glob("*"))])
        self.label_encoder = LabelEncoder().fit(self.categories)
        self.num_categories = len(list(path.glob("*")))
        self.onehot_encoder = OneHotEncoder(n_values=self.num_categories)

        self.data = []
        for cat in self.categories:
            label_id = self.label_encoder.transform([cat])[0]
            all_img_paths = list(Path(path / cat).glob("*.png"))
            num_all_images = len(all_img_paths)
            cat_labels = [label_id]*num_all_images
            self.data.extend(zip(all_img_paths, cat_labels))

        if self.shuffle:
            random.shuffle(self.data)

        return self._flow_from_directory_gen()

    def resize_image(self,
                     img: np.array,
                     target_size: int):
        """ Resize image and preserve aspect ratio.
        The smaller dimension is resized to `target_size`.
        """
        img_shape = img.shape[0:2]
        factor = min(img_shape) / target_size

        new_size = np.array([int(size/factor) for size in img_shape])
        new_size[new_size < target_size] = target_size

        return resize(img, new_size,
                      mode='constant', cval=0,
                      clip=True, preserve_range=False)

    def resize_central_crop_aug(self,
                                img: np.array,
                                target_size: int):
        img_resized = self.resize_image(img, target_size)
        height, width, _ = img_resized.shape

        def compute_offset(current_size, target_size):
            return (current_size // 2) - target_size // 2

        y_offset = compute_offset(height, target_size)
        x_offset = compute_offset(width, target_size)

        return img_resized[y_offset:y_offset+target_size,
                           x_offset:x_offset+target_size]

    def resize_random_crop_aug(self,
                             img: np.array,
                             target_size: int):
        def randint(max_val: int):
            if max_val != 0:
                return np.random.randint(0, max_val)
            else:
                return 0

        img_resized = self.resize_image(img, target_size)
        resized_shape = img_resized.shape[0:2]

        y_range, x_range = [size - target_size for size in resized_shape]
        y_offset, x_offset = 0, 0

        if y_range != 0:
            y_offset = randint(y_range)

        if x_range != 0:
            x_offset = randint(x_range)

        return img_resized[y_offset:y_offset+target_size,
                           x_offset:x_offset+target_size]
