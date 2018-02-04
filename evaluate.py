import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.optimizers import RMSprop
from settings import CATEGORIES
from collections import Counter

from generator import DataGenerator

data_dir = Path("data")
valid_dir = data_dir / "valid"

valid_generator = DataGenerator.get_test_generator(valid_dir, class_mode="categorical")

# model_name = "inceptionresnet"
model_name = "xception"
model = load_model(f"{model_name}_final")
model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
eval_scores = model.evaluate_generator(valid_generator)
print(eval_scores)
# y_pred = np.argmax(eval_scores, axis=1)
# print(Counter(y_pred))
