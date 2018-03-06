import argparse
from pathlib import Path
from typing import List

import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from skimage.io import imread
from keras.optimizers import RMSprop
from settings import CATEGORIES
from keras.applications.xception import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    average_precision_score, confusion_matrix

from utils import load_model, format_text, OneHotEncoder, Submission
from generator import DataGenerator, resize_central_crop_aug, resize_5_crop_aug

"""
TODO load target size from config.json
TODO better separation evaluate/submission
"""


def evaluate(y_true_one_hot: np.array,
             y_pred_one_hot: np.array,
             labels: List[int]):
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_pred_one_hot, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average="micro", labels=labels)
    precision_macro = precision_score(y_true, y_pred, average="macro", labels=labels)
    recall_micro = recall_score(y_true, y_pred, average="micro", labels=labels)
    recall_macro = recall_score(y_true, y_pred, average="macro", labels=labels)

    mAP_micro = average_precision_score(y_true_one_hot,
                                        y_pred_one_hot,
                                        average="micro")
    mAP_macro = average_precision_score(y_true_one_hot,
                                        y_pred_one_hot,
                                        average="macro")

    print(f"accuracy: {accuracy}")
    print(f"precision (micro): {precision_micro}")
    print(f"precision (macro): {precision_macro}")
    print(f"recall (micro): {recall_micro}")
    print(f"recall (macro): {recall_macro}")
    print(f"mAP (micro): {mAP_micro}")
    print(f"mAP (macro): {mAP_macro}")

    print(confusion_matrix(y_true, y_pred))


def preprocess_5_images(img_path, target_size):
    img = imread(img_path)
    img = img[:, :, 0:3]  # remove alpha channel from RGBA
    img = resize_5_crop_aug(img, target_size, offset=20)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_image(img_path, target_size):
    img = imread(img_path)
    img = img[:, :, 0:3]  # remove alpha channel from RGBA
    img = resize_central_crop_aug(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main(args):
    checkpoint_path = args.checkpoint_path
    checkpoint_path_silky_vs_grass = args.checkpoint_path
    target_size = args.target_size

    with format_text("red") as fmt:
        if args.submission:
            print(fmt("SUBMISSION"))
        else:
            print(fmt("EVALUATION"))

    data_dir = Path(args.data_path)
    if args.submission:
        eval_dir = data_dir / "test"
        data = eval_dir.glob("*.png")
    else:
        eval_dir = data_dir / "valid"
        categories = sorted([p.name for p in list(eval_dir.glob("*"))])
        num_categories = len(categories)
        label_encoder = LabelEncoder().fit(categories)
        onehot_encoder = OneHotEncoder(n_values=num_categories)

        data = []
        for cat in categories:
            label_id = label_encoder.transform([cat])[0]
            onehot_label = onehot_encoder.transform(label_id)
            for img_path in Path(eval_dir / cat).glob("*"):
                data.append((img_path, onehot_label))

    if args.submission:
        submission = Submission(checkpoint_path)

    model, epoch, _ = load_model(f"{checkpoint_path}")
    model_silky_vs_grass, epoch_new, _ = load_model(f"{checkpoint_path_silky_vs_grass}")

    with format_text("green") as fmt:
        print(fmt(f"MODEL: {checkpoint_path}"))
        print(fmt(f"EPOCH: {epoch}"))

    # parameters are arbitrary, really?!
    model.compile(optimizer=RMSprop(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model_silky_vs_grass.compile(optimizer=RMSprop(1e-3),
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])

    predictions = []
    ground_truth = []

    for data_point in tqdm(data):
        if args.submission:
            img_path = data_point
        else:
            img_path, onehot_label = data_point

        # img = preprocess_image(img_path, target_size)
        img = preprocess_5_images(img_path, target_size)
        pred = np.mean(model.predict(img[0]), axis=0)

        pred_cat = CATEGORIES[np.argmax(pred)]
        if pred_cat == "Loose Silky-bent" or pred_cat == "Black-grass":
            pred = np.mean(model_silky_vs_grass.predict(img[0]), axis=0)

        if args.submission:
            submission.add(img_path.name,
                           CATEGORIES[np.argmax(pred)])
        else:
            predictions.append(pred)
            ground_truth.append(onehot_label)

    if args.submission:
        submission.save()
    else:
        evaluate(np.vstack(ground_truth),
                 np.vstack(predictions),
                 list(range(num_categories)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--checkpoint_path_silky_vs_grass", type=str, required=True)
    parser.add_argument("--target_size", type=int, required=True)
    parser.add_argument("--submission", dest="submission", action="store_true",
                        help=("Makes prediction on test dataset."
                              "Creates submission.csv file."))
    parser.add_argument("--no-submission", dest="submission", action="store_false",
                        help=("Makes prediction on validation dataset."
                              "Prints evalution metrics."))
    parser.set_defaults(submission=False)
    args = parser.parse_args()
    main(args)
