from BoundingBoxCNNmodel import config
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

for csvPath in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):
    rows = open(csvPath).read().strip().split("\n")

    for row in rows:
        row = row.split(",")
        (filename, x1, y1, x2, y2, label) = row

        imagePath = os.path.sep.join([config.IMAGES_PATH, label, filename])
        image = cv2.imread(imagePath)
        (h, w) = image.shape[:2]

        x1 = float(x1) / w
        y1 = float(y1) / h
        x2 = float(x2) / w
        y2 = float(y2) / h

        image = load_img(imagePath, target_size=(224,224))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)
        bboxes.append((x1, y1, x2, y2))
        imagePaths.append(imagePath)

data = np.array(data, dtype='float32') /255.0
labels = np.array(labels)
bboxes = np.array(bboxes)
imagePaths = np.array(imagePaths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

if len(lb.classes_) == 2:
    labels = to_categorical(labels)

imgtrain, imgtest, labtrain, labtest, boxtrain, boxtest, pathtrain, pathtest = train_test_split(data, labels, bboxes, imagePaths, test_size=0.2, random_state=42, shuffle=True)

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, 'w')
f.write("\n".join(pathtest))
f.close

vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model( inputs=vgg.input, outputs=(bboxHead, softmaxHead))

losses = { "class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error",}

lossWeights = { "class_label": 1.0, "bounding_box": 1.0}

opt = Adam(learning_rate=config.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=[['accuracy'], ['accuracy']], loss_weights=lossWeights)
print(model.summary())

trainTargets = { "class_label": labtrain, "bounding_box": boxtrain}

testTargets = { "class_label": labtest, "bounding_box": boxtest }

print("[INFO] training model...")
H = model.fit( imgtrain, trainTargets, validation_data=(imgtest, testTargets), batch_size=config.BATCH_SIZE, epochs=config.NUM_EPOCHS, verbose=2)

print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

print("[INFO] saving label binarizer...")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, H.history[l], label=l)
	ax[i].plot(N, H.history["val_" + l], label="val_" + l)
	ax[i].legend()
     
plt.tight_layout()
plotPath = os.path.sep.join([config.PLOTS_PATH, "losses.png"])
plt.savefig(plotPath)
plt.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"], label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"], label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

plotPath = os.path.sep.join([config.PLOTS_PATH, "accs.png"])
plt.savefig(plotPath)