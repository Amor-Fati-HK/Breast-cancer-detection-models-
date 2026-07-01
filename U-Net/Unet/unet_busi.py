# %%
import tensorflow as tf

# %%
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
import os, shutil, pathlib

# %%
import random

# %%
##### Preprocessing ########

# %%
categories=["benign","malignant","normal"]
splits={'train':0.7,'validation':0.2,'test':0.1}

random.seed(1337)
for cat in categories:
    source_cat_dir=base_dir / cat
    all_files=os.listdir(source_cat_dir)
    images=[f for f in all_files if "mask" not in f and f.endswith('.png')]

    random.shuffle(images)

    n=len(images)
    idx_val=int(n*splits['train'])
    idx_test=int(n*(splits['train']+splits['validation']))

    data_split={ 
        'train': images[:idx_val],
        'validation': images[idx_val:idx_test],
        'test': images[idx_test:]
    }
    for phase, file_list in data_split.items():
        input_path=output_dir / phase / 'inputs' / cat
        target_path= output_dir / phase / 'targets' / cat

        os.makedirs(input_path, exist_ok=True)
        os.makedirs(target_path, exist_ok=True)

        for fname in file_list: 
            shutil.copyfile(src=source_cat_dir / fname, dst=input_path / fname)
            mask_name=fname.replace(".png", "_mask.png")
            if os.path.exists(source_cat_dir / mask_name):
                shutil.copyfile(src=source_cat_dir / mask_name, dst=target_path / mask_name)
            else:
                Print(f"Mask Missing for {fname}")


# %%
base_dir=pathlib.Path("archive/Dataset_BUSI_with_GT") #Source doc
output_dir=pathlib.Path("archive/BUSI_Splitted")

# %%
from tensorflow.keras.utils import load_img,img_to_array

# %%
base_path = pathlib.Path("archive/BUSI_Splitted")
input_img_paths = sorted([str(p) for p in (base_path).rglob("inputs/*/*.png")])
target_paths = sorted([str(p) for p in (base_path).rglob("targets/*/*.png")])
print(f"Total images trouvées : {len(input_img_paths)}")
print(f"Total masques trouvés : {len(target_paths)}")

# %%
img_size=(256,256)
num_imgs=len(input_img_paths)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))
def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale"))
    img=(img>127).astype("uint8")
    return img 

input_imgs=np.zeros((num_imgs,)+img_size+(3,),dtype="float32")
targets=np.zeros((num_imgs,)+img_size+(1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i]=path_to_input_image(input_img_paths[i])
    targets[i]=path_to_target(target_paths[i])
num_val_samples=200
train_input_imgs=input_imgs[:-num_val_samples]
train_targets=targets[:-num_val_samples]
val_input_imgs=input_imgs[-num_val_samples:]
val_targets=targets[-num_val_samples:]

# %%
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img

plt.axis("off")
plt.imshow(array_to_img(train_input_imgs[10]))
plt.show()

plt.axis("off")
plt.imshow(array_to_img(train_targets[10]), cmap="gray")
plt.show()

# %%
##### Metrics ########

# %%
from tensorflow.keras import backend as K
def dice_coef(y_true,y_pred,smooth=1e-6):
    # we flat out tensor
    y_true_f=K.flatten(K.cast(y_true,'float32'))
    y_pred_f=K.flatten(y_pred)

    #calculate intersection
    intersection=K.sum(y_true_f*y_pred_f)
    return (2.*intersection+smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)
def dice_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)

# %%
def specificity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def f1_score(y_true, y_pred):
    p = keras.metrics.Precision()(y_true, y_pred)
    r = keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

# %%
def bce_dice_loss(y_true, y_pred):
    # BCE classique
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Dice Loss
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    # On additionne les deux !
    return bce + dice_loss

# %%
##### Model ########

# %%
def unet_model(img_size, num_classes):
    inputs=keras.Input(shape=img_size+(3,))
    x=keras.layers.Rescaling(1./255)(inputs)
    c1=keras.layers.Conv2D(16,3,activation="relu", kernel_initializer="he_normal", padding="same")(x)
    c1=keras.layers.Dropout(0.1)(c1)
    c1=keras.layers.Conv2D(16,3,activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1=keras.layers.MaxPooling2D((2,2))(c1)

    c2=keras.layers.Conv2D(32,3,activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2=keras.layers.Dropout(0.1)(c2)
    c2=keras.layers.Conv2D(32,3,activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2=keras.layers.MaxPooling2D((2,2))(c2)

    c3=keras.layers.Conv2D(64,3,activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3=keras.layers.Dropout(0.2)(c3)
    c3=keras.layers.Conv2D(64,3,activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3=keras.layers.MaxPooling2D((2,2))(c3)

    c4=keras.layers.Conv2D(128,3,activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4=keras.layers.Dropout(0.2)(c4)
    c4=keras.layers.Conv2D(128,3,activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4=keras.layers.MaxPooling2D((2,2))(c4)

    c5=keras.layers.Conv2D(256,3,activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5=keras.layers.Dropout(0.3)(c5)
    c5=keras.layers.Conv2D(256,3,activation="relu", kernel_initializer="he_normal", padding="same")(c5)

    u6=keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c5)
    u6=keras.layers.concatenate([u6,c4])
    c6=keras.layers.Conv2D(128,3,activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6=keras.layers.Dropout(0.2)(c6)
    c6=keras.layers.Conv2D(128,3,activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    u7=keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding="same")(c6)
    u7=keras.layers.concatenate([u7,c3])
    c7=keras.layers.Conv2D(64,3,activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7=keras.layers.Dropout(0.2)(c7)
    c7=keras.layers.Conv2D(64,3,activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    u8=keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding="same")(c7)
    u8=keras.layers.concatenate([u8,c2])
    c8=keras.layers.Conv2D(32,3,activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8=keras.layers.Dropout(0.1)(c8)
    c8=keras.layers.Conv2D(32,3,activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    u9=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding="same")(c8)
    u9=keras.layers.concatenate([u9,c1])
    c9=keras.layers.Conv2D(16,3,activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9=keras.layers.Dropout(0.1)(c9)
    c9=keras.layers.Conv2D(16,3,activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    outputs=keras.layers.Conv2D(1,(1,1),activation="sigmoid")(c9)
    model=keras.Model(inputs,outputs)
    return model

model=unet_model(img_size,1)
model.compile(optimizer="adam", loss=bce_dice_loss, metrics=[
    "accuracy",
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'), 
    specificity, 
    dice_coef])
model.summary()

# %%
from tensorflow.keras import backend as K
def dice_coef(y_true,y_pred,smooth=1e-6):
    # we flat out tensor
    y_true_f=K.flatten(K.cast(y_true,'float32'))
    y_pred_f=K.flatten(y_pred)

    #calculate intersection
    intersection=K.sum(y_true_f*y_pred_f)
    return (2.*intersection+smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)
def dice_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)

# %%
##### Training ########

# %%
callbacks=[
    keras.callbacks.ModelCheckpoint("BUSI_UNet.keras",
                                    save_best_only=True),
    keras.callbacks.EarlyStopping(
        patience=10, monitor="val_loss"),
    
    keras.callbacks.ReduceLROnPlateau(
        factor=0.2, patience=5, monitor="val_loss")
]
history=model.fit(train_input_imgs,train_targets,
                  epochs=50,
                  batch_size=32,
                  validation_data=(val_input_imgs, val_targets),
                  callbacks=callbacks)

# %%
##### Visualisation ########

# %%
dice = history.history["dice_coef"]
val_dice = history.history["val_dice_coef"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(dice) + 1)
plt.plot(epochs, dice, "bo", label="Training dice")
plt.plot(epochs, val_dice, "b", label="Validation dice")
plt.title("Training and validation dice")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# %%
i = 5
test_image = val_input_imgs[i]
true_mask = val_targets[i] 

pred = model.predict(np.expand_dims(test_image, 0))[0]
pred_mask = (pred > 0.5).astype("uint8")


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.title("Échographie")
plt.axis("off")
plt.imshow(array_to_img(test_image))


plt.subplot(1, 3, 2)
plt.title("Masque Original")
plt.axis("off")
plt.imshow(true_mask.squeeze(), cmap="gray")


plt.subplot(1, 3, 3)
plt.title("Prédiction IA")
plt.axis("off")
plt.imshow(pred_mask.squeeze() * 255, cmap="gray")

plt.show()

# %%
###### Evaluation #######

# %%
model.evaluate(val_input_imgs, val_targets)


