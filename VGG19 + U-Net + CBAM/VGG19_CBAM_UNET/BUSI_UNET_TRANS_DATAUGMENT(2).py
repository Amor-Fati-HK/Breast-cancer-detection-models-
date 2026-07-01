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
base_dir=pathlib.Path("archive/Dataset_BUSI_with_GT") 
output_dir=pathlib.Path("archive/BUSI_Splitted")

# %%
base_path = pathlib.Path("archive/BUSI_Splitted")
input_img_paths = sorted([str(p) for p in (base_path).rglob("inputs/*/*.png")])
target_paths = sorted([str(p) for p in (base_path).rglob("targets/*/*.png")])
print(f"Total images trouvées : {len(input_img_paths)}")
print(f"Total masques trouvés : {len(target_paths)}")

# %%
base_path = pathlib.Path("archive/BUSI_Splitted")
input_img_paths = sorted([str(p) for p in (base_path).rglob("inputs/*/*.png")])
target_paths = sorted([str(p) for p in (base_path).rglob("targets/*/*.png")])
print(f"Total images trouvées : {len(input_img_paths)}")
print(f"Total masques trouvés : {len(target_paths)}")

# %%
from tensorflow.keras.utils import load_img,img_to_array
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
from tensorflow.keras.applications.vgg19 import preprocess_input

def simple_augment(image, mask):
    seed = tf.random.uniform([2], maxval=10000, dtype=tf.int32)
    
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=seed)

    angles = tf.random.stateless_uniform([], seed=seed, minval=-0.2, maxval=0.2)
    image = tf.image.rot90(image, k=tf.cast(tf.random.stateless_uniform([], seed=seed, minval=0, maxval=4, dtype=tf.int32), tf.int32))
    mask = tf.image.rot90(mask, k=tf.cast(tf.random.stateless_uniform([], seed=seed, minval=0, maxval=4, dtype=tf.int32), tf.int32))

    image = tf.image.stateless_random_brightness(image, max_delta=0.05, seed=seed)
    image = tf.image.stateless_random_contrast(image, lower=0.95, upper=1.05, seed=seed)

    do_sobel = tf.random.stateless_uniform([], seed=seed, minval=0, maxval=1)
    if do_sobel > 0.7:
        image_gray = tf.image.rgb_to_grayscale(image)
        sobel = tf.image.sobel_edges(tf.expand_dims(image_gray, 0))
        sobel_mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))
        sobel_mag = tf.squeeze(sobel_mag, axis=0) 
        sobel_mag = (sobel_mag / (tf.reduce_max(sobel_mag) + 1e-7)) * 255.0
        image = tf.concat([sobel_mag, sobel_mag, sobel_mag], axis=-1)

    
    
    image = tf.keras.applications.vgg19.preprocess_input(image)
    mask = tf.cast(mask, tf.float32) 
    return image, mask

def simple_preprocess_val(image, mask):
    image = preprocess_input(image)
    mask = tf.cast(mask, tf.float32)
    return image, mask

# %%
train_ds = tf.data.Dataset.from_tensor_slices((train_input_imgs, train_targets))
val_ds = tf.data.Dataset.from_tensor_slices((val_input_imgs, val_targets))

train_ds = (train_ds
            .shuffle(len(train_input_imgs))
            .map(simple_augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(8) 
            .prefetch(tf.data.AUTOTUNE))

val_ds = (val_ds
          .map(simple_preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(8)
          .prefetch(tf.data.AUTOTUNE))

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
def channel_attention_module(x, ratio=8):
    channels=x.shape[-1]
    shared_layer_one=keras.layers.Dense(channels // ratio, activation="relu", use_bias=False)
    shared_layer_two=keras.layers.Dense(channels, use_bias=False)

    # avgpool 
    avg_pool=keras.layers.GlobalAveragePooling2D()(x)
    avg_pool=keras.layers.Reshape((1,1,channels))(avg_pool)
    avg_out=shared_layer_two(shared_layer_one(avg_pool))

    max_pool=keras.layers.GlobalMaxPooling2D()(x)
    max_pool=keras.layers.Reshape((1,1,channels))(max_pool)
    max_out=shared_layer_two(shared_layer_one(max_pool))

    cbam_feature=keras.layers.Add()([avg_out,max_out])
    cbam_feature=keras.layers.Activation('sigmoid')(cbam_feature)

    return keras.layers.multiply([x,cbam_feature])

# %%
from keras import ops
def spatial_attention_module(x):
    #
    avg_pool = ops.mean(x,axis=-1,keepdims=True)
    max_pool= ops.max(x,axis=-1,keepdims=True)

    #concatenation
    concat=keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])

    #7x7 filter and sigmoid
    cbam_feature=keras.layers.Conv2D(
        filters=1, kernel_size=7, strides=1,
        padding="same", activation="sigmoid", use_bias=False
    )(concat)

    #muliply and return
    return keras.layers.multiply([x,cbam_feature])

# %%
def cbam_block(x,ratio=8):
    x=channel_attention_module(x,ratio)
    x=spatial_attention_module(x)
    return x

# %%
model = keras.models.load_model("BUSI_VGG19_trams_Unet.keras")

# %%
model = keras.models.load_model(
    "BUSI_VGG19_trams_Unet.keras", 
    custom_objects={
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "log_cosh_dice_loss": log_cosh_dice_loss,
        "cbam_block": cbam_block 
    }
)

# %%
def total_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl

# %%
def hybrid_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return (0.95 * dice_loss(y_true, y_pred)) + (0.05 * tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred)))

# %%
def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log(tf.math.cosh(x))

# %%
custom_dict = {
    "dice_loss": dice_loss,
    "dice_coef": dice_coef,
    "specificity": specificity
}

# %%
model = keras.models.load_model("BUSI_VGG19_trams_Unet.keras", custom_objects=custom_dict, compile=False)

# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=log_cosh_dice_loss,
    metrics=["accuracy", 
             keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall'), 
             specificity, 
             dice_coef]
)

# %%
callbacks=[
    keras.callbacks.ModelCheckpoint("BUSI_VGG19_trams_Unet_data_aug.keras",
                                    save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", 
                                  patience=10, restore_best_weights=True),
    
    keras.callbacks.ReduceLROnPlateau(
        factor=0.2, patience=5, monitor="val_loss")
]
history=model.fit(train_ds, 
                  validation_data=val_ds,
                  epochs=50,
                  callbacks=callbacks)

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
model.load_weights("BUSI_VGG19_trams_Unet_data_aug.keras")
for layer in model.layers:
    if "block5" in layer.name or "block4" in layer.name:
        layer.trainable = True
    else:
        
        if "block" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=log_cosh_dice_loss,
    metrics=["accuracy", 
             keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall'), 
             specificity, 
             dice_coef]
)

# %%
callbacks=[
    keras.callbacks.ModelCheckpoint("BUSI_VGG19_trams_Unet_data_aug_FT.keras",
                                    save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", 
                                  patience=10, restore_best_weights=True),
    
    keras.callbacks.ReduceLROnPlateau(
        factor=0.2, patience=5, monitor="val_loss")
]
history=model.fit(train_ds, 
                  validation_data=val_ds,
                  epochs=50,
                  callbacks=callbacks)

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
model.evaluate(val_ds)

# %%
model= keras.models.load_model("/home/bigbro/Bureau/Projet_PFE_models/BUSI_VGG19_trams_Unet_data_aug_FT.keras", custom_objects=custom_dict, compile=False)
from tensorflow.keras.utils import array_to_img
i = 10
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



