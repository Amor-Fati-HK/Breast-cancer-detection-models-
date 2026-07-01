
import pandas as pd
import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import glob
import warnings
import csv
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# %%
import tensorflow as tf
from tensorflow import keras

# %% [markdown]
# ### Preprocessing

# %%
BASE_INPUT_PATH = "/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
CSV_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "csv")
IMAGE_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "jpeg")
BASE_OUTPUT_PATH = "/kaggle/working/"

# %%
IMAGE_SIZE = 256
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
RANDOM_SEED = 42


# %%
def find_image_in_folder(folder_path):
    """
    Trouve la première image .jpg ou .png dans le dossier ET ses sous-dossiers.
    """
    if not folder_path or not os.path.isdir(folder_path):
        return None
        
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                return os.path.join(root, file)
                
    return None

def compute_all_bounding_boxes(mask_path, min_area=100):
    """
    Returns a list of bounding boxes [[x_min, y_min, width, height],...]
    Returns None if mask doesn't exist or is invalid
    """
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append([x, y, w, h])

    return boxes if boxes else None

def build_metadata_lookup(dicom_info_path, jpeg_base_dir, *args):
    print(f"Building metadata lookup from: {dicom_info_path}")
    master_map = {}
    
    try:
        dicom_info = pd.read_csv(dicom_info_path, dtype=str)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {dicom_info_path}")
        return master_map

    valid_descriptions = {arg for arg in args}
    filtered_df = dicom_info[dicom_info['SeriesDescription'].isin(valid_descriptions)]

    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Building lookup map"):
        series_desc = row['SeriesDescription'] 
        patient_id_composite = row['PatientID'] 
        
        full_path = None
        
        # 1. On essaye de construire le chemin depuis 'image_path'
        if 'image_path' in row and pd.notna(row['image_path']):
            rel_path = row['image_path']
            if 'jpeg' in rel_path:
                clean_rel_path = rel_path.split('jpeg')[-1].strip("/\\")
                tmp_path = os.path.join(jpeg_base_dir, clean_rel_path)
            else:
                tmp_path = os.path.join(jpeg_base_dir, rel_path)

            if os.path.exists(tmp_path):
                full_path = tmp_path

        if not full_path:
            series_uid = row['SeriesInstanceUID']
            folder_path = os.path.join(jpeg_base_dir, series_uid)
            full_path = find_image_in_folder(folder_path)
            
        if not full_path or not os.path.exists(full_path): 
            continue

        if patient_id_composite not in master_map:
            master_map[patient_id_composite] = {}
            
        master_map[patient_id_composite][series_desc] = full_path
            
    print(f"Metadata lookup map built. Found {len(master_map)} unique composite keys.")
    return master_map


# %%
def BuildMasterDataset(MASTER_LIST_PATH="/kaggle/working/master_dataset.csv", 
                       argument1="cropped images", 
                       argument2="ROI mask images"):
    BASE_INPUT_PATH = "/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
    IMAGE_FOLDER_PATH = os.path.join(BASE_INPUT_PATH, "jpeg")
    DICOM_INFO_PATH = os.path.join(BASE_INPUT_PATH, "csv/dicom_info.csv")
    
    INPUT_CSVS = [
        os.path.join(BASE_INPUT_PATH, "csv/mass_case_description_train_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/mass_case_description_test_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/calc_case_description_train_set.csv"),
        os.path.join(BASE_INPUT_PATH, "csv/calc_case_description_test_set.csv")
    ]

    master_map = build_metadata_lookup(DICOM_INFO_PATH, IMAGE_FOLDER_PATH, argument1, argument2)
    
    if not master_map:
        return

    found_pairs_count = 0
    missing_mask_count = 0
    skipped_malignant_count = 0
    
    with open(MASTER_LIST_PATH, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow([
            'cropped_image_path', 'roi_mask_path',
            'x_min', 'y_min', 'width', 'height',
            'pathology', 'assessment', 'patient_id', 'series_type', 'mask_status',
            'breast_density', 'abnormality_shape', 'abnormality_margin', 'subtlety'
        ])

        for filepath in INPUT_CSVS:
            filename = os.path.basename(filepath)
            if not filepath or not os.path.exists(filepath):
                continue
            
            is_mass = "mass" in filename.lower()
            
            if is_mass:
                type_prefix = "Mass"
            else:
                type_prefix = "Calc"
                
            if "train" in filename.lower():
                split_prefix = "Training"
            else:
                split_prefix = "Test"
                
            full_prefix = f"{type_prefix}-{split_prefix}"

            with open(filepath, "r") as infile:
                csv_reader = csv.reader(infile)
                header = next(csv_reader)
                
                pathology_idx = header.index('pathology')
                assessment_idx = header.index('assessment')
                patient_id_idx = header.index('patient_id')
                breast_idx = header.index('left or right breast')
                view_idx = header.index('image view')
                abnormality_id_idx = header.index('abnormality id')
                
                if 'breast density' in header:
                    density_idx = header.index('breast density')
                else:
                    density_idx = header.index('breast_density')
                
                subtlety_idx = header.index('subtlety')
                
                if is_mass:
                    shape_idx = header.index('mass shape') if 'mass shape' in header else header.index('mass_shape')
                    margin_idx = header.index('mass margins') if 'mass margins' in header else header.index('mass_margins')
                else:
                    shape_idx = header.index('calc type') if 'calc type' in header else header.index('calc_type')
                    margin_idx = header.index('calc distribution') if 'calc distribution' in header else header.index('calc_distribution')

                for row in tqdm(csv_reader, desc=f"Processing {filename}"):
                    if not any(row):
                        continue
                    
                    pathology = row[pathology_idx]
                    assessment = row[assessment_idx]
                    patient_id = row[patient_id_idx]
                    side = row[breast_idx]
                    view = row[view_idx]
                    abn_id = row[abnormality_id_idx]
                    
                    density = row[density_idx]
                    subtlety = row[subtlety_idx]
                    abn_shape = row[shape_idx]
                    abn_margin = row[margin_idx]
                    
                    try:
                        abn_id_clean = str(int(float(abn_id)))
                    except ValueError:
                        abn_id_clean = str(abn_id).strip()

                    composite_key = f"{full_prefix}_{patient_id}_{side}_{view}_{abn_id_clean}"
                    
                    study_data = master_map.get(composite_key)
                    if not study_data:
                        continue

                    full_crop_path = study_data.get('cropped images')
                    full_mask_path = study_data.get('ROI mask images')
                    
                    if not full_crop_path:
                        continue
                    
                    mask_status = 'valid'
                    
                    if not full_mask_path:
                        pathology_upper = str(pathology).upper()
                        is_benign = 'BENIGN' in pathology_upper and 'MALIGNANT' not in pathology_upper
                        
                        if is_benign:
                            full_mask_path = 'n/a'
                            mask_status = 'n/a'
                            missing_mask_count += 1
                            
                            csv_writer.writerow([
                                full_crop_path, 'n/a',
                                'n/a', 'n/a', 'n/a', 'n/a',
                                pathology, assessment, patient_id, full_prefix, 'n/a',
                                density, abn_shape, abn_margin, subtlety
                            ])
                            found_pairs_count += 1
                        else:
                            skipped_malignant_count += 1
                        continue
                    
                    if full_crop_path == full_mask_path and mask_status == 'valid':
                        continue
                    
                    boxes = compute_all_bounding_boxes(full_mask_path, min_area=100)
                     
                    if boxes is None:
                        csv_writer.writerow([
                            full_crop_path, full_mask_path,
                            'n/a', 'n/a', 'n/a', 'n/a',
                            pathology, assessment, patient_id, full_prefix, mask_status,
                            density, abn_shape, abn_margin, subtlety
                        ])
                        found_pairs_count += 1
                    else:
                        for (x_min, y_min, width, height) in boxes:
                            csv_writer.writerow([
                                full_crop_path, full_mask_path,
                                x_min, y_min, width, height,
                                pathology, assessment, patient_id, full_prefix, mask_status,
                                density, abn_shape, abn_margin, subtlety
                            ])
                            found_pairs_count += 1

    print(f"\n{'='*60}")
    print(f"DATASET BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"Master list saved to: {MASTER_LIST_PATH}")
    print(f"Valid pairs found: {found_pairs_count}")
    print(f"Benign cases without masks (n/a): {missing_mask_count}")
    print(f"Malignant cases skipped (no mask): {skipped_malignant_count}")
    print(f"{'='*60}")

BuildMasterDataset()

# %%
MASTER_LIST_PATH = "/kaggle/working/master_dataset.csv"

# Load the master list (prevent 'n/a' from being read as NaN)
df_master = pd.read_csv(MASTER_LIST_PATH, keep_default_na=False)

# Get unique list of patient IDs and split them for training and validation
unique_patients = df_master["patient_id"].unique()
train_patients, val_patients = train_test_split(
    unique_patients,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED
)

print(f"Splitting {len(unique_patients)} unique patients: {len(train_patients)} for training, {len(val_patients)} for validation.")

# %%
train_df = df_master[df_master['patient_id'].isin(train_patients)].reset_index(drop=True)
val_df = df_master[df_master['patient_id'].isin(val_patients)].reset_index(drop=True)
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.GridDistortion(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
   
])

val_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
])

# %%
class MultimodalSequence(tf.keras.utils.Sequence):
    def __init__(self, dataframe, clinical_dict, batch_size=16, img_size=256, transforms=None):
        self.df = dataframe
        self.clinical_dict = clinical_dict
        self.batch_size = batch_size
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        imgs, msks, embs = [], [], []

        for _, row in batch_df.iterrows():
            image, mask = self._load_data(row)
            
            p_id = str(row['patient_id'])
            embedding = self.clinical_dict.get(p_id, np.zeros(768))
            
            imgs.append(image)
            msks.append(mask)
            embs.append(embedding)

        return (np.array(imgs), np.array(embs)), np.array(msks)

    def _load_data(self, row):
        
        image_path = row["cropped_image_path"]
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))

        mask_path = row["roi_mask_path"]
        if row["mask_status"] == 'n/a' or mask_path == 'n/a':
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if full_mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                # Secure data
                x_min_val = row['x_min']
                y_min_val = row['y_min']
                
                if x_min_val in ['n/a', ''] or pd.isna(x_min_val):
                    x_min = 0
                    y_min = 0
                    x_max = full_mask.shape[1]
                    y_max = full_mask.shape[0]
                else:
                    x_min = max(0, int(float(x_min_val)))
                    y_min = max(0, int(float(y_min_val)))
                    x_max = min(full_mask.shape[1], x_min + int(float(row['width'])))
                    y_max = min(full_mask.shape[0], y_min + int(float(row['height'])))
                
                mask_crop = full_mask[y_min:y_max, x_min:x_max]
                
                if mask_crop.size == 0:
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                else:
                    mask = cv2.resize(mask_crop, (self.img_size, self.img_size))

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
            
        return image, mask

# %%
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from transformers import pipeline

pipe = pipeline("feature-extraction", 
                model="emilyalsentzer/Bio_ClinicalBERT", 
                framework="tf",
                device=-1)

def get_embedding_with_pipe(text):
    features = pipe(text)
    feat_array = np.array(features[0])
    return np.mean(feat_array, axis=0)

print("Reading Master Dataset...")
df_master = pd.read_csv("/kaggle/working/master_dataset.csv", keep_default_na=False)

def combine_features_master(row):
    pathology = str(row['pathology']).replace('_', ' ').lower()
    mask_status = str(row['mask_status']).strip().lower()
    
    if mask_status == 'n/a' or pathology in ['normal', 'nan', '']:
        return "Normal breast tissue, no mass or calcification detected."
    
    shape = str(row['abnormality_shape']).replace('_', ' ').lower()
    margin = str(row['abnormality_margin']).replace('_', ' ').lower()
    
    if shape in ['nan', '']: shape = "irregular"
    if margin in ['nan', '']: margin = "spiculated"
    
    is_mass = "mass" in str(row['series_type']).lower()
    lesion_type = "mass" if is_mass else "calcification"
    
    return f"A {pathology} {lesion_type} with {shape} shape and {margin} margins."

print("Generating training embeddings...")

unique_patients_df = df_master.drop_duplicates(subset=['patient_id'])

clinical_map = {}
for _, row in tqdm(unique_patients_df.iterrows(), total=len(unique_patients_df), desc="Generating Embeddings"):
    text = combine_features_master(row)
    vec = get_embedding_with_pipe(text)
    
    p_id = str(row['patient_id'])
    clinical_map[p_id] = vec

with open("clinical_embeddings_dict_v2.pkl", "wb") as f:
    pickle.dump(clinical_map, f)

print(f"\nDictionary saved! {len(clinical_map)} unique patients processed.")

unique_patients_master = set(df_master["patient_id"].unique())
unique_patients_dict = set(clinical_map.keys())
missing = unique_patients_master - unique_patients_dict

print(f"\n--- VERIFICATION REPORT ---")
print(f"Unique patients in CSV: {len(unique_patients_master)}")
print(f"Patients in dictionary: {len(unique_patients_dict)}")
if len(missing) == 0:
    print(" SUCCESS: 100% of patients are covered by the embedding dictionary!")
else:
    print(f" WARNING: {len(missing)} patients are missing from the dictionary!")

# %%

import pickle
with open("clinical_embeddings_dict_v2.pkl", "rb") as f:
    clinical_dict = pickle.load(f)

train_generator = MultimodalSequence(
    dataframe=train_df, 
    clinical_dict=clinical_dict,
    batch_size=BATCH_SIZE, 
    img_size=IMAGE_SIZE,
    transforms=train_transforms
)

val_generator = MultimodalSequence(
    dataframe=val_df, 
    clinical_dict=clinical_dict,
    batch_size=BATCH_SIZE, 
    img_size=IMAGE_SIZE,
    transforms=val_transforms
)

print(f" Générateurs Multimodaux prêts !")
print(f"Échantillons : Train={len(train_df)} | Val={len(val_df)}")

# %% [markdown]
# ### Metrics

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

# %% [markdown]
# ### Model

# %%
def channel_attention_module(x, ratio=8):
    channels=x.shape[-1]
    shared_layer_one=keras.layers.Dense(channels // ratio, activation="relu", use_bias=False)
    shared_layer_two=keras.layers.Dense(channels, use_bias=False)

    # avgpool 
    avg_pool=keras.layers.GlobalAveragePooling2D()(x)
    avg_pool=keras.layers.Reshape((1,1,channels))(avg_pool)
    avg_out=shared_layer_two(shared_layer_one(avg_pool))

    #maxpool
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
def AE_module(skip_connection, bert_vector, channels):
    
    gate = keras.layers.Dense(channels, activation="sigmoid")(bert_vector)
    gate = keras.layers.Reshape((1, 1, channels))(gate)
    x = keras.layers.Multiply()([skip_connection, gate])

    x = cbam_block(x) 
    
    return x

# %%
def cross_attention_gate(img_features, bert_vector, channels):
    """
    Vraie Cross-Attention (Query-Key-Value)
    Image = Query | Texte = Key & Value
    """
    bert_quiet = keras.layers.Dropout(0.8)(bert_vector)
    bert_norm = keras.layers.BatchNormalization()(bert_quiet)

    query = keras.layers.Conv2D(channels, 1, padding='same')(img_features)

    key = keras.layers.Dense(channels)(bert_norm)
    key = keras.layers.Reshape((1, 1, channels))(key)
    
    value = keras.layers.Dense(channels)(bert_norm)
    value = keras.layers.Reshape((1, 1, channels))(value)

    attention_map = keras.layers.Multiply()([query, key])
    
    attention_map = keras.layers.Activation('sigmoid')(attention_map) 

    out = keras.layers.Multiply()([attention_map, value])
    
    return keras.layers.Add()([img_features, out])

# %%
def VGG19_unet_Multi_CBERT(img_size):
    img_inputs = keras.Input(shape=img_size + (3,), name="image_input")
    bert_inputs = keras.Input(shape=(768,), name="bert_input")
    
    base_model = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(256,256,3),
        input_tensor=img_inputs
    )
    base_model.trainable = False

    s1 = base_model.get_layer("block1_conv2").output
    s2 = base_model.get_layer("block2_conv2").output
    s3 = base_model.get_layer("block3_conv2").output
    s4 = base_model.get_layer("block4_conv2").output
    b1 = base_model.get_layer("block5_conv4").output
    b1=cbam_block(b1)
    b1_fused = cross_attention_gate(b1, bert_inputs, 512)

    s4_f = cross_attention_gate(s4, bert_inputs, 512)
    s3_f = cross_attention_gate(s3, bert_inputs, 256)
    s2_f = s2
    s1_f = s1


    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(b1_fused)
    u6 = keras.layers.concatenate([u6, s4_f])
    c6 = keras.layers.Conv2D(128, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = keras.layers.Dropout(0.5)(c6)
    c6 = keras.layers.Conv2D(128, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    
    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = keras.layers.concatenate([u7, s3_f])
    c7 = keras.layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = keras.layers.Dropout(0.4)(c7)
    c7 = keras.layers.Conv2D(64, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    
    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = keras.layers.concatenate([u8, s2_f])
    c8 = keras.layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = keras.layers.Dropout(0.3)(c8)
    c8 = keras.layers.Conv2D(32, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    
    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = keras.layers.concatenate([u9, s1_f])
    c9 = keras.layers.Conv2D(16, 3, activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = keras.layers.Dropout(0.2)(c9)
    c9 = keras.layers.Conv2D(16, 3, activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    outputs=keras.layers.Conv2D(1,(1,1), activation="sigmoid")(c9)
    keras.backend.clear_session()
    model = keras.Model(inputs=[img_inputs, bert_inputs], outputs=outputs)
    return model

keras.backend.clear_session()
model=VGG19_unet_Multi_CBERT((256,256))
model.summary()

# %%
def bce_dice_loss(y_true, y_pred):
    # BCE 
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Dice Loss
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    return bce + dice_loss

# %%
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_multimodal_model_CBERT.keras", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=["accuracy", 
             keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall'), 
             specificity, 
             dice_coef] 
)

# %% [markdown]
# ### Training

# %%
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)

# %% [markdown]
# ### Visualization

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
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_pred(idx):
    IMAGE_SIZE = 256 
    
    img_path_to_test = val_df['cropped_image_path'].iloc[idx]
    msk_path_to_test = val_df['roi_mask_path'].iloc[idx]
    p_id = str(val_df['patient_id'].iloc[idx])
    
    raw_img = cv2.imread(img_path_to_test)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (IMAGE_SIZE, IMAGE_SIZE))
    img_norm = raw_img.astype(np.float32) / 255.0 
    
    raw_msk = cv2.imread(msk_path_to_test, cv2.IMREAD_GRAYSCALE)
    x_min = max(0, int(float(val_df['x_min'].iloc[idx])))
    y_min = max(0, int(float(val_df['y_min'].iloc[idx])))
    x_max = min(raw_msk.shape[1], x_min + int(float(val_df['width'].iloc[idx])))
    y_max = min(raw_msk.shape[0], y_min + int(float(val_df['height'].iloc[idx])))
    
    mask_crop = raw_msk[y_min:y_max, x_min:x_max]
    mask_res = cv2.resize(mask_crop, (IMAGE_SIZE, IMAGE_SIZE))
    _, mask_final = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)
    
    
    transformed = val_transforms(image=raw_img)
    img_input = np.expand_dims(transformed["image"], axis=0)
    
    clinical_vector = clinical_dict.get(p_id, np.zeros(768))
    txt_input = np.expand_dims(clinical_vector, axis=0)
    
    preds = model.predict([img_input, txt_input])
    pred_final = preds[0].squeeze()
    
    pred_mask = (pred_final > 0.5).astype("uint8")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.title(f"Original Image ({val_df['pathology'].iloc[idx]})")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_final, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask * 255, cmap="gray")
    plt.title(f"Prediction IA (Max confidence: {np.max(pred_final):.2f})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

show_pred(4)

# %% [markdown]
# ### Training (Fine tune)

# %%
for layer in model.layers:
    if "block5" in layer.name or "block4" in layer.name:
        layer.trainable = True
    else:
        
        if "block" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True

# %%
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_multimodal_model_CBERT_FT.keras", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=bce_dice_loss,
    metrics=["accuracy", 
             keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall'), 
             specificity, 
             dice_coef]
)

# %%
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks
)

# %% [markdown]
# ### Visualization

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
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_pred(idx):
    IMAGE_SIZE = 256 
    
    img_path_to_test = val_df['cropped_image_path'].iloc[idx]
    msk_path_to_test = val_df['roi_mask_path'].iloc[idx]
    p_id = str(val_df['patient_id'].iloc[idx])
    
    raw_img = cv2.imread(img_path_to_test)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (IMAGE_SIZE, IMAGE_SIZE))
    img_norm = raw_img.astype(np.float32) / 255.0 
    
    raw_msk = cv2.imread(msk_path_to_test, cv2.IMREAD_GRAYSCALE)
    x_min = max(0, int(float(val_df['x_min'].iloc[idx])))
    y_min = max(0, int(float(val_df['y_min'].iloc[idx])))
    x_max = min(raw_msk.shape[1], x_min + int(float(val_df['width'].iloc[idx])))
    y_max = min(raw_msk.shape[0], y_min + int(float(val_df['height'].iloc[idx])))
    
    mask_crop = raw_msk[y_min:y_max, x_min:x_max]
    mask_res = cv2.resize(mask_crop, (IMAGE_SIZE, IMAGE_SIZE))
    _, mask_final = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)
    
    
    transformed = val_transforms(image=raw_img)
    img_input = np.expand_dims(transformed["image"], axis=0)
    
    clinical_vector = clinical_dict.get(p_id, np.zeros(768))
    txt_input = np.expand_dims(clinical_vector, axis=0)
    
    preds = model.predict([img_input, txt_input])
    pred_final = preds[0].squeeze()
    
    pred_mask = (pred_final > 0.5).astype("uint8")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.title(f"Original Image ({val_df['pathology'].iloc[idx]})")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_final, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask * 255, cmap="gray")
    plt.title(f"Prediction IA (Max confidence: {np.max(pred_final):.2f})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

show_pred(4)

# %% [markdown]
# ### Evaluation

# %%
model.evaluate(val_generator)


