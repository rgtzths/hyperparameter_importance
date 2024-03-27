import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist, cifar100
from textblob import TextBlob
from sklearn.model_selection import train_test_split
import cv2

from datasets import load_dataset

'''
    Loads a csv file in the format feature1, feature2, ..., label.
    Divides it into 80% for training and 20% for validation.
    Returns ((x_train, y_train), (x_val, y_val))
'''
def load_dataset_from_file(file_location): 

    if file_location.split(".")[-1] != "csv":
        raise ValueError("Dataset should be a csv file.")

    data = pd.read_csv(file_location, index_col=False)

    train_length = int(data.shape[0]*0.8)

    x_train = data.iloc[:train_length,:data.shape[1]-1].values
    y_train = data.iloc[:train_length, data.shape[1]-1].values

    x_val = data.iloc[train_length: , :data.shape[1]-1].values
    y_val = data.iloc[train_length: , data.shape[1]-1].values

    return (x_train, y_train), (x_val, y_val)

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Parse numbers as floats
    x_train = x_train.astype('float32')

    # Normalize data
    x_train = x_train / 255

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Parse numbers as floats
    x_train = x_train.astype('float32')

    # Normalize data
    x_train = x_train / 255

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    return (x_train, y_train), (x_val, y_val)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')

    # Normalize data
    x_train = x_train / 255

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    return (x_train, y_train), (x_val, y_val)

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32')

    # Normalize data
    x_train = x_train / 255

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    return (x_train, y_train), (x_val, y_val)

def load_covertype(): #Tabular classification categoric_numeric (high number of examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_cat/covertype.csv", split="train")
    dataset = dataset.to_pandas().values

    x = dataset[:,:-1]
    y = dataset[:,-1] -1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def load_higgs(): #Tabular classification numeric (high number of examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_num/Higgs.csv", split="train")
    dataset = dataset.to_pandas().values

    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def load_compas(): #Tabular classification categoric (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="clf_cat/compas-two-years.csv", split="train")
    dataset = dataset.to_pandas().values
    
    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def load_delays_zurich(): #Tabular regression numeric
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_num/delays_zurich_transport.csv", split="train")
    dataset = dataset.to_pandas().values
    
    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def load_abalone(): #Tabular regression mixture (low examples)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/abalone.csv", split="train")
    dataset = dataset.to_pandas().values
    
    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def load_bike_sharing(): #Tabular regression mixture (numerous examples, more cat then reg)
    dataset = load_dataset("inria-soda/tabular-benchmark",  data_files="reg_cat/Bike_Sharing_Demand.csv", split="train")
    dataset = dataset.to_pandas().values

    x = dataset[:,:-1]
    y = dataset[:,-1] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def load_license_plate():

    def preprocess(sample):
        image = sample["image"]
        image = np.array(image)
        target_size = 224
        img_shape = image.shape
        bbox = sample["objects"]["bbox"][0]
        # resize the image
        image = cv2.resize(image, (target_size, target_size))
        # normalize the image
        image = image / 255.0
        # update the bounding box
        bbox = [
            int(bbox[0] * (target_size / img_shape[1])),
            int(bbox[1] * (target_size / img_shape[0])),
            int(bbox[2] * (target_size / img_shape[1])),
            int(bbox[3] * (target_size / img_shape[0])),
        ]
        return image, bbox

    ds = load_dataset("keremberke/license-plate-object-detection", name="full")

    train_x = []
    train_y = []
    for sample in ds["train"]:
        image, bbox = preprocess(sample)
        train_x.append(image)
        train_y.append(bbox)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    val_x = []
    val_y = []
    for sample in ds["validation"]:
        image, bbox = preprocess(sample)
        val_x.append(image)
        val_y.append(bbox)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    return (train_x, train_y), (val_x, val_y)

def load_utk_faces():
    dataset = load_dataset("nlphuji/utk_faces")["test"]
    
    images = []
    ages = []
    for sample in dataset:
        images.append(np.array(sample["image"]))
        ages.append(sample["age"])

    images = np.array(images)
    ages = np.array(ages)

    x_train, x_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_val, y_val)

def load_mri():

    def preprocess(sample):
        image = sample["image"]
        image = np.array(image)
        target_size = 224
        img_shape = image.shape
        bbox = sample["objects"]["bbox"][0]
        # resize the image
        image = cv2.resize(image, (target_size, target_size))
        # normalize the image
        image = image / 255.0
        # update the bounding box
        bbox = [
            int(bbox[0] * (target_size / img_shape[1])),
            int(bbox[1] * (target_size / img_shape[0])),
            int(bbox[2] * (target_size / img_shape[1])),
            int(bbox[3] * (target_size / img_shape[0])),
        ]
        return image, bbox
    
    ds = load_dataset("Francesco/abdomen-mri")
    
    train_x = []
    train_y = []
    for sample in ds["train"]:
        #image, bbox = preprocess(sample)
        train_x.append(sample["image"])
        train_y.append(sample["objects"]["bbox"][0])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(train_x.shape)
    val_x = []
    val_y = []
    for sample in ds["validation"]:
        #image, bbox = preprocess(sample)
        val_x.append(sample["image"])
        val_y.append(sample["objects"]["bbox"][0])
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    return (train_x, train_y), (val_x, val_y)