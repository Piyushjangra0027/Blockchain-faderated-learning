import tensorflow as tf
import numpy as np
import pickle
import tensorflow_datasets as tfds
print("ok")
def get_mnist():
    '''
    Function to get the MNIST dataset from the TensorFlow Datasets library.
    '''

    
    # Load the MNIST dataset
    mnist, info = tfds.load('mnist', with_info=True, as_supervised=True)
    
    # Extract the train and test datasets
    train_dataset, test_dataset = mnist['train'], mnist['test']
    
    # Convert datasets to numpy arrays
    def dataset_to_numpy(dataset):
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy().reshape(-1))
            labels.append(tf.keras.utils.to_categorical(label.numpy(), 10))
        return np.array(images), np.array(labels)
    
    train_images, train_labels = dataset_to_numpy(train_dataset)
    test_images, test_labels = dataset_to_numpy(test_dataset)
    
    dataset = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels
    }
    
    return dataset


def save_data(dataset,name="mnist.d"):
    '''
    Func to save mnist data in binary mode(its good to use binary mode)
    '''
    with open(name,"wb") as f:
        pickle.dump(dataset,f)

def load_data(name="mnist.d"):
    '''
    Func to load mnist data in binary mode(for reading also binary mode is important)
    ''' 
    with open(name,"rb") as f:
        return pickle.load(f)

# federated_data_extractor.py

def get_dataset_details(dataset):
    '''
    Func to display information on data
    '''
    if dataset is None:
        raise ValueError("Dataset is None.")
    
    required_keys = ['train_images', 'train_labels', 'test_images', 'test_labels']
    for key in required_keys:
        if key not in dataset:
            raise KeyError(f"Key {key} is missing in the dataset.")
    
    data_size = len(dataset['train_images']) if 'train_images' in dataset else 0
    
    # Flatten the train_labels to ensure it's a 1D list
    train_labels = dataset['train_labels']
    if isinstance(train_labels, np.ndarray):
        train_labels = train_labels.flatten()
    elif isinstance(train_labels, list):
        train_labels = [item for sublist in train_labels for item in sublist]
    
    num_classes = len(set(train_labels)) if 'train_labels' in dataset else 0
    
    for k in dataset.keys():
        print(f"{k}: {dataset[k].shape}")
    
    return data_size, num_classes


def split_dataset(dataset,split_count):
    '''
    Function to split dataset to federated data slices as per specified count so as to try federated learning
    '''
    datasets = []
    split_data_length = len(dataset["train_images"])//split_count
    for i in range(split_count):
        d = dict()
        d["test_images"] = dataset["test_images"][:]
        d["test_labels"] = dataset["test_labels"][:]
        d["train_images"] = dataset["train_images"][i*split_data_length:(i+1)*split_data_length]
        d["train_labels"] = dataset["train_labels"][i*split_data_length:(i+1)*split_data_length]
        datasets.append(d)
    return datasets


if __name__ == '__main__':
    save_data(get_mnist())
    dataset = load_data()
    get_dataset_details(dataset)
    for n,d in enumerate(split_dataset(dataset,2)):
        save_data(d,"federated_data_"+str(n)+".d")
        dk = load_data("federated_data_"+str(n)+".d")
        get_dataset_details(dk)
        print()