# icyt-tfds
[TensorFlow Datasets](https://www.tensorflow.org/datasets) for the iCyt platform.

## poldiv/all
All samples without "Others", channels 1/2/3/4/5/6/9 only

### Installation
1. Request access to the dataset through the [UFZ data research portal](https://www.ufz.de/record/dmp/archive/11592/).
2. Copy the `.tar.gz` dataset to `~/tensorflow_datasets/downloads/manual/`.
3. Clone this repository and execute `tfds build --config all poldiv`. The installation can take several hours.

### Usage
```python
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_validation, ds_test), ds_info = tfds.load('poldiv/all:2.4.0', split=['train[:80%]','train[80%:90%]','train[90%:]'], shuffle_files=True, with_info=True)
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_validation, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)
print(ds_info.features)
```
Output:
```
FeaturesDict({
    'channels': FeaturesDict({
        '1': Tensor(shape=(None, None), dtype=tf.uint16),
        '2': Tensor(shape=(None, None), dtype=tf.uint16),
        '3': Tensor(shape=(None, None), dtype=tf.uint16),
        '4': Tensor(shape=(None, None), dtype=tf.uint16),
        '5': Tensor(shape=(None, None), dtype=tf.uint16),
        '6': Tensor(shape=(None, None), dtype=tf.uint16),
        '9': Tensor(shape=(None, None), dtype=tf.uint16),
    }),
    'filename': tf.string,
    'genus': ClassLabel(shape=(), dtype=tf.int64, num_classes=53),
    'masks': FeaturesDict({
        '1': Tensor(shape=(None, None), dtype=tf.uint16),
        '2': Tensor(shape=(None, None), dtype=tf.uint16),
        '3': Tensor(shape=(None, None), dtype=tf.uint16),
        '4': Tensor(shape=(None, None), dtype=tf.uint16),
        '5': Tensor(shape=(None, None), dtype=tf.uint16),
        '6': Tensor(shape=(None, None), dtype=tf.uint16),
        '9': Tensor(shape=(None, None), dtype=tf.uint16),
    }),
    'species': ClassLabel(shape=(), dtype=tf.int64, num_classes=102),
})
```

## poldiv_balanced
Balanced dataset in 12 classes with stratified train/validation/test splits, channels 1/2/3/4/5/6/9 only.

### Installation
1. Download the dataset on the [UFZ data research portal](https://www.ufz.de/record/dmp/archive/12200).
2. Copy the `.tar.gz` dataset to `~/tensorflow_datasets/downloads/manual/`.
3. Clone this repository and execute `tfds build poldiv_balanced`.

### Usage
```python
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_validation, ds_test), ds_info = tfds.load('poldiv_balanced:3.0.0', split=['train','valid','test'], with_info=True)
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_validation, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)
print(ds_info.features)
```
Output:
```
FeaturesDict({
    'channels': FeaturesDict({
        '1': Tensor(shape=(None, None), dtype=tf.uint16),
        '2': Tensor(shape=(None, None), dtype=tf.uint16),
        '3': Tensor(shape=(None, None), dtype=tf.uint16),
        '4': Tensor(shape=(None, None), dtype=tf.uint16),
        '5': Tensor(shape=(None, None), dtype=tf.uint16),
        '6': Tensor(shape=(None, None), dtype=tf.uint16),
        '9': Tensor(shape=(None, None), dtype=tf.uint16),
    }),
    'filename': tf.string,
    'genus': ClassLabel(shape=(), dtype=tf.int64, num_classes=7),
    'masks': FeaturesDict({
        '1': Tensor(shape=(None, None), dtype=tf.uint16),
        '2': Tensor(shape=(None, None), dtype=tf.uint16),
        '3': Tensor(shape=(None, None), dtype=tf.uint16),
        '4': Tensor(shape=(None, None), dtype=tf.uint16),
        '5': Tensor(shape=(None, None), dtype=tf.uint16),
        '6': Tensor(shape=(None, None), dtype=tf.uint16),
        '9': Tensor(shape=(None, None), dtype=tf.uint16),
    }),
    'species': ClassLabel(shape=(), dtype=tf.int64, num_classes=12),
})
```

## phytoplankton/rep-{0,1}
Phytoplankton dataset.

### Installation
1. Copy the provided `.tar.gz` dataset to `~/tensorflow_datasets/downloads/manual/`.
2. Clone this repository and execute `tfds build --config rep-0 phytoplankton`. 
3. Execute the command with `rep-1` to create the second dataset.

### Usage
```python
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_validation, ds_test), ds_info = tfds.load('phytoplankton/rep-0:1.0.0', split=['train','valid','test'], shuffle_files=True, with_info=True)
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_validation, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)
print(ds_info.features)
```
Output:
```
FeaturesDict({
    'channels': FeaturesDict({
        '1': Tensor(shape=(None, None), dtype=tf.uint16),
        '10': Tensor(shape=(None, None), dtype=tf.uint16),
        '11': Tensor(shape=(None, None), dtype=tf.uint16),
        '12': Tensor(shape=(None, None), dtype=tf.uint16),
        '2': Tensor(shape=(None, None), dtype=tf.uint16),
        '3': Tensor(shape=(None, None), dtype=tf.uint16),
        '4': Tensor(shape=(None, None), dtype=tf.uint16),
        '5': Tensor(shape=(None, None), dtype=tf.uint16),
        '6': Tensor(shape=(None, None), dtype=tf.uint16),
        '7': Tensor(shape=(None, None), dtype=tf.uint16),
        '8': Tensor(shape=(None, None), dtype=tf.uint16),
        '9': Tensor(shape=(None, None), dtype=tf.uint16),
    }),
    'filename': tf.string,
    'species': ClassLabel(shape=(), dtype=tf.int64, num_classes=6),
})
```

## romania/all

### Installation
1. Request access to the dataset through the [UFZ data research portal](https://www.ufz.de/record/dmp/archive/11920/).
2. Copy the provided `.tar.gz` dataset to `~/tensorflow_datasets/downloads/manual/`.
3. Clone this repository and execute `tfds build --config all romania`. 

### Usage
```python
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_validation, ds_test), ds_info = tfds.load('romania/all:1.0.0', split=['train[:80%]','train[80%:90%]','train[90%:]'], with_info=True)
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_validation, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)
print(ds_info.features)
```
Output:
```
FeaturesDict({
        'channels': FeaturesDict({
            '1': Tensor(shape=(None, None), dtype=tf.uint16),
            '10': Tensor(shape=(None, None), dtype=tf.uint16),
            '11': Tensor(shape=(None, None), dtype=tf.uint16),
            '12': Tensor(shape=(None, None), dtype=tf.uint16),
            '2': Tensor(shape=(None, None), dtype=tf.uint16),
            '3': Tensor(shape=(None, None), dtype=tf.uint16),
            '4': Tensor(shape=(None, None), dtype=tf.uint16),
            '5': Tensor(shape=(None, None), dtype=tf.uint16),
            '6': Tensor(shape=(None, None), dtype=tf.uint16),
            '7': Tensor(shape=(None, None), dtype=tf.uint16),
            '8': Tensor(shape=(None, None), dtype=tf.uint16),
            '9': Tensor(shape=(None, None), dtype=tf.uint16),
        }),
        'filename': tf.string,
        'masks': FeaturesDict({
            '1': Tensor(shape=(None, None), dtype=tf.uint16),
            '10': Tensor(shape=(None, None), dtype=tf.uint16),
            '11': Tensor(shape=(None, None), dtype=tf.uint16),
            '12': Tensor(shape=(None, None), dtype=tf.uint16),
            '2': Tensor(shape=(None, None), dtype=tf.uint16),
            '3': Tensor(shape=(None, None), dtype=tf.uint16),
            '4': Tensor(shape=(None, None), dtype=tf.uint16),
            '5': Tensor(shape=(None, None), dtype=tf.uint16),
            '6': Tensor(shape=(None, None), dtype=tf.uint16),
            '7': Tensor(shape=(None, None), dtype=tf.uint16),
            '8': Tensor(shape=(None, None), dtype=tf.uint16),
            '9': Tensor(shape=(None, None), dtype=tf.uint16),
        }),
        'species': ClassLabel(shape=(), dtype=tf.int64, num_classes=104),
    })
````
