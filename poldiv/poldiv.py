"""poldiv dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import tifffile as tiff
import csv
import os
import re

_DESCRIPTION = """The poldiv dataset contains IFC-measured pollen samples from 2018, 2019, 2020 and REF in 102 
classes. The images are R3/R4-gated and depict single in-focus, non-cropped cells (R4) or cells/multiple cells of the 
same species of poor quality that are cropped or polluted (R3). The dataset yields the individual multispectral 
channels and their corresponding default masks that are generated by the Amnis ImageStream Mk II as separate 16-bit 
images with varying width and height."""

# TODO(poldiv): BibTeX citation
_CITATION = """
"""

_DATASET = "poldiv-dataset-2.0.0.tar.gz"

_DATA_OPTIONS = ['all-species','all-genus']


class PoldivConfig(tfds.core.BuilderConfig):
    """BuilderConfig for poldiv dataset."""

    def __init__(self, selection=None, **kwargs):
        """Constructs a PoldivConfig.

        Args:
          selection: `str`, one of `_DATA_OPTIONS`.
          **kwargs: keyword arguments forwarded to super.
        """

        if selection not in _DATA_OPTIONS:
            raise ValueError('Selection must be one of %s' % _DATA_OPTIONS)

        super(PoldivConfig, self).__init__(
            version=tfds.core.Version('2.1.0'),
            release_notes={
                '2.1.0': 'Builder configs for all-species and all-genus',
                '2.0.0': 'Additional Urtica samples',
                '1.0.0': 'Full dataset',
                '0.1.0': 'Initial release.'
            },
            **kwargs)
        self.selection = selection


class Poldiv(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for poldiv dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Place the dataset tar.gz file in the `~/tensorflow_datasets/downloads/manual` dir.
    """

    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        PoldivConfig(name='all-species', selection='all-species', description='All samples without "Others" on species level, channels 1/2/3/4/5/6/9 only'),
        PoldivConfig(name='all-genus', selection='all-genus', description='All samples without "Others" on genus level, channels 1/2/3/4/5/6/9 only')
    ]

    # pytype: enable=wrong-keyword-args

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        if self.builder_config.selection.startswith('all'):
            channels = {str(i + 1): tfds.features.Tensor(dtype=tf.uint16, shape=(None, None), encoding='zlib') for i in range(6)}
            channels['9'] = tfds.features.Tensor(dtype=tf.uint16, shape=(None, None), encoding='zlib')
            masks = {str(i + 1): tfds.features.Tensor(dtype=tf.uint16, shape=(None, None), encoding='zlib') for i in range(6)}
            masks['9'] = tfds.features.Tensor(dtype=tf.uint16, shape=(None, None), encoding='zlib')

            if 'species' in self.builder_config.selection:
                class_names_file = './classes-all-species.txt'

            elif 'genus' in self.builder_config.selection:
                class_names_file = './classes-all-genus.txt'

            else:
                raise Exception(f'Unknows builder config name {self.builder_config.selection}')

            features = {'channels': {**channels},
                        'masks': {**masks},
                        'filename': tf.string,
                        'label': tfds.features.ClassLabel(names_file=class_names_file)}

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            supervised_keys=None,
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path = os.path.join(dl_manager.manual_dir, _DATASET)

        if not tf.io.gfile.exists(path):
            raise AssertionError(
                f'You must download the dataset .tar.gz file and place it into {dl_manager.manual_dir}')

        path_iter = dl_manager.iter_archive(path)

        return {
            'train': self._generate_examples(path_iter),
        }

    def _generate_examples(self, path_iter):
        """Yields examples."""

        path_regex = r'^([a-zA-Z]+\.?[a-zA-Z]+).+$'

        mapping_reader = csv.DictReader(open('./mapping-species-genus.csv'), fieldnames=['species', 'genus'])
        mappings = list(mapping_reader)

        for filename, fobj in path_iter:
            if self.builder_config.selection.startswith('all'):
                m = re.match(path_regex, filename)
                if m is None:
                    raise AssertionError(filename)
                clazz = m.group(1).lower()
                if clazz == 'others':
                    continue

                if 'genus' in self.builder_config.selection:
                    found = next((item for item in mappings if item['species'] == clazz), None)
                    if found is not None:
                        clazz = found['genus']
                    else:
                        raise Exception('Genus not found for {clazz}')

                img = tiff.imread(fobj)
                num_channels = img.shape[-1] / 2  # for 2018 there are 7, 9 or 12 channels

                if num_channels == 7 or num_channels == 9:
                    channels = {str(i + 1): img[:, :, i] for i in range(0, 6)}
                    channels['9'] = img[:, :, 6]
                    masks = {str(i - 6): img[:, :, i] for i in range(7, 13)}
                    masks['9'] = img[:, :, 13]

                elif num_channels == 12:
                    channels = {str(i + 1): img[:, :, i] for i in range(0, 6)}
                    channels['9'] = img[:, :, 8]
                    masks = {str(i - 11): img[:, :, i] for i in range(12, 18)}
                    masks['9'] = img[:, :, 20]

                else:
                    raise AssertionError(f'Unknown number of channels ({num_channels}) for file {filename}')

                features = {
                    'channels': {**channels},
                    'masks': {**masks},
                    'filename': filename,
                    'label': clazz}
            yield filename, features
