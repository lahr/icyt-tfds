"""romania dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import tifffile as tiff
import os
import re

_DESCRIPTION = """"""

# TODO(romania): BibTeX citation
_CITATION = """
"""

_DATA_OPTIONS = ['all']


class RomaniaConfig(tfds.core.BuilderConfig):
    """BuilderConfig for romania dataset."""

    def __init__(self, dataset=None, selection=None, **kwargs):
        """Constructs a RomaniaConfig.

        Args:
          selection: `str`, one of `_DATA_OPTIONS`.
          **kwargs: keyword arguments forwarded to super.
        """

        if selection not in _DATA_OPTIONS:
            raise ValueError('Selection must be one of %s' % _DATA_OPTIONS)

        super(RomaniaConfig, self).__init__(
            version=tfds.core.Version('1.0.0'),
            release_notes={
                '1.0.0': 'Full dataset'
            },
            **kwargs)
        self.selection = selection
        self.dataset = dataset


class Romania(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for romania dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Place the dataset tar.gz file in the `~/tensorflow_datasets/downloads/manual` dir.
    """

    # pytype: disable=wrong-keyword-args
    BUILDER_CONFIGS = [
        RomaniaConfig(name='all', selection='all', dataset="romania-train-1.0.0.tar.gz", description='All training samples')
    ]

    # pytype: enable=wrong-keyword-args
    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        channels = {str(i + 1): tfds.features.Tensor(dtype=tf.uint16, shape=(None, None), encoding='zlib') for i in range(12)}
        masks = {str(i + 1): tfds.features.Tensor(dtype=tf.uint16, shape=(None, None), encoding='zlib') for i in range(12)}

        features = {'channels': {**channels},
                    'masks': {**masks},
                    'filename': tf.string,
                    'species': tfds.features.ClassLabel(names_file=f'romania/classes-{self.builder_config.selection}-species.txt')}

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            supervised_keys=None,
            homepage='https://github.com/lahr/icyt-tfds',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path = os.path.join(dl_manager.manual_dir, self.builder_config.dataset)

        if not tf.io.gfile.exists(path):
            raise AssertionError(
                f'You must download the dataset .tar.gz file and place it into {dl_manager.manual_dir}')

        if self.builder_config.selection == 'all':
            path_iter = dl_manager.iter_archive(path)
            return {
                'train': self._generate_examples(path_iter)
            }

    def _generate_examples(self, path_iter, split_name=None):
        """Yields examples."""

        path_regex = r'^(?:([^/\n.A-Z]+)/)?([a-zA-Z]+\.?[a-zA-Z]+).*$'

        for filename, fobj in path_iter:
            assert filename is not None
            assert fobj is not None

            m = re.match(path_regex, filename)

            species = m.group(2).lower()

            img = tiff.imread(fobj)
            num_channels = img.shape[-1] / 2

            if num_channels == 12:
                channels = {str(i + 1): img[:, :, i] for i in range(0, 12)}
                masks = {str(i - 11): img[:, :, i] for i in range(12, 24)}

            else:
                raise AssertionError(f'Unknown number of channels ({num_channels}) for file {filename}')

            features = {
                'channels': {**channels},
                'masks': {**masks},
                'filename': filename,
                'species': species}

            yield filename, features
