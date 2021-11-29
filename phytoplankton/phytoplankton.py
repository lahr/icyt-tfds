"""phytoplankton dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import tifffile as tiff
import os
import re

# TODO(phytoplankton): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(phytoplankton): BibTeX citation
_CITATION = """
"""

_DATA_OPTIONS = ['rep-0', 'rep-1']


class PhytoplanktonConfig(tfds.core.BuilderConfig):
    """BuilderConfig for pythoplankton dataset."""

    def __init__(self, dataset=None, selection=None, **kwargs):
        """Constructs a PhytoplanktonConfig.

      Args:
        selection: `str`, one of `_DATA_OPTIONS`.
        **kwargs: keyword arguments forwarded to super.
      """

        if selection not in _DATA_OPTIONS:
            raise ValueError('Selection must be one of %s' % _DATA_OPTIONS)

        super(PhytoplanktonConfig, self).__init__(
            version=tfds.core.Version('1.0.0'),
            release_notes={
                '1.0.0': 'Full dataset'
            },
            **kwargs)
        self.selection = selection
        self.dataset = dataset


class Phytoplankton(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for phytoplankton dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place the dataset tar.gz file in the `~/tensorflow_datasets/downloads/manual` dir.
  """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        PhytoplanktonConfig(name='rep-0', selection='rep-0', dataset="phytoplankton-1.0.0.tar.gz", description='rep-0'),
        PhytoplanktonConfig(name='rep-1', selection='rep-1', dataset="phytoplankton-1.0.0.tar.gz", description='rep-1')
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        channels = {str(i): tfds.features.Tensor(dtype=tf.uint16, shape=(None, None), encoding='zlib') for i in
                    range(1, 13)}
        features = {'channels': {**channels},
                    'filename': tf.string,
                    'species': tfds.features.ClassLabel(names_file='phytoplankton/classes.txt')}

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

        return {
            'train': self._generate_examples(dl_manager.iter_archive(path), self.builder_config.selection, 'train'),
            'valid': self._generate_examples(dl_manager.iter_archive(path), self.builder_config.selection,
                                             'validation'),
            'test': self._generate_examples(dl_manager.iter_archive(path), self.builder_config.selection, 'test')
        }

    def _generate_examples(self, path_iter, rep, split_name=None):
        """Yields examples."""

        path_regex = r'^(rep-\d)/(train|validation|test)/\d{8}_\d{2}_(\w)_\d+.*$'

        for filename, fobj in path_iter:
            assert filename is not None
            assert fobj is not None

            m = re.match(path_regex, filename)

            if m.group(1) != rep or m.group(2) != split_name:
                continue

            species = m.group(3)

            img = tiff.imread(fobj)
            num_channels = img.shape[-1]

            if num_channels == 12:
                channels = {str(i): img[:, :, i - 1] for i in range(1, 13)}

            else:
                raise AssertionError(f'Unknown number of channels ({num_channels}) for file {filename}')

            features = {
                'channels': {**channels},
                'filename': filename,
                'species': species
            }

            yield filename, features
