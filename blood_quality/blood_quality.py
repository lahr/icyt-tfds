"""blood_quality dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import tifffile as tiff
import os
import re
import collections

_DESCRIPTION = """"""

# TODO(blood_quality): BibTeX citation
_CITATION = """
"""

_DATA_OPTIONS = ['canadian', 'swiss']

class BloodQualityConfig(tfds.core.BuilderConfig):
    """BuilderConfig for blood_quality dataset."""

    def __init__(self, dataset=None, selection=None, **kwargs):
        """Constructs a BloodQualityConfig.

        Args:
          selection: `str`, one of `_DATA_OPTIONS`.
          **kwargs: keyword arguments forwarded to super.
        """

        if selection not in _DATA_OPTIONS:
            raise ValueError('Selection must be one of %s' % _DATA_OPTIONS)

        super(BloodQualityConfig, self).__init__(
            version=tfds.core.Version('1.0.0'),
            release_notes={
                '1.0.0': 'Initial release.'
            },
            **kwargs)
        self.selection = selection
        self.dataset = dataset

class BloodQuality(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for blood_quality dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Place the dataset zip file in the `~/tensorflow_datasets/downloads/manual` dir.
    """

  # pytype: disable=wrong-keyword-args
  BUILDER_CONFIGS = [
      BloodQualityConfig(name='canadian', selection='canadian', dataset="Training_Test2_Canada_Swiss.zip", description='Canadian samples'),
      BloodQualityConfig(name='swiss', selection='swiss', dataset="Training_Test2_Canada_Swiss.zip", description='Swiss samples')
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    channels = {str(i): tfds.features.Tensor(dtype=tf.uint8, shape=(None, None), encoding='zlib') for i in [1,9,12]}

    features = {'channels': {**channels},
                'filename': tf.string,
                'morphology': tfds.features.ClassLabel(names_file=f'blood_quality/classes.txt')}

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
    # TODO(blood_quality): Downloads the data and defines the splits
    path = os.path.join(dl_manager.manual_dir, self.builder_config.dataset)

    if not tf.io.gfile.exists(path):
        raise AssertionError(
            f'You must download the dataset .zip file and place it into {dl_manager.manual_dir}')

    path_iter = dl_manager.iter_archive(path)
    return {
        'train': self._generate_examples(path_iter)
    }

  def _generate_examples(self, path_iter):
        """Yields examples."""
        path_regex = fr'^.*/{self.builder_config.selection.title()}.*/.*/.*/(.*)/(.*)_Ch(\d+)\.ome\.tif$'
        d = collections.defaultdict(dict)
        for filename, fobj in path_iter:
            assert filename is not None
            assert fobj is not None

            m = re.match(path_regex, filename)
            if m:
              morphology = m.group(1)
              if morphology == 'Side' or morphology == 'Undecidable':
                continue
              basename = m.group(2)
              channel = m.group(3)
              d[basename][channel] = tiff.imread(fobj)

              if len(d[basename]) == 3:
                channels = d[basename]
                d.pop(basename)

                features = {
                    'channels': {**channels},
                    'filename': filename,
                    'morphology': morphology}

                yield filename, features
