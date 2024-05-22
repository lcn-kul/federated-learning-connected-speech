"""Dataset loading script for external transcripts based on huggingface's dataset library."""

from glob import glob

import datasets
from chamd import ChatReader # type: ignore

from constants import EXTERNAL_DIR

# Path to raw dir

# Description
_DESCRIPTION = """\
EXTERNAL data, NOT INTENDED FOR PUBLIC USE.
"""

# Specify the data path
_URLS = {
    "subject_wise_controls": EXTERNAL_DIR,
    "subject_wise_dementia": EXTERNAL_DIR,
    "subject_wise_combined": EXTERNAL_DIR,
}


class ExternalDataset(datasets.GeneratorBasedBuilder):
    """EXTERNAL dataset (for now)."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    # You will be able to load one or the other configurations in the following list with
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="subject_wise_controls",
            version=VERSION,
            description="This dataset building configuration only chooses healthy controls from EXTERNAL",
        ),
        datasets.BuilderConfig(
            name="subject_wise_dementia",
            version=VERSION,
            description="This dataset building configuration only chooses subjects affected by AD from EXTERNAL",
        ),
        datasets.BuilderConfig(
            name="subject_wise_combined",
            version=VERSION,
            description="This dataset building configuration chooses both controls and those affected by AD",
        ),
    ]

    DEFAULT_CONFIG_NAME = "subject_wise_combined"

    def _info(self):
        if self.config.name == "combined":
            features = datasets.Features(
                {
                    "subject_id": datasets.Value("int32"),
                    "sentences": datasets.features.Sequence(datasets.Value("string")),
                    "label": datasets.Value("string"),
                }
            )
        else:
            features = datasets.Features(
                {
                    "subject_id": datasets.Value("int32"),
                    "sentences": datasets.features.Sequence(datasets.Value("string")),
                    "label": datasets.Value("string"),
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same
        # structure with the url replaced with path to local files.
        # By default, the archives will be extracted and a path to a cached
        # folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)
        data_dir = _URLS[self.config.name]
        # As of now, there is only a train split that contains the full dataset
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # Generate examples specific for a subject based on the first three FPACK questions
        # Use a simple counter for unique keys
        key = 0

        reader = ChatReader()

        folder_regex = ""
        
        if self.config.name == "subject_wise_controls":
            folder_regex = "healthy"
        elif self.config.name == "subject_wise_dementia":
            folder_regex = "ad-dementia"
        else:
            folder_regex = "*"

        # Get all the relevant files indicated by the previously defined file indicator
        relevant_files = [f for f in glob(filepath + "/" + folder_regex + "/*")]

        # Create one example per subject ("sentences")
        for f in relevant_files:

            # Read the file
            chat = reader.read_file(f)
            sentences = {}
            parts = []

            # Filter for the participant, exclude the interviewer
            for line in chat.lines:
                if "PAR" in str(line.metadata["speaker"]):
                    parts.append(str(line.text)[:-2] + ".")

            sentences["sentences"] = parts

            key += 1
            # Add subject ID
            sentences["subject_id"] = int(f.split("/")[-1].strip("S").strip(".cha"))
            # Add a label
            sentences["label"] = "ad-dementia" if "ad-dementia" in f else "healthy"

            # Yields subject-specific examples as (key, example) tuples
            yield key, sentences
