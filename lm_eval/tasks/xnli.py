"""
XNLI is an evaluation corpus for language transfer and cross-lingual sentence classification in 15 languages.
https://arxiv.org/abs/1809.05053
Homepage: None, Repo: https://github.com/facebookresearch/XNLI
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{conneau2018xnli,
  title={XNLI: Evaluating Cross-lingual Sentence Representations},
  author={Conneau, Alexis and Rinott, Ruty and Lample, Guillaume and Williams, Adina and Bowman, Samuel and Schwenk, Holger and Stoyanov, Veselin},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={2475--2485},
  year={2018}
}
}"""


class XNLI(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "xnli"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class XNLIEn(XNLI):
    DATASET_NAME = "en"


class XNLIFr(XNLI):
    DATASET_NAME = "fr"


XNLI_TASKS = [
    XNLIEn,
    XNLIFr,
]


def construct_tasks() -> typing.Dict[str, XNLI]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "GEM/wiki_lingua_ar"
    will dispatch to the GEM WikiLingua Arabic class.
    """
    tasks = {}
    for task_class in XNLI_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks
