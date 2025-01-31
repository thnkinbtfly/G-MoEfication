"""
Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language
https://epub.oeaw.ac.at/0xc1aa5576_0x003a10d2.pdf

The GermEval2018 task is a benchmark to indetify offensive language. This shared
task deals with the classification of German tweets from Twitter. It comprises two tasks,
a coarse-grained binary classification task (OTHER, OFFENSE) and a fine-grained multi-class classification
task(OTHER, INSULT, ABUSE, PROFANITY). This script focuses on the binary (coarse) classification.

Homepage: https://projects.cai.fbi.h-da.de/iggsa/
"""
import datasets
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from functools import partial
import numpy as np

_CITATION = """
@inproceedings{vamvas2020germeval,
    author    = "Wiegand, Michael, and Siegel, Melanie and Ruppenhofer, Josef",
    title     = "Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language",
    booktitle = "Proceedings of the GermEval 2018 Workshop  14th Conference on Natural Language Processing (KONSENS)",
    address   = "Vienna, SAustria",
    year      = "2018",
    month     = "sep",
    url       = "https://epub.oeaw.ac.at/0xc1aa5576_0x003a10d2.pdf"
}"""


def _germeval_agg_precision(key, items):
    references, predictions = zip(*items)
    precision_metric = datasets.load_metric("precision")
    return precision_metric.compute(
        references=references,
        predictions=predictions,
        average="binary",
    )[key]


def _germeval_agg_recall(key, items):
    references, predictions = zip(*items)
    recall_metric = datasets.load_metric("recall")
    return recall_metric.compute(
        references=references,
        predictions=predictions,
        average="binary",
    )[key]


def _germeval_agg_f1(key, items):
    references, predictions = zip(*items)
    f1_metric = datasets.load_metric("f1")
    return f1_metric.compute(
        references=references,
        predictions=predictions,
        average="binary",
    )[key]


def _germeval_fine_agg_precision(key, items):
    references, predictions = zip(*items)
    precision_metric = datasets.load_metric("precision")
    return precision_metric.compute(
        references=references,
        predictions=predictions,
        average="macro",
        # labels=[0,1,2,3],
    )[key]


def _germeval_fine_agg_recall(key, items):
    references, predictions = zip(*items)
    recall_metric = datasets.load_metric("recall")
    return recall_metric.compute(
        references=references,
        predictions=predictions,
        average="macro",
        # labels=[0,1,2,3],
    )[key]


def _germeval_fine_agg_f1(key, items):
    references, predictions = zip(*items)
    f1_metric = datasets.load_metric("f1")
    return f1_metric.compute(
        references=references,
        predictions=predictions,
        average="macro",
        # labels=[0,1,2,3],
    )[key]


class GermEval2018(Task):
    VERSION = 0
    DATASET_PATH = "philschmid/germeval18"
    DATASET_NAME = None

    binary_dict = {
        "OTHER": "Sonstiges",
        "OFFENSE": "Beleidigung",
    }

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

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

    def doc_to_text(self, doc):
        return (
            "Tweet (Beleidigung, Sonstiges): "
            + doc["text"]
            + "\n\n"
            + "Sprache (Beleidigung, Sonstiges): "
        )

    def doc_to_target(self, doc):
        label = doc["binary"]
        target = ""
        if label in self.binary_dict.keys():
            target = self.binary_dict[label]
        else:
            target = ""

        return " " + target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        request_list = []
        for key, value in self.binary_dict.items():
            request_list.append(rf.loglikelihood(ctx, " " + value))

        return tuple(request_list)

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pred = float("-inf")

        scores = [i[0] for i in results]
        pred = scores.index(max(scores))
        binary = doc["binary"]
        true_label = list(self.binary_dict).index(binary)

        return {
            "acc": pred == true_label,
            "precision": (true_label, pred),
            "recall": (true_label, pred),
            "f1": (true_label, pred),
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            "acc": mean,
            "precision": partial(_germeval_agg_precision, "precision"),
            "recall": partial(_germeval_agg_recall, "recall"),
            "f1": partial(_germeval_agg_f1, "f1"),
        }

    def higher_is_better(self):
        return {"acc": True, "precision": True, "recall": True, "f1": True}


class GermEval2018_fine(GermEval2018):
    multi_dict = {
        "OTHER": "Sonstiges",
        "INSULT": "Beleidigung",
        "ABUSE": "Beschimpfung",
        "PROFANITY": "Profanität",
    }

    def doc_to_text(self, doc):
        return (
            "Tweet (Beleidigung, Beschimpfung, Profanität, Sonstiges): "
            + doc["text"]
            + "\n\n"
            + "Sprache (Beleidigung, Beschimpfung, Profanität, Sonstiges): "
        )

    def doc_to_target(self, doc):
        label = doc["multi"]
        target = ""
        if label in self.multi_dict.keys():
            target = self.multi_dict[label]
        else:
            target = ""

        return " " + target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        request_list = []
        for key, value in self.multi_dict.items():
            request_list.append(rf.loglikelihood(ctx, " " + value))

        return tuple(request_list)

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pred = float("-inf")

        scores = [i[0] for i in results]
        pred = scores.index(max(scores))
        multi = doc["multi"]
        true_label = list(self.multi_dict).index(multi)

        return {
            "acc": pred == true_label,
            "precision": (true_label, pred),
            "recall": (true_label, pred),
            "f1": (true_label, pred),
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            "acc": mean,
            "precision": partial(_germeval_fine_agg_precision, "precision"),
            "recall": partial(_germeval_fine_agg_recall, "recall"),
            "f1": partial(_germeval_fine_agg_f1, "f1"),
        }
