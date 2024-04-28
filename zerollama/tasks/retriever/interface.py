from pydantic import BaseModel, ConfigDict
from typing import Optional

from zerollama.tasks.base.interface import ModelBase


class Retriever(ModelBase):
    protocol = "retriever"


class RetrieverInterface(object):
    protocol = "retriever"

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def encode(self, sentences, **options):
        """

        :param sentences:
        :param options:
        :return:
        """
        raise NotImplementedError

