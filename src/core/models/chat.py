

class ChatInterfaces(object):
    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def chat(self, messages, options=None):
        """

        :param messages:
        :param options:
        :return:
        """
        raise NotImplementedError

    def stream_chat(self, messages, options=None):
        """

        :param messages:
        :param options:
        :return:
        """
        raise NotImplementedError