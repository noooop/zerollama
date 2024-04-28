

class ModelBase(object):
    family = ""
    protocol = ""
    model_kwargs = {}
    header = []
    info = []
    inference_backend = ""

    @classmethod
    def model_names(cls):
        return [x[0] for x in cls.info]

    @classmethod
    def prettytable(cls):
        from prettytable import PrettyTable
        table = PrettyTable(cls.header + ["family", "protocol"], align='l')

        for x in cls.info:
            table.add_row(x+[cls.family, cls.protocol])

        return table

    @classmethod
    def get_model_config(cls, model_name):
        raise NotImplementedError
