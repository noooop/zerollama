

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
        info_dict = {x[0]: {k: v for k, v in zip(cls.header, x)} for x in cls.info}

        if model_name not in info_dict:
            return None

        info = info_dict[model_name]
        info.update({"family": cls.family, "protocol": cls.protocol})

        model_config = {
            "name": model_name,
            "info": info,
            "family": cls.family,
            "protocol": cls.protocol,
            "model_kwargs": cls.model_kwargs}

        return model_config
