
base = "zerollama.microservices.retriever_database.backend"


BACKEND = [
    ["BM25sRetrieverDatabase", f"{base}.use_bm25s"],
]

BACKEND_MAP = {x[0]: x[1] for x in BACKEND}


def get_backend_by_name(name):
    return BACKEND_MAP.get(name, None)


