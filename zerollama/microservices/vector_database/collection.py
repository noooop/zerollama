
base = "zerollama.microservices.vector_database.backend"


BACKEND = [
    ["BruteForceVectorDatabase", f"{base}.use_bruteforce"],
    ["ChromadbVectorDatabase", f"{base}.use_chromadb"],
    ["FaissVectorDatabase", f"{base}.use_chromadb"],
    ["HnswlibVectorDatabase", f"{base}.use_hnswlib"],
]

BACKEND_MAP = {x[0]: x[1] for x in BACKEND}


def get_backend_by_name(name):
    return BACKEND_MAP.get(name, None)


