

from zerollama.core.config.main import config_setup


def download(model_name=None, repo_id=None, filename=None):
    if model_name is not None:
        repo_id, filename = model_name.split("+")

    config_setup()

    import json
    import fnmatch
    from pathlib import Path

    try:
        from huggingface_hub import hf_hub_download, HfFileSystem
        from huggingface_hub.utils import validate_repo_id
    except ImportError:
        raise ImportError(
            "Llama.from_pretrained requires the huggingface-hub package. "
            "You can install it with `pip install huggingface-hub`."
        )

    validate_repo_id(repo_id)

    hffs = HfFileSystem()

    files = [
        file["name"] if isinstance(file, dict) else file
        for file in hffs.ls(repo_id)
    ]

    # split each file into repo_id, subfolder, filename
    file_list = []
    for file in files:
        rel_path = Path(file).relative_to(repo_id)
        file_list.append(str(rel_path))

    matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

    if len(matching_files) == 0:
        raise ValueError(
            f"No file found in {repo_id} that match {filename}\n\n"
            f"Available Files:\n{json.dumps(file_list)}"
        )

    if len(matching_files) > 1:
        raise ValueError(
            f"Multiple files found in {repo_id} matching {filename}\n\n"
            f"Available Files:\n{json.dumps(files)}"
        )

    (matching_file,) = matching_files

    subfolder = str(Path(matching_file).parent)
    filename = Path(matching_file).name

    # download the file
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )


if __name__ == '__main__':
    for size in ["0.5B", "1.8B", "4B", "7B", "32B"]:
        for k in ["q8_0", "q6_k", "q5_k_m", "q5_0", "q4_k_m", "q4_0", "q3_k_m", "q2_k"]:
            repo_id = f"Qwen/Qwen1.5-{size}-Chat-GGUF"
            filename = f"*{k}.gguf"
            print(repo_id)
            print(filename)
            download(model_name=f"{repo_id}+{filename}")



