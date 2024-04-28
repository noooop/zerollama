
import click


@click.command()
def list_families():
    from zerollama.models.collection.chat import chat_families
    click.echo("Supported chat families:")
    for name in chat_families():
        click.echo(name)


@click.command()
@click.argument('name')
def list_family(name):
    from zerollama.models.collection.chat import chat_family
    c = chat_family(name)
    if c is None:
        click.echo(f"[ {name} ] not support.")
    else:
        click.echo(c.prettytable())


@click.command()
@click.argument('model_name')
def pull(model_name):
    from zerollama.inference_backend.hf_transformers.download import download
    download(model_name)


@click.command()
@click.argument('model_name')
@click.option("--nowait", default=False)
def start(model_name, nowait):
    import time
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient

    name = "ZeroInferenceManager"
    manager_client = ZeroManagerClient(name)
    model_class = "zerollama.inference_backend.hf_transformers.main:HuggingFaceTransformersChat"
    model_kwargs = {"model_name": model_name}
    manager_client.start(model_name, model_class, model_kwargs)

    if nowait:
        return
    print(f"Wait {model_name} available.")

    for i in range(100):
        time.sleep(0.5)
        rep = manager_client.status(model_name)
        if rep.state == "error":
            continue

        state = rep.msg["status"]
        exception = rep.msg["exception"]
        if state in ["prepare", "started"]:
            print(f"{model_name} {state}.")
        elif state in ["error"]:
            print(f"{model_name} {state}. {exception}.")
            return
        elif state in ["running"]:
            print(f"{model_name} available now.")
            return


@click.command()
@click.argument('model_name')
def terminate(model_name):
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient
    name = "ZeroInferenceManager"
    manager_client = ZeroManagerClient(name)
    manager_client.terminate(model_name)


@click.group()
def chat():
    pass


chat.add_command(list_families)
chat.add_command(list_family)
chat.add_command(pull)
chat.add_command(start)
chat.add_command(terminate)


if __name__ == '__main__':
    chat()
