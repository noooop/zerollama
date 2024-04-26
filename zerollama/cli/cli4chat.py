
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
@click.option("--wait", default=True)
def start(model_name, wait):
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient

    name = "ZeroInferenceManager"
    manager_client = ZeroManagerClient(name)
    model_class = "zerollama.inference_backend.hf_transformers.main:HuggingFaceTransformersChat"
    model_kwargs = {"model_name": model_name}
    manager_client.start(model_name, model_class, model_kwargs)

    if wait:
        from zerollama.core.framework.inference_engine.client import ChatClient
        client = ChatClient()
        print(f"Wait {model_name} available.")
        client.wait_service_available(model_name)
        print(f"{model_name} available now.")


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
