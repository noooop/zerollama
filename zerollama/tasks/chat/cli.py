
import click
from zerollama.tasks.chat.protocol import MANAGER_NAME


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
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient

    manager_client = ZeroManagerClient(MANAGER_NAME)
    manager_client.start(model_name)

    if nowait:
        return
    print(f"Wait {model_name} available.")

    manager_client.wait_service_status(model_name)


@click.command()
@click.argument('model_name')
def terminate(model_name):
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient
    manager_client = ZeroManagerClient(MANAGER_NAME)
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
