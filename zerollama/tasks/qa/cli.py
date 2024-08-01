
import click
from zerollama.tasks.qa.protocol import MANAGER_NAME


@click.command()
def list_families():
    from zerollama.tasks.qa.collection import families
    click.echo("Supported qa families:")
    for name in families():
        click.echo(name)


@click.command()
@click.argument('name')
def list_family(name):
    from zerollama.tasks.qa.collection import family
    c = family(name)
    if c is None:
        click.echo(f"[ {name} ] not support.")
    else:
        click.echo(c.prettytable())


@click.command()
@click.argument('model_name')
def pull(model_name):
    from zerollama.tasks.base.download import download
    from zerollama.tasks.qa.collection import get_model_by_name
    download(model_name, get_model_by_name)


@click.command()
@click.argument('model_name')
@click.option("--wait/--nowait", default=True)
def start(model_name, wait):
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient

    manager_client = ZeroManagerClient(MANAGER_NAME)
    manager_client.start(model_name)

    if not wait:
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
def qa():
    pass


qa.add_command(list_families)
qa.add_command(list_family)
qa.add_command(pull)
qa.add_command(start)
qa.add_command(terminate)


if __name__ == '__main__':
    qa()
