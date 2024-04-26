
from zerollama.cli.cli4chat import click, chat


@click.command()
def run_server():
    click.echo("Control-C terminate")

    from applications.chat_webui.server import setup, run
    server = setup()
    run(server)


chat.add_command(run_server)


if __name__ == '__main__':
    chat()
