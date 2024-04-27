
from zerollama.cli.cli4chat import click, chat


@click.command()
def run():
    click.echo("Control-C terminate")

    from applications.chat_webui.server import setup, run
    server = setup()
    run(server)


@click.group()
def server():
    pass


server.add_command(run)


@click.group()
def main():
    pass


main.add_command(chat)
main.add_command(server)


if __name__ == '__main__':
    main()
