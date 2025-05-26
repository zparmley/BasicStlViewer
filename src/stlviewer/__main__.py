from pathlib import Path
from typing_extensions import Annotated

import typer

from .gui.main import GUI


def main(stls: Annotated[list[Path] | None, typer.Argument()] = None, debug: bool = False):
    gui = GUI(*stls if stls else ())
    gui.start(debug=debug)

if __name__ == '__main__':
    typer.run(main)
