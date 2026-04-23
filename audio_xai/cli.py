"""Console script for audio_xai."""

import typer
from rich.console import Console
from rich.table import Table

from audio_xai import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Show a quick status of core project directories."""
    status_rows = utils.get_project_paths_status()

    table = Table(title="Audio-XAI project status")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="white")
    table.add_column("Exists")

    for name, path, exists in status_rows:
        table.add_row(name, str(path), "✅" if exists else "❌")

    console.print(table)


if __name__ == "__main__":
    app()
