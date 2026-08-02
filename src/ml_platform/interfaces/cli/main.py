"""Platform command-line entry point."""

from pathlib import Path

import typer

from ml_platform.application.train_model import train
from ml_platform.domain.config import load_training_config
from ml_platform.domain.errors import PlatformError

app = typer.Typer(
    name="ml-platform",
    help="Configuration-driven machine learning training and serving platform.",
    no_args_is_help=True,
)


@app.command()
def train_model(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Path to a YAML training configuration."),
) -> None:
    try:
        result = train(load_training_config(config))
    except PlatformError as error:
        typer.echo(f"Training failed: {error}", err=True)
        if error.details:
            typer.echo(f"Details: {error.details}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Training completed. Run ID: {result.run_id}")
    typer.echo(f"Artifact path: {result.artifact_path}")
    typer.echo(f"Validation metrics: {result.metrics}")
