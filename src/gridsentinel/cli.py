import typer

app = typer.Typer(help="GridSentinel CLI")


@app.command()
def simulate(days: int = 30, seed: int = 42) -> None:
    """Generate synthetic telemetry + ticket streams."""
    from gridsentinel.simulator.emit import run

    run(days=days, seed=seed)


@app.command()
def report(week: str | None = None) -> None:
    """Render the weekly auto-written ops report."""
    from gridsentinel.reporting.render import render_weekly

    path = render_weekly(week=week)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
