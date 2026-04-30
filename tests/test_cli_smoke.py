from typer.testing import CliRunner

from embodied_data.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "embodied-data" in result.stdout


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.stdout
    assert "validate" in result.stdout
    assert "preview" in result.stdout
