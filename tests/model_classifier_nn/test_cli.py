import json

from click.testing import CliRunner

from birdclef.workflows.classify_nn import fit


def test_model_fit(
    monkeypatch, tmp_path, train_root, label_encoder, model_checkpoint, z_dim
):
    monkeypatch.chdir(tmp_path)
    filter_set = tmp_path / "filter_set.json"
    filter_set.write_text(
        json.dumps([x for x in label_encoder.classes_ if x != "noise"])
    )

    runner = CliRunner().invoke(
        fit,
        [
            f"--dataset-dir={train_root}",
            f"--embedding-checkpoint={model_checkpoint}",
            f"--dim={z_dim}",
            f"--filter-set={filter_set}",
            f"--parallelism=2",
        ],
        catch_exceptions=True,
    )
    assert runner.exit_code == 0, runner.output
