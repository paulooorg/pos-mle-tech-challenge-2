"""Testes da linhagem do run (hash do dado e commit do código)."""

from pathlib import Path

from pos_mle_tech_challenge_2.training.lineage import (
    RunLineage,
    read_data_md5,
    read_git_commit,
)

COMMIT = "a" * 40


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_no_git_directory_gives_no_commit(tmp_path: Path) -> None:
    assert read_git_commit(tmp_path) is None


def test_commit_is_read_from_a_loose_branch_ref(tmp_path: Path) -> None:
    _write(tmp_path / ".git/HEAD", "ref: refs/heads/main\n")
    _write(tmp_path / ".git/refs/heads/main", f"{COMMIT}\n")

    assert read_git_commit(tmp_path) == COMMIT


def test_commit_is_read_from_packed_refs(tmp_path: Path) -> None:
    _write(tmp_path / ".git/HEAD", "ref: refs/heads/main\n")
    _write(
        tmp_path / ".git/packed-refs",
        f"# pack-refs\n{COMMIT} refs/heads/main\n{'b' * 40} refs/heads/dev\n",
    )

    assert read_git_commit(tmp_path) == COMMIT


def test_detached_head_is_returned_as_is(tmp_path: Path) -> None:
    _write(tmp_path / ".git/HEAD", f"{COMMIT}\n")

    assert read_git_commit(tmp_path) == COMMIT


def test_unresolvable_ref_gives_no_commit(tmp_path: Path) -> None:
    _write(tmp_path / ".git/HEAD", "ref: refs/heads/ghost\n")

    assert read_git_commit(tmp_path) is None


def test_data_md5_is_read_from_the_dvc_pointer(tmp_path: Path) -> None:
    pointer = _write(
        tmp_path / "data.csv.dvc",
        "outs:\n- md5: abc123\n  size: 10\n  path: data.csv\n",
    )

    assert read_data_md5(pointer) == "abc123"
    assert read_data_md5(tmp_path / "missing.dvc") is None


def test_lineage_omits_unavailable_values(tmp_path: Path) -> None:
    assert RunLineage(tmp_path, tmp_path / "missing.dvc").collect() == {}


def test_lineage_collects_data_and_code_tags(tmp_path: Path) -> None:
    _write(tmp_path / ".git/HEAD", f"{COMMIT}\n")
    pointer = _write(tmp_path / "d.dvc", "outs:\n- md5: abc123\n")

    tags = RunLineage(tmp_path, pointer).collect()

    assert tags == {"data_md5": "abc123", "git_commit": COMMIT}
