"""Linhagem de um run: qual dado e qual código geraram o modelo."""

from pathlib import Path

import yaml


def read_git_commit(project_root: Path) -> str | None:
    """Lê o hash do commit atual direto do `.git`, sem depender do binário `git`.

    Returns:
        Hash do commit, ou None se não houver repositório git.
    """
    head_file = project_root / ".git" / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text().strip()
    if not head.startswith("ref: "):
        return head  # HEAD destacado: já é o hash
    return _resolve_ref(project_root / ".git", head.removeprefix("ref: "))


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    ref_file = git_dir / ref
    if ref_file.is_file():
        return ref_file.read_text().strip()
    packed_refs = git_dir / "packed-refs"
    if not packed_refs.is_file():
        return None
    for line in packed_refs.read_text().splitlines():
        if line.endswith(f" {ref}"):
            return line.split()[0]
    return None


def read_data_md5(pointer_file: Path) -> str | None:
    """Lê o hash do dado no ponteiro `.dvc`, ou None se o ponteiro não existir."""
    if not pointer_file.is_file():
        return None
    outputs = yaml.safe_load(pointer_file.read_text())["outs"]
    return outputs[0]["md5"]


class RunLineage:
    """Coleta as tags que ligam um run ao dado (hash do DVC) e ao código (commit)."""

    def __init__(self, project_root: Path, data_pointer: Path) -> None:
        self._project_root = project_root
        self._data_pointer = data_pointer

    def collect(self) -> dict[str, str]:
        """Tags `data_md5` e `git_commit`; as indisponíveis são omitidas."""
        tags = {
            "data_md5": read_data_md5(self._data_pointer),
            "git_commit": read_git_commit(self._project_root),
        }
        return {name: value for name, value in tags.items() if value is not None}
