"""Tests that no file of the tool names a folder on a network drive, a
person's home folder or an email address (0.23.0).

The tool is public (on GitHub). Earlier versions named the folders of a
group's network drive in the README, the user guide and the notebook; this
test keeps such paths out. It looks only for the *form* of such paths, so
that the test itself names no drive, group or person:

- a path on a drive letter from D: to Z: (network drives are usually
  connected as one of these), e.g. ``X:\\folder``;
- a path under /g/, /scratch/, /home/, /Users/ or /Volumes/ (group drives
  on a cluster, home folders, network volumes on a Mac);
- a Windows network path (two backslashes, server, share);
- a Windows home folder (``C:\\Users\\name``);
- an email address.

Placeholders such as ``<tool folder>``, ``C:\\Users\\<you>`` or
``/Volumes/…`` are fine: a folder name must start with a letter, a digit
or ``_`` to count. The network paths ``\\\\host\\share`` and
``\\\\server\\share`` are fine too (tests use them as examples).

Run with ``python run_all_tests.py public`` (or this file alone; no napari
needed). In a git clone it reads the files git tracks plus new files not
yet added (but not ignored ones); in a copy without git (e.g. a download
from GitHub) every file in the tool folder. Binary files (pictures, PDFs)
are skipped: the PDFs are made from the Markdown files, which are read.
"""

# The tool folder is one level up from this tests folder.
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()     # holds made-up examples (see below)

# What counts, with a description for the report. Drive letters only in
# upper case: in code, "s:\n" is not a path.
PATTERNS = {
    "a path on a network drive letter":
        re.compile(r"(?<![A-Za-z0-9_])[D-Z]:[\\/]+[A-Za-z0-9_]"),
    "a cluster, home-folder or network-volume path":
        re.compile(r"(?<![A-Za-z0-9_.])/(?:g|scratch|home|Users|Volumes)/"
                   r"[A-Za-z0-9_]"),
    # (the placeholders \\host\share and \\server\share are fine: tests
    # use them to check that files on network shares are refused)
    "a Windows network path":
        re.compile(r"\\\\(?!(?:host|server)\\)"
                   r"[A-Za-z0-9_][A-Za-z0-9_.-]*\\+[A-Za-z0-9_$]"),
    "a Windows home folder":
        re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_]"),
    "an email address":
        re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
                   r"\.[A-Za-z]{2,}"),
}

SKIP_DIRS = {".git", "__pycache__"}


def _tool_files() -> list[Path]:
    """The tool's files: git's list in a clone, else every file."""
    try:
        run = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            cwd=ROOT, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        run = None
    if run is not None and run.returncode == 0 and run.stdout:
        names = [n for n in run.stdout.decode("utf-8").split("\0") if n]
        return [ROOT / n for n in names if (ROOT / n).is_file()]
    return [p for p in ROOT.rglob("*")
            if p.is_file() and not SKIP_DIRS & set(p.relative_to(ROOT).parts)]


def _text(path: Path) -> str | None:
    """The file's text, or None for a binary file (one with a NUL byte)."""
    data = path.read_bytes()
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace")


def _findings(text: str) -> list[tuple[int, str, str]]:
    """(line number, what, the matching text) for every match."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        for what, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                found.append((number, what, match.group(0)))
    return found


def test_patterns_catch_paths_and_leave_placeholders():
    # Made-up examples (no real drive, group or person), built in pieces
    # where a whole example would be found in this file by the next test.
    drive = "X" + ":"
    caught = [
        drive + "\\shared\\tools\\volume_cropper",
        drive + "/shared/tools",
        "/" + "g/somegroup/tools",
        "/" + "Volumes/groupdrive/tools",
        "/" + "scratch/someone/data",
        "/" + "home/someone/envs",
        "\\" + "\\server1\\share1\\folder",
        "C" + ":\\Users\\someone\\Documents",
        "someone" + "@example.org",
        '"' + drive + '\\\\shared\\\\tools"',     # as written in JSON
    ]
    for example in caught:
        assert _findings(example), f"not caught: {example!r}"
    fine = [
        "<tool folder>/env/environment.yml",
        "C:\\Users\\<you>\\.volume_cropper\\settings.json",
        "The folder then appears under /Volumes/….",
        'print(f"took {t:.1f} s:\\n{text}")',
        "napari 0.9.2, zarr >= 3, mg/g/day",
        "conda env create -f \"<tool folder>\\env\\environment.yml\"",
        r"\\host\share\x.ome.tif",
        "a network share (\\\\\\\\host\\\\share) or a ..",
        r"\\server\share",
    ]
    for example in fine:
        assert not _findings(example), f"wrongly caught: {example!r}"


def test_no_file_names_a_drive_folder_or_address():
    files = _tool_files()
    assert len(files) > 50, f"only {len(files)} files found in {ROOT}"
    problems = []
    for path in files:
        if path.resolve() == SELF:
            continue
        text = _text(path)
        if text is None:
            continue
        for number, what, found in _findings(text):
            name = path.relative_to(ROOT).as_posix()
            problems.append(f"{name}:{number}: {what}: {found}")
    assert not problems, (
        "Files name a folder on a drive or an address (use placeholders "
        "such as <tool folder> instead):\n" + "\n".join(problems))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("PASS", name)
