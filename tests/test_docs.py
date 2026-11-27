"""Tests that the PDFs for colleagues are in step with their Markdown
(0.22.0, the user's wish: "always keep a PDF of the README for colleagues
that don't know .md"; also BETA_TESTING.pdf).

Run with ``python run_all_tests.py docs`` (or this file alone; no napari,
no pandoc needed). They check that:
- README.pdf and docs/BETA_TESTING.pdf exist and were made from the
  Markdown as it is now (docs/build_user_guide.py notes a fingerprint of
  each Markdown file it turns into a PDF, in docs/pdf_sources.json);
- that bookkeeping notices a change: a Markdown file edited after its PDF
  was made is reported, and so is a missing PDF;
- every link in README.md to a file in the tool folder leads to a file
  that exists (e.g. the guide's PDF, whose name changes with the version).
"""

# The tool folder is one level up from this tests folder.
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _build_script():
    """docs/build_user_guide.py as a module (docs/ is not a package)."""
    path = ROOT / "docs" / "build_user_guide.py"
    spec = importlib.util.spec_from_file_location("build_user_guide", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdfs_for_colleagues_in_step():
    build = _build_script()
    assert "README.md" in build.ALWAYS
    assert "docs/BETA_TESTING.md" in build.ALWAYS
    problems = build.out_of_step(build.ALWAYS)
    assert not problems, (
        "\n".join(problems) + "\nMake them again: python "
        "docs/build_user_guide.py " + " ".join(build.ALWAYS))


def test_out_of_step_notices_changes():
    build = _build_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        record = root / "record.json"
        source, target = root / "NOTES.md", root / "NOTES.pdf"
        source.write_text("# Notes\n", encoding="utf-8")
        # no PDF yet
        assert build.out_of_step(["NOTES.md"], root, record) == \
            ["NOTES.md: no NOTES.pdf"]
        # a PDF that was never noted as made from it
        target.write_bytes(b"%PDF-1.4 stand-in")
        assert len(build.out_of_step(["NOTES.md"], root, record)) == 1
        # made from it, as it is now: in step
        build.note_built(source, target, root, record)
        assert build.out_of_step(["NOTES.md"], root, record) == []
        # the Markdown edited afterwards: out of step
        source.write_text("# Notes\n\nOne more line.\n", encoding="utf-8")
        assert build.out_of_step(["NOTES.md"], root, record) == \
            ["NOTES.md changed since NOTES.pdf was made"]


def test_readme_links_lead_to_files():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    local = [t for t in targets
             if not re.match(r"[a-zA-Z][\w+.-]*:", t) and not t.startswith("#")]
    assert local, "no links found in README.md"
    missing = [t for t in local if not (ROOT / t).exists()]
    assert not missing, f"README.md links missing files: {missing}"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("PASS", name)
