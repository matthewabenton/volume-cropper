"""Build the PDFs of the user guide, the README and the other notes from
their Markdown (for whoever maintains them).

Edit docs/USER_GUIDE.md, then run (Python 3.10 or later, no extra packages):

    python docs/build_user_guide.py

The guide's PDF is named after the tool's version, e.g. USER_GUIDE_0172.pdf
for 0.17.2 (since 0.17.2; `.devN` is left out). A new name for each new
guide means a release never has to overwrite a PDF that someone has open on
the group drive (that stopped the releases of 0.6.0 and 0.17.0 halfway). The
script then points README.md's links at the new PDF, makes README.pdf again
(see below), and lists older guide PDFs still in docs/: remove those with
`git rm`.

The other PDFs, each next to its Markdown (0.17.0; README.pdf and
BETA_TESTING.pdf since 0.22.0, the user's wish: for colleagues who don't
read Markdown):

    python docs/build_user_guide.py README.md docs/BETA_TESTING.md
    python docs/build_user_guide.py docs/PATCH_NOTES.md
    python docs/build_user_guide.py --all       # every PDF listed in SOURCES
    python docs/build_user_guide.py --check     # which PDFs are out of step

(A table of contents only for the guide.) Each build notes a fingerprint
(SHA-256) of the Markdown it was made from in docs/pdf_sources.json, so
that `--check`, and the test tests/test_docs.py for the PDFs in ALWAYS, can
tell when a Markdown file has changed since its PDF was made.

It needs pandoc (https://pandoc.org) and Microsoft Edge or Google Chrome:
pandoc turns the Markdown into one web page (styled by docs/guide.css, with
the pictures from docs/images built in), and the browser prints that page to
PDF without opening a window. Links to files next to the page (e.g.
docs/PATCH_NOTES.pdf in the README) are printed as their path, as text: in
a PDF they would point into the temporary folder the page was printed from.
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # docs/
ROOT = HERE.parent                              # the tool folder
SOURCE = HERE / "USER_GUIDE.md"
GUIDE_PDF = re.compile(r"USER_GUIDE(_\d+)?\.pdf")   # any guide PDF's name
README = ROOT / "README.md"
# Every Markdown file that has a PDF (paths from the tool folder)
SOURCES = ("docs/USER_GUIDE.md", "README.md", "docs/BETA_TESTING.md",
           "docs/PATCH_NOTES.md", "docs/NOT_YET_TESTED.md")
# (0.22.0) The PDFs kept in step with their Markdown at every change (the
# test fails otherwise); the others are made again before a release.
ALWAYS = ("README.md", "docs/BETA_TESTING.md")
# Which Markdown (and its fingerprint) each PDF was made from
RECORD = HERE / "pdf_sources.json"
BROWSERS = [
    shutil.which("msedge"), shutil.which("chrome"), shutil.which("chromium"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]
# A pandoc filter (in Lua, pandoc's language for filters): links to files
# next to the page become their path as text (see the top of this file).
# Web links (https:, mailto:, ...) and links within the page (#...) stay.
LINK_FILTER = r"""
function Link(link)
  local target = link.target
  if target:match('^%a[%w+.-]*:') or target:sub(1, 1) == '#' then
    return nil
  end
  local name = target:match('([^/]+)$') or target
  if pandoc.utils.stringify(link.content) == name then
    return pandoc.Code(target)
  end
  local result = pandoc.Inlines(link.content)
  result:extend({pandoc.Space(), pandoc.Str('('), pandoc.Code(target),
                 pandoc.Str(')')})
  return result
end
"""


def guide_pdf_name(root: Path = ROOT) -> str:
    """'USER_GUIDE_0172.pdf' for version 0.17.2 (or 0.17.2.dev0).

    The version is read from volume_cropper/core.py as text, so this script
    needs none of the tool's packages.
    """
    text = (root / "volume_cropper" / "core.py").read_text(encoding="utf-8")
    found = re.search(r'^__version__ = "(\d+)\.(\d+)\.(\d+)', text, re.M)
    if found is None:
        sys.exit("No __version__ found in volume_cropper/core.py.")
    return f"USER_GUIDE_{''.join(found.groups())}.pdf"


def pdf_for(source: Path, root: Path = ROOT) -> Path:
    """The PDF made from a Markdown file: the guide's is named after the
    version, the others' after the Markdown (README.md -> README.pdf)."""
    if source.name == SOURCE.name:
        return source.with_name(guide_pdf_name(root))
    return source.with_suffix(".pdf")


def fingerprint(path: Path) -> str:
    """The SHA-256 of a file's bytes: any change to it changes this."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _from_root(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _load(record: Path) -> dict:
    try:
        return json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def note_built(source: Path, target: Path, root: Path = ROOT,
               record: Path = RECORD):
    """Note in the record that target was made from source as it is now."""
    entries = _load(record)
    entries[_from_root(source, root)] = {
        "pdf": _from_root(target, root), "sha256": fingerprint(source)}
    text = json.dumps(entries, indent=2, sort_keys=True) + "\n"
    record.write_text(text, encoding="utf-8", newline="\n")


def out_of_step(sources=SOURCES, root: Path = ROOT,
                record: Path = RECORD) -> list[str]:
    """What is wrong with the PDFs of these Markdown files (paths from the
    tool folder): missing, or made from an older version of the Markdown.
    An empty list if all are in step."""
    entries = _load(record)
    problems = []
    for name in sources:
        source = root / name
        target = pdf_for(source, root)
        entry = entries.get(name, {})
        if not target.exists():
            problems.append(f"{name}: no {_from_root(target, root)}")
        elif not entry:
            problems.append(f"{name}: no fingerprint of the Markdown that "
                            f"{_from_root(target, root)} was made from "
                            "(made before 0.22.0, or by hand)")
        elif entry.get("pdf") != _from_root(target, root) or \
                entry.get("sha256") != fingerprint(source):
            problems.append(f"{name} changed since "
                            f"{_from_root(target, root)} was made")
    return problems


def local_links(pdf: Path) -> list[str]:
    """Links in a PDF to files on the computer that made it (file:...)."""
    found = re.findall(rb"/URI\s*\((file:[^)]*)\)", Path(pdf).read_bytes())
    return [link.decode("latin-1") for link in found]


def point_readme_at(name: str):
    """Make README.md's links to the guide name the new PDF."""
    text = README.read_text(encoding="utf-8")
    new = GUIDE_PDF.sub(name, text)
    if new != text:
        README.write_text(new, encoding="utf-8", newline="")
        print(f"README.md now links {name}")


def wait_for_file(path: Path, timeout: float = 120) -> bool:
    """True once `path` exists and its size has stopped changing."""
    end = time.time() + timeout
    last = -1
    while time.time() < end:
        size = path.stat().st_size if path.exists() else -1
        if size > 0 and size == last:
            return True
        last = size
        time.sleep(1)
    return False


def _page_title(source: Path) -> list[str]:
    """pandoc options naming the page (the PDF's title, which PDF viewers
    show) after the first heading, for Markdown without a title block."""
    text = source.read_text(encoding="utf-8")
    if text.startswith("---"):                 # it has a title block
        return []
    heading = re.search(r"^# (.+)$", text, re.M)
    title = heading.group(1).strip() if heading else source.stem
    return ["--metadata", f"pagetitle={title}"]


def build(source: Path) -> Path:
    """Make the PDF of one Markdown file; return its path."""
    source = Path(source).resolve()
    is_guide = source.name == SOURCE.name
    target = pdf_for(source)
    toc = (["--toc", "--toc-depth=2", "--metadata", "toc-title=Contents"]
           if is_guide else [])
    pandoc = shutil.which("pandoc")
    browser = next((b for b in BROWSERS if b and Path(b).exists()), None)
    if pandoc is None or browser is None:
        sys.exit("Needs pandoc and Edge or Chrome (see the top of this file).")
    target.unlink(missing_ok=True)      # so an old PDF can't pass as new
    # The browser may still hold files in the folder when it is removed.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        html = Path(tmp) / f"{source.stem}.html"
        links = Path(tmp) / "links.lua"
        links.write_text(LINK_FILTER, encoding="utf-8")
        # One self-contained page: table of contents, style and pictures.
        subprocess.run([pandoc, str(source), "-o", str(html), "--standalone",
                        *toc, *_page_title(source), "--embed-resources",
                        "--lua-filter", str(links),
                        "--css", str(HERE / "guide.css"),
                        "--resource-path", str(source.parent)],
                       check=True, cwd=source.parent)
        # Print it to PDF with a headless browser (no window, no header).
        subprocess.run([browser, "--headless", "--disable-gpu",
                        "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}",
                        f"--print-to-pdf={target}", html.as_uri()],
                       check=True, capture_output=True)
        # Edge's launcher can return before the PDF is written; wait for
        # it here, while the page in `tmp` still exists.
        if not wait_for_file(target):
            sys.exit("The browser wrote no PDF within 2 minutes.")
    leaks = local_links(target)
    if leaks:
        target.unlink()
        sys.exit(f"{target.name} would link files on this computer: "
                 f"{', '.join(sorted(set(leaks)))}. Not kept.")
    note_built(source, target)
    print(f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    if is_guide:
        point_readme_at(target.name)
        older = sorted(p.name for p in HERE.glob("USER_GUIDE*.pdf")
                       if p.name != target.name and GUIDE_PDF.fullmatch(p.name))
        if older:
            print("Older guide PDFs still here (remove with git rm): "
                  + ", ".join(older))
    return target


def main(args: list[str]) -> int:
    if args == ["--check"]:
        problems = out_of_step()
        for problem in problems:
            print(problem)
        print("All PDFs in step." if not problems else
              "Make them again: python docs/build_user_guide.py --all")
        return 1 if problems else 0
    if args == ["--all"]:
        sources = [ROOT / name for name in SOURCES]
    else:
        sources = [Path(arg).resolve() for arg in args] or [SOURCE]
    # The guide's build changes README.md's links: README.pdf after it.
    if any(s.name == SOURCE.name for s in sources):
        sources = [s for s in sources if s != README.resolve()] + [README]
    for source in sources:
        build(source)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
