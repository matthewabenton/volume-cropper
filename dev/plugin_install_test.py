"""0.13.0 napari plugin: does installing it from a (shared) tool folder work,
and does it leave that folder untouched? No windows are shown.

Usage (napari-env, which only runs this script; the plugin is installed
into ENV, a clone of napari-env such as napari-plugin-test):

    python plugin_install_test.py TOOL_DIR WORK_DIR [ENV] [--keep]

TOOL_DIR: the tool folder to test (e.g. a copy of the tool). Its
files (git's tracked and new ones, the working tree as it is) are copied
into WORK_DIR / "vc copy (with spaces)", and the copy is used as the
"shared folder". Checks:

1. volume_cropper.bat --install-plugin (VOLUME_CROPPER_ENV=ENV, no input:
   its "pause" gets end of file): exit 0; pip's editable finder maps only
   napari_volume_cropper, to the copy; the environment gets
   PYTHONDONTWRITEBYTECODE=1; nothing is written into the copy.
2. In ENV (activated): npe2 finds the plugin; the version and the tool
   come from the copy; napari starts (hidden window) with the plugin;
   still nothing written into the copy.
3. A release that adds a module and changes napari.yaml reaches the
   plugin without a reinstall.
4. The copy renamed (the drive not connected): napari still starts.
5. bash volume_cropper.sh --install-plugin (Git Bash) works too.
6. --uninstall-plugin: the plugin and the variable are gone; the copy is
   untouched.
Finally (unless --keep) ENV gets the plugin from TOOL_DIR again, so that
it can be tried there.
"""
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOL = Path(sys.argv[1]).absolute()
WORK = Path(sys.argv[2]).absolute()
ENV = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") \
    else "napari-plugin-test"
KEEP = "--keep" in sys.argv
# These reach cmd.exe (conda's .bat and volume_cropper.bat run through
# it), which would split a path or name at & | ^ < > and expand %...%:
# refuse such characters rather than quote them.
if set('&|^<>%!"') & set(str(TOOL) + str(WORK) + ENV):
    sys.exit("plugin_install_test: a path or the environment name holds one "
             "of & | ^ < > % ! \" (unsafe in cmd.exe); use another.")
COPY = WORK / "vc copy (with spaces)"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def listing(folder: Path) -> dict:
    """Every file and folder below `folder`, with its size and time."""
    out = {}
    for path in folder.rglob("*"):
        stat = path.stat()
        out[str(path.relative_to(folder))] = (
            path.is_dir(), 0 if path.is_dir() else stat.st_size,
            stat.st_mtime_ns)
    return out


def changes(before: dict, after: dict) -> list:
    return sorted(set(before.items()) ^ set(after.items()))


def run_bat(folder: Path, *args) -> subprocess.CompletedProcess:
    env = {**os.environ, "VOLUME_CROPPER_ENV": ENV}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    return subprocess.run(["cmd", "/c", str(folder / "volume_cropper.bat"),
                           *args], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, env=env,
                          timeout=600)


def in_env(code: str, activated: bool = True) -> subprocess.CompletedProcess:
    """Python code in ENV, run in WORK (not in the copy). (From a file:
    conda run cannot take a multi-line -c.)"""
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    script = WORK / "_in_env.py"
    script.write_text(code, "utf-8")
    if activated:
        command = ["conda", "run", "--no-capture-output", "-n", ENV,
                   "python", str(script)]
    else:
        command = [str(PREFIX / "python.exe"), str(script)]
    return subprocess.run(command, capture_output=True, text=True, cwd=WORK,
                          env=env, timeout=600, shell=activated)


def env_vars() -> dict:
    done = subprocess.run(["conda", "env", "config", "vars", "list", "-n",
                           ENV, "--json"], capture_output=True, text=True,
                          shell=True)
    try:
        return json.loads(done.stdout)
    except ValueError:
        return {"error": done.stdout + done.stderr}


def finder_mapping() -> dict | None:
    site = PREFIX / "Lib" / "site-packages"
    finders = list(site.glob("__editable___napari_volume_cropper_*_finder.py"))
    if not finders:
        return None
    for line in finders[0].read_text("utf-8").splitlines():
        if line.startswith("MAPPING"):
            return ast.literal_eval(line.split("=", 1)[1].strip())  # a dict
    return {}


done = subprocess.run(["conda", "run", "-n", ENV, "python", "-c",
                       "import sys; print(sys.prefix)"],
                      capture_output=True, text=True, shell=True)
PREFIX = Path(done.stdout.strip().splitlines()[-1])
print(f"tool {TOOL}\nenv {ENV} at {PREFIX}\ncopy {COPY}\n", flush=True)

# --- the copy: the working tree as it is (tracked and new files) -----------
if COPY.exists():
    shutil.rmtree(COPY)
files = subprocess.run(["git", "-C", str(TOOL), "ls-files", "-co",
                        "--exclude-standard"], capture_output=True,
                       text=True).stdout.splitlines()
for name in files:
    source = TOOL / name
    if source.is_file():
        (COPY / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, COPY / name)
before = listing(COPY)

# --- 1. install from the copy ---------------------------------------------------
done = run_bat(COPY, "--install-plugin")
check("1. --install-plugin exits 0", done.returncode == 0,
      (done.stdout + done.stderr)[-600:] if done.returncode else "")
mapping = finder_mapping()
check("1. the editable install maps only napari_volume_cropper, to the copy",
      mapping is not None and list(mapping) == ["napari_volume_cropper"]
      and Path(mapping["napari_volume_cropper"]).parent.samefile(COPY),
      str(mapping))
check("1. the environment has PYTHONDONTWRITEBYTECODE=1",
      env_vars().get("PYTHONDONTWRITEBYTECODE") == "1", str(env_vars()))
check("1. nothing was written into the copy",
      not changes(before, listing(COPY)), str(changes(before, listing(COPY))[:5]))

# --- 2. napari finds it and starts with it ----------------------------------------
code = (
    "import sys\n"
    "from npe2 import PluginManager\n"
    "pm = PluginManager.instance(); pm.discover()\n"
    "m = pm.get_manifest('napari-volume-cropper')\n"
    "from napari_volume_cropper import _tool\n"
    "crop_gui = _tool.import_tool()\n"
    "import volume_cropper as vc\n"
    "import napari\n"
    "v = napari.Viewer(show=False)\n"
    "v.open_sample('napari-volume-cropper', 'demo')\n"
    "names = [layer.name for layer in v.layers]\n"
    "v.close()\n"
    "print('RESULT', m.package_version, vc.__version__, crop_gui.__file__, "
    "names)\n")
done = in_env(code)
line = [x for x in done.stdout.splitlines() if x.startswith("RESULT")]
check("2. npe2 finds the plugin; tool and sample come from the copy",
      bool(line) and str(COPY) in line[0] and "nuclei" in line[0],
      line[0] if line else (done.stdout + done.stderr)[-800:])
check("2. still nothing written into the copy (activated environment)",
      not changes(before, listing(COPY)), str(changes(before, listing(COPY))[:5]))
# Without activation (python.exe straight away): the plugin's own guard
done = in_env("from napari_volume_cropper import _tool; _tool.import_tool(); "
              "print('ok')", activated=False)
written = changes(before, listing(COPY))
check("2. without activation only the plugin package's own cache files (if "
      "any) are written", done.returncode == 0 and all(
          "napari_volume_cropper" in w[0] for w in written),
      str(written[:6]) + (done.stderr[-300:] if done.returncode else ""))
for cache in COPY.rglob("__pycache__"):        # remove them again
    shutil.rmtree(cache, ignore_errors=True)
before = listing(COPY)

# --- 3. a release: a new module and a changed manifest --------------------------------
(COPY / "new_module_for_test.py").write_text("VALUE = 42\n")
yaml = COPY / "napari_volume_cropper" / "napari.yaml"
original = yaml.read_bytes()             # (bytes: keep its line ends)
yaml.write_bytes(original.replace(b"display_name: Volume Cropper\n",
                                  b"display_name: Volume Cropper NEW\n", 1))
done = in_env(
    "from npe2 import PluginManager\n"
    "pm = PluginManager.instance(); pm.discover()\n"
    "m = pm.get_manifest('napari-volume-cropper')\n"
    "from napari_volume_cropper import _tool\n"
    "_tool.import_tool()\n"
    "import new_module_for_test\n"
    "print('RESULT', m.display_name, new_module_for_test.VALUE)\n")
check("3. a new module and a new manifest work without a reinstall",
      "RESULT Volume Cropper NEW 42" in done.stdout,
      (done.stdout + done.stderr)[-500:])
(COPY / "new_module_for_test.py").unlink()
yaml.write_bytes(original)
before = listing(COPY)                   # (the test's own edits undone)

# --- 4. the folder gone (drive not connected) ---------------------------------------
moved = COPY.with_name("vc copy moved")
COPY.rename(moved)
try:
    done = in_env(
        "import napari\n"
        "v = napari.Viewer(show=False)\n"
        "v.close()\n"
        "print('RESULT started')\n")
    check("4. napari starts while the tool folder is unreachable",
          "RESULT started" in done.stdout, (done.stdout + done.stderr)[-500:])
finally:
    moved.rename(COPY)

# --- 5. the .sh launcher (Git Bash) -------------------------------------------------------
bash = shutil.which("bash")
if bash:
    env = {**os.environ, "VOLUME_CROPPER_ENV": ENV,
           "VOLUME_CROPPER_CONDA": str(Path(os.environ.get(
               "CONDA_EXE", PREFIX.parent.parent / "Scripts" / "conda.exe")
           ).parent.parent).replace("\\", "/")}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    sh = str(COPY / "volume_cropper.sh").replace("\\", "/")
    done = subprocess.run([bash, sh, "--install-plugin"], capture_output=True,
                          text=True, env=env, stdin=subprocess.DEVNULL,
                          timeout=600)
    mapping = finder_mapping()
    check("5. bash volume_cropper.sh --install-plugin works (Git Bash)",
          done.returncode == 0 and mapping is not None
          and list(mapping) == ["napari_volume_cropper"],
          (done.stdout + done.stderr)[-600:] if done.returncode else "")
else:
    print("SKIP 5. no bash found")
check("5. nothing was written into the copy",
      not changes(before, listing(COPY)), str(changes(before, listing(COPY))[:5]))

# --- 6. uninstall -------------------------------------------------------------------------------
done = run_bat(COPY, "--uninstall-plugin")
check("6. --uninstall-plugin: plugin and variable gone",
      done.returncode == 0 and finder_mapping() is None
      and "PYTHONDONTWRITEBYTECODE" not in env_vars()
      and not list((PREFIX / "Lib" / "site-packages").glob(
          "napari_volume_cropper-*.dist-info")),
      (done.stdout + done.stderr)[-400:])
check("6. nothing was written into the copy",
      not changes(before, listing(COPY)), str(changes(before, listing(COPY))[:5]))

# --- finally: ENV gets the plugin from TOOL_DIR -----------------------------------------
if not KEEP:
    tool_before = listing(TOOL / "napari_volume_cropper")
    done = run_bat(TOOL, "--install-plugin")
    mapping = finder_mapping()
    print(f"\n{ENV} now has the plugin from {TOOL}: "
          f"{done.returncode == 0 and mapping is not None}", flush=True)
    shutil.rmtree(COPY, ignore_errors=True)

print(f"\n{sum(results)} of {len(results)} checks passed.")
