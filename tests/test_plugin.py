"""Tests for the napari plugin (0.13.0): napari_volume_cropper and the parts
of the tool it uses (file_images.LayerReader, crop_gui.open_layers and the
window bookkeeping).

Run with ``python run_all_tests.py plugin`` (or this file alone). The
windows are hidden (show=False). The tests check that:
- napari's manifest is valid and names functions that exist, and that
  napari can find them without importing the tool itself (napari then
  starts as fast as without the plugin);
- napari layers are read as an image exactly: channels, names, colours,
  contrast, voxel size, position, axes; lazy (dask) layers chunk by chunk;
  layers deleted in napari are let go; napari's "public-only" proxies give
  no warnings;
- crops of layers are checked OME-TIFFs;
- the commands open a new Volume Cropper window and leave the user's
  window as it was (its layers, napari's buttons); "Use this window…"
  turns the window into the tool; a closed window lets its image go; a
  window is not closed during a crop without asking;
- the launchers keep their line endings (CRLF for .bat, LF for .sh) and
  start the window from the user's folder, and only the plugin package is
  installed (pyproject.toml).
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import os
import subprocess
import sys
import tempfile
import time
import tomllib
import warnings
import weakref
from pathlib import Path

import numpy as np
import tifffile

from volume_cropper import file_images as fi
from volume_cropper import core as vc

HERE = Path(__file__).absolute().parent.parent     # the tool folder
RNG = np.random.default_rng(1)


def _pump(seconds=0.3):
    """Let Qt work, as napari's event loop would, including deleting
    closed windows (which processEvents alone leaves for the loop)."""
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.01)


def _layers(n=3, shape=(6, 40, 50), **kw):
    """n napari image layers of one image (its channels), not in a viewer."""
    from napari.layers import Image
    data = RNG.integers(1, 4000, size=(n, *shape), dtype=np.uint16)
    names = ["nuclei", "membranes", "marker"][:n]
    colors = ["magenta", "green", "cyan"][:n]
    layers = [Image(data[k], name=names[k], colormap=colors[k],
                    scale=kw.get("scale", (2.0, 0.5, 0.5)),
                    translate=kw.get("translate", (4.0, 1.0, 1.5)),
                    units=kw.get("units", ("um", "um", "um")),
                    contrast_limits=(10.0 * k, 3000.0 + k))
              for k in range(n)]
    return data, layers


def test_manifest_is_valid_and_napari_needs_no_tool_import():
    code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "from npe2 import PluginManifest\n"
        "m = PluginManifest.from_file("
        f"{str(HERE / 'napari_volume_cropper' / 'napari.yaml')!r})\n"
        "c = m.contributions\n"
        "assert len(c.commands) == 4 and len(c.widgets) == 1, c\n"
        "assert len(c.sample_data) == 1\n"
        "assert set(c.menus) == {'napari/layers/data', "
        "'napari/file/io_utilities'}, c.menus\n"
        "crop = [x for x in c.commands if x.id.endswith('crop_selected_layers')]\n"
        "assert crop[0].enablement == 'num_selected_image_layers >= 1'\n"
        "for command in c.commands:\n"
        "    module, name = command.python_name.split(':')\n"
        "    assert callable(getattr(importlib.import_module(module), name))\n"
        "assert 'volume_cropper.crop_gui' not in sys.modules, 'the tool was imported'\n"
        "assert 'volume_cropper' not in sys.modules\n"
        "print('OK')\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=tempfile.gettempdir(),
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0 and "OK" in done.stdout, (done.stdout
                                                         + done.stderr)


def test_tool_found_from_the_plugin_and_version():
    from napari_volume_cropper import _tool
    assert _tool.version() == vc.__version__
    before = sys.dont_write_bytecode
    try:
        crop_gui = _tool.import_tool()
        assert Path(crop_gui.__file__).absolute().parent.parent == HERE
        assert str(_tool.TOOL_DIR) in sys.path
        assert sys.dont_write_bytecode
    finally:
        sys.dont_write_bytecode = before


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=tempfile.gettempdir(),
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_the_running_tool_is_the_one_the_plugin_uses():
    """The launchers (volume_cropper/__main__.py) and python -m volume_cropper
    run the window from the package: the plugin must use that running module
    (its windows), not load a second copy."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import volume_cropper.crop_gui as running   # as __main__.py does\n"
        "from napari_volume_cropper import _tool\n"
        "assert _tool.import_tool() is running\n"
        "print('OK')\n")
    done = _run_python(code)
    assert "OK" in done.stdout, done.stdout + done.stderr


def test_another_copy_of_the_tool_is_refused():
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        other = Path(tmp) / "other copy"
        other.mkdir()
        shutil.copytree(HERE / "volume_cropper", other / "volume_cropper",
                        ignore=shutil.ignore_patterns("__pycache__"))
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(other)!r})\n"
            "import volume_cropper        # from another copy, first\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "from napari_volume_cropper import _tool\n"
            "try:\n"
            "    _tool.import_tool()\n"
            "except RuntimeError as err:\n"
            "    print('REFUSED', err)\n")
        done = _run_python(code)
        assert "REFUSED" in done.stdout and "other copy" in done.stdout, (
            done.stdout + done.stderr)


def test_only_the_plugin_package_is_installed():
    project = tomllib.loads((HERE / "pyproject.toml").read_text("utf-8"))
    tools = project["tool"]["setuptools"]
    assert tools["packages"] == ["napari_volume_cropper"]
    assert tools["py-modules"] == []
    assert "Private :: Do Not Upload" in project["project"]["classifiers"]
    entry = project["project"]["entry-points"]["napari.manifest"]
    assert entry == {"napari-volume-cropper": "napari_volume_cropper:napari.yaml"}


def test_launchers_keep_their_line_endings():
    for name in ("volume_cropper.bat", "install_napari_plugin.bat"):
        raw = (HERE / name).read_bytes()
        assert raw.isascii(), f"{name}: not plain ASCII"
        lines = raw.split(b"\n")
        assert lines[-1] == b"", f"{name}: no line end at the end"
        bad = [k + 1 for k, line in enumerate(lines[:-1])
               if not line.endswith(b"\r")]
        assert not bad, f"{name}: lines without CRLF: {bad[:5]}"
    # (bytes, not text: reading text turns CRLF into LF)
    text = (HERE / "volume_cropper.bat").read_bytes().decode("ascii")
    for label in (":install", ":uninstall", ":napari", ":plugin_env"):
        assert f"\n{label}\r" in text, label
    assert '-e "%~dp0."' in text and "--no-build-isolation" in text
    raw = (HERE / "volume_cropper.sh").read_bytes()
    assert b"\r" not in raw, "volume_cropper.sh must have LF line ends"
    assert b"--install-plugin" in raw and b"--no-build-isolation" in raw


def test_launchers_start_the_window_from_the_users_folder():
    """The launchers run volume_cropper/__main__.py as a file and do not
    change the working folder first, so a relative image path stays the
    user's (2026-11-24: the new package layout first changed into the tool
    folder). Run that way from another folder, __main__.py finds the package
    itself; --help stops before any window opens. (0.23.0) The .bat calls
    the environment's own python.exe by its full path: cmd.exe would run a
    python.bat or python.exe in the current folder first."""
    bat = (HERE / "volume_cropper.bat").read_bytes().decode("ascii")
    assert ('"%CONDA_PREFIX%\\python.exe" "%~dp0volume_cropper\\__main__.py" %*'
            in bat)
    assert "\npython " not in bat            # every Python by its full path
    assert "cd /d" not in bat
    sh = (HERE / "volume_cropper.sh").read_text(encoding="utf-8")
    assert 'run=("$here/volume_cropper/__main__.py")' in sh
    assert '\ncd "$here"' not in sh
    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run(
            [sys.executable, str(HERE / "volume_cropper" / "__main__.py"),
             "--help"], capture_output=True, text=True, cwd=tmp,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("usage: volume_cropper"), done.stdout


def test_dropped_name_with_ampersand_runs_nothing():
    """(0.23.2, the pre-publication review) A name dropped onto the .bat
    arrives as  cmd.exe /c ""<the .bat>" <name>",  unquoted when it has no
    space, and cmd.exe runs what follows an "&" in it as a command after
    the .bat ("ver" here prints the Windows version). The .bat's own end
    (its :end part, behind a stand-in that only prints what it got) ends
    that cmd.exe first, so nothing more runs; without it, "ver" runs."""
    if sys.platform != "win32":
        return
    bat = (HERE / "volume_cropper.bat").read_bytes().decode("ascii")
    end = bat[bat.index("\r\n:end\r\n") + 2:]     # the label, not a "goto :end"
    assert end.count("\r\n") < 25, "not the .bat's last part"
    head = '@echo off\r\nsetlocal\r\nset "EXIT_CODE=0"\r\necho got [%*]\r\n'
    with tempfile.TemporaryDirectory() as tmp:
        cases = {"with": head + "goto :end\r\n" + end, "without": head}
        for name, text in cases.items():
            stand_in = Path(tmp) / f"{name}.bat"
            stand_in.write_bytes(text.encode("ascii"))
            run = subprocess.run(
                f'cmd.exe /d /c ""{stand_in}" img&ver"',
                capture_output=True, text=True, timeout=60)
            assert "got [img]" in run.stdout, run.stdout
            ran = "Microsoft Windows" in run.stdout
            assert ran == (name == "without"), (name, run.stdout)


def test_layers_as_channels_of_one_image():
    data, layers = _layers()
    reader = fi.LayerReader(layers)
    scene = reader.scenes[0]
    # z guessed by position (it could be t in a 2D movie: changeable)
    assert scene.axes == "C?YX" and fi.default_axes(scene) == "CZYX"
    assert scene.shape == (3, 6, 40, 50)
    assert scene.unit == "micrometer"
    assert scene.voxel == {"z": 2.0, "y": 0.5, "x": 0.5}
    assert scene.origin == {"z": 2.0, "y": 2.0, "x": 3.0}
    assert scene.channel_names == ["nuclei", "membranes", "marker"]
    assert scene.channel_colors == ["FF00FF", "00FF00", "00FFFF"]
    assert scene.channel_windows == [(0.0, 3000.0), (10.0, 3001.0),
                                     (20.0, 3002.0)]
    assert reader.path == Path.home() / "nuclei"       # no file: home
    image = fi.load(reader, fi.FileChoice())
    level = image.levels[0]
    assert np.array_equal(np.asarray(level.array[:]), data)
    # voxel 0 sits where napari shows it (the layers' translate)
    assert np.allclose(level.translation[-3:], [4.0, 1.0, 1.5])
    assert image.channel_labels == ["nuclei", "membranes", "marker"]
    # a part, two channels
    choice = fi.FileChoice(channels=[0, 2], ranges={"z": (1, 4),
                                                    "x": (10, 30)})
    part = fi.load(reader, choice)
    assert np.array_equal(np.asarray(part.levels[0].array[:]),
                          data[[0, 2], 1:4, :, 10:30])


def test_one_layer_axes_pixels_multiscale_and_refusals():
    from napari.layers import Image
    movie = RNG.integers(0, 255, size=(3, 5, 30, 40), dtype=np.uint8)
    layer = Image(movie, axis_labels=("t", "z", "y", "x"))
    reader = fi.LayerReader([layer])
    scene = reader.scenes[0]
    assert scene.axes == "TZYX" and scene.channel_names == []
    assert scene.unit is None and scene.voxel["x"] is None    # pixels
    assert scene.interval is None
    assert any("no voxel size" in note for note in scene.notes)
    # napari's default labels ("-1" ...) say nothing: guessed by position,
    # and the guesses (not y, x) can be changed in "What to load"
    layer = Image(movie)
    scene = fi.LayerReader([layer]).scenes[0]
    assert scene.axes == "??YX" and fi.default_axes(scene) == "TZYX"
    image = fi.load(fi.LayerReader([layer]), fi.FileChoice(axes="CZYX"))
    assert image.axis_names == ["c", "z", "y", "x"]     # chosen otherwise
    # five unnamed axes: as Fiji's TCZYX; a named axis is kept where it is
    five = Image(np.zeros((2, 3, 4, 8, 8), np.uint8))
    assert fi.default_axes(fi.LayerReader([five]).scenes[0]) == "TCZYX"
    mixed = Image(movie, axis_labels=("a", "t", "y", "x"))
    scene = fi.LayerReader([mixed]).scenes[0]
    assert scene.axes == "?TYX" and fi.default_axes(scene) == "ZTYX"
    # multiscale: the full resolution is read
    big = RNG.integers(0, 255, size=(4, 64, 64), dtype=np.uint8)
    layer = Image([big, big[:, ::2, ::2]], multiscale=True)
    reader = fi.LayerReader([layer])
    assert reader.scenes[0].shape == (4, 64, 64)
    image = fi.load(reader, fi.FileChoice())
    assert np.array_equal(np.asarray(image.levels[0].array[:]), big)
    # a scale without units (napari's default "pixel"): the voxel size is
    # kept, only the unit is unknown
    layer = Image(big, scale=(2.0, 0.3, 0.3))
    scene = fi.LayerReader([layer]).scenes[0]
    assert scene.voxel == {"z": 2.0, "y": 0.3, "x": 0.3} and scene.unit is None
    assert any("no unit" in note for note in scene.notes)
    # refused: colour layers, layers of different sizes, voxel sizes or
    # positions, rotated layers
    for layers, error in (
            ([Image(np.zeros((4, 8, 8), np.uint8)),
              Image(np.zeros((4, 8, 8), np.uint8), translate=(0, 5, 5))],
             ValueError),
            ([Image(np.zeros((4, 8, 8), np.uint8), rotate=30)],
             NotImplementedError),
            ([Image(np.zeros((2, 3, 4, 8, 8), np.uint8)),
              Image(np.zeros((2, 3, 4, 8, 8), np.uint8))], ValueError),
            ([Image(np.zeros((8, 8, 3), np.uint8), rgb=True)],
             NotImplementedError),
            ([Image(np.zeros((4, 8, 8), np.uint8)),
              Image(np.zeros((4, 8, 9), np.uint8))], ValueError),
            ([Image(np.zeros((4, 8, 8), np.uint8), scale=(1, 1, 1)),
              Image(np.zeros((4, 8, 8), np.uint8), scale=(2, 1, 1))],
             ValueError)):
        try:
            fi.LayerReader(layers)
        except error:
            pass
        else:
            raise AssertionError(f"accepted: {layers}")


def test_lazy_layers_are_read_chunk_by_chunk():
    import dask
    import dask.array as da
    from napari.layers import Image
    stack = RNG.integers(0, 4000, size=(2, 40, 48, 48), dtype=np.uint16)
    calls = []

    def read_stack(t):                 # one z-stack per chunk, as readers do
        calls.append(t)
        return stack[t]

    lazy = da.stack([da.from_delayed(dask.delayed(read_stack)(t),
                                     shape=stack.shape[1:], dtype=stack.dtype)
                     for t in range(2)])
    layer = Image(lazy, axis_labels=("t", "z", "y", "x"))
    reader = fi.LayerReader([layer])
    calls.clear()        # (napari itself read a chunk to set the contrast)
    # the memory estimate includes the chunks held while reading
    block = 2 * 40 * 48 * 48 * 2
    assert fi.memory_needed(reader, fi.FileChoice()) >= stack.nbytes + block
    with dask.config.set(scheduler="synchronous"):
        image = fi.load(reader, fi.FileChoice())
    assert reader._cache is None      # let go before the smaller copies
    assert np.array_equal(np.asarray(image.levels[0].array[:]), stack)
    # 40 planes are read 16 at a time (3 reads per timepoint), but each
    # chunk only once
    assert sorted(calls) == [0, 1], calls


def test_deleted_layers_are_let_go():
    _, layers = _layers(n=2)
    reader = fi.LayerReader(layers)
    assert reader.alive()
    gone = weakref.ref(layers[0])
    del layers
    gc.collect()
    assert gone() is None, "the reader kept a deleted layer alive"
    assert not reader.alive()
    try:
        reader.read(0, (0, slice(0, 1), slice(None), slice(None)))
    except fi.LayersGone as err:
        assert "nuclei" in str(err)
    else:
        raise AssertionError("read a deleted layer")


def test_layers_kept_when_asked():
    _, layers = _layers(n=2)
    kept = fi.LayerReader(layers, keep=True)
    gone = weakref.ref(layers[0])
    del layers
    gc.collect()
    assert gone() is not None and kept.alive()


def test_proxied_layers_give_no_warnings():
    from napari.utils._proxies import PublicOnlyProxy
    data, layers = _layers(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        reader = fi.LayerReader([PublicOnlyProxy(layer) for layer in layers])
        image = fi.load(reader, fi.FileChoice())
        assert fi.source_file([PublicOnlyProxy(layers[0])]) is None
    assert np.array_equal(np.asarray(image.levels[0].array[:]), data[:2])
    assert reader.alive()               # the layers, not the proxies


def test_source_file_of_layers():
    from napari.layers import Image
    from napari.layers._source import layer_source
    with tempfile.TemporaryDirectory() as tmp:
        tiff = Path(tmp) / "stack.tif"
        tifffile.imwrite(tiff, np.zeros((3, 8, 8), np.uint8))
        zarr_dir = Path(tmp) / "image.zarr"
        zarr_dir.mkdir()
        (zarr_dir / "zarr.json").write_text("{}")
        lif = Path(tmp) / "movie.lif"
        lif.write_bytes(b"0")

        def layer_from(path):
            with layer_source(path=str(path)):
                return Image(np.zeros((3, 8, 8), np.uint8))

        assert fi.source_file([layer_from(tiff)]) == tiff
        assert fi.source_file([layer_from(zarr_dir)] * 2) == zarr_dir
        assert fi.source_file([layer_from(lif)]) is None       # can't open
        assert fi.source_file([layer_from(lif)], openable=False) == lif
        assert fi.source_file([layer_from(tiff), layer_from(lif)]) is None
        assert fi.source_file([Image(np.zeros((3, 8, 8), np.uint8))]) is None
        # the crops of layers from a file are named after it
        assert fi.LayerReader([layer_from(lif)]).path == lif


def test_crop_of_layers_is_a_checked_ome_tiff():
    data, layers = _layers()
    image = fi.load(fi.LayerReader(layers), fi.FileChoice())
    with tempfile.TemporaryDirectory() as tmp:
        box = vc.CropBox("blob", {"z": (6.0, 12.0), "y": (5.0, 15.0),
                                  "x": (5.0, 20.0)}, channels=[0, 2])
        result = vc.crop_box(image, box, Path(tmp) / "out",
                             compression="zstd")
        assert result.problems == [], result.problems
        assert result.path.name == "nuclei_blob.ome.tif"
        with tifffile.TiffFile(result.path) as tif:
            assert tif.pages.first.compression.name == "ZSTD"
            pixels = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"][
                "Pixels"]
            assert [c["Name"] for c in pixels["Channel"]] == ["nuclei",
                                                               "marker"]
            series = tif.series[0]
            crop = series.asarray().transpose(
                [series.axes.index(a) for a in "CZYX"])
        s = result.plan.starts[0]
        assert np.array_equal(
            crop, data[[0, 2], s[1]:s[1] + crop.shape[1],
                       s[2]:s[2] + crop.shape[2], s[3]:s[3] + crop.shape[3]])


class _Patched:
    """Answers the windows' dialogs in the tests (and puts things back)."""

    def __init__(self, crop_gui, accept_load=True, choose=None):
        from qtpy.QtWidgets import QMessageBox
        self.crop_gui, self.messages = crop_gui, []
        self.saved = [(crop_gui.FileLoadDialog, "exec"),
                      (crop_gui, "_choose"), (crop_gui, "open_window"),
                      (QMessageBox, "information"),
                      (QMessageBox, "warning"), (QMessageBox, "critical"),
                      (crop_gui.QMessageBox, "information"),
                      (crop_gui.QMessageBox, "warning"),
                      (crop_gui.QMessageBox, "critical")]
        self.saved = [(o, n, getattr(o, n)) for o, n in self.saved]
        crop_gui.FileLoadDialog.exec = (lambda self: 1) if accept_load \
            else (lambda self: 0)
        crop_gui._choose = lambda *a, **k: choose
        opener = self.saved[2][2]
        crop_gui.open_window = lambda *a, **k: opener(*a, show=False, **k)
        record = staticmethod(lambda *a, **k: self.messages.append(a[1:3]))
        QMessageBox.information = QMessageBox.warning = record
        QMessageBox.critical = record
        crop_gui.QMessageBox.information = record
        crop_gui.QMessageBox.warning = crop_gui.QMessageBox.critical = record

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for owner, name, value in self.saved:
            setattr(owner, name, value)


def _user_window(n=3):
    """A hidden napari window with n image layers, all selected."""
    import napari
    data, layers = _layers(n=n)
    viewer = napari.Viewer(show=False)
    for layer in layers:
        viewer.add_layer(layer)
    viewer.layers.selection.update(set(viewer.layers))
    return viewer, data


def test_command_opens_a_new_window_and_leaves_the_users_alone():
    import napari
    from napari.utils._proxies import PublicOnlyProxy
    from napari_volume_cropper import _commands, _tool
    crop_gui = _tool.import_tool()
    viewer, data = _user_window()
    buttons = viewer.window._qt_viewer.viewerButtons
    names = [layer.name for layer in viewer.layers]
    windows = len(crop_gui._WINDOWS)
    try:
        with _Patched(crop_gui) as patched, warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert _commands.crop_selected_layers(PublicOnlyProxy(viewer)) \
                is None
            _pump()
            assert len(crop_gui._WINDOWS) == windows + 1, patched.messages
            tool_viewer, panel = crop_gui._WINDOWS[-1]
            assert tool_viewer is not viewer
            assert np.array_equal(np.asarray(panel.image.levels[0].array[:]),
                                  data)
            assert panel.path_edit.text() == \
                "napari layers: nuclei, membranes, marker"
            # the user's window: the same layers, napari's buttons still there
            assert [layer.name for layer in viewer.layers] == names
            assert not buttons.ndisplayButton.isHidden()
            assert crop_gui.is_tool_viewer(tool_viewer)
            assert not crop_gui.is_tool_viewer(viewer)
            # in a Volume Cropper window the command refuses (its layers are
            # the tool's own)
            assert not _commands.open_layers_in_new_window(
                PublicOnlyProxy(tool_viewer))
            assert "Volume Cropper window" in patched.messages[-1][1]
            # closing the tool window lets its image go
            image = weakref.ref(panel.image.levels[0].array)
            tool_viewer.close()
            del panel, tool_viewer
            _pump()
            gc.collect()
            assert len(crop_gui._WINDOWS) == windows
            assert image() is None, "the closed window's image was kept"
        # Cancel in "What to load": no empty window is left
        with _Patched(crop_gui, accept_load=False):
            assert not _commands.open_layers_in_new_window(viewer)
            _pump()
            assert len(crop_gui._WINDOWS) == windows
        assert napari.current_viewer() is not None
    finally:
        viewer.close()
        _pump()


def test_use_this_window_turns_it_into_the_tool():
    from napari.utils._proxies import PublicOnlyProxy
    from napari_volume_cropper import _tool
    from napari_volume_cropper._widget import VolumeCropperPanel
    from qtpy.QtWidgets import QDockWidget
    crop_gui = _tool.import_tool()
    viewer, data = _user_window(n=2)
    try:
        panel = VolumeCropperPanel(PublicOnlyProxy(viewer))
        viewer.window.add_dock_widget(panel, name="Volume Cropper")
        assert "nuclei, membranes" in panel.selection_label.text()
        with _Patched(crop_gui, choose=None):          # Cancel: nothing
            panel._use_this_window()
            assert not crop_gui.is_tool_viewer(viewer)
        with _Patched(crop_gui, choose="Use this window") as patched:
            panel._use_this_window()
            _pump()
            assert crop_gui.is_tool_viewer(viewer), patched.messages
            tool = [w for v, w in crop_gui._WINDOWS if v is viewer][0]
            # the tool's panel sits next to the views (not in a dock that
            # could be closed); the small panel's dock is gone
            assert tool.ortho._outer.indexOf(tool.parentWidget()
                                             .parentWidget()) == 1
            docks = viewer.window._qt_window.findChildren(QDockWidget)
            assert not any(d.widget() is panel for d in docks)
            assert np.array_equal(np.asarray(tool.image.levels[0].array[:]),
                                  data[:2])
            buttons = viewer.window._qt_viewer.viewerButtons
            assert buttons.ndisplayButton.isHidden()     # as it warned
            # the user's layers were removed, but are kept for "What to
            # load…", which reads them again
            assert tool._layer_reader.alive()
            # a second time (e.g. from a new panel): refused
            again = VolumeCropperPanel(PublicOnlyProxy(viewer))
            again._use_this_window()
            assert "already" in patched.messages[-1][1]
    finally:
        viewer.close()
        _pump()


class _Event:
    """A stand-in for Qt's close event (a real one cannot be made
    "spontaneous", as when the user clicks the window's X)."""

    def __init__(self):
        self.ignored = False

    def ignore(self):
        self.ignored = True


def _tool_window_with_image(crop_gui, data):
    """A hidden Volume Cropper window with `data` read into memory."""
    from napari.layers import Image
    with _Patched(crop_gui):
        tool_viewer, panel = crop_gui.open_window()
        assert panel.open_layers([Image(data[0], name="nuclei")])
    _pump()
    return tool_viewer, panel


def test_closing_a_window_frees_its_image():
    """However a Volume Cropper window is closed (its X, or napari's File >
    Close Window), napari's Viewer.close() runs, and the image (read into
    memory) is let go."""
    from napari_volume_cropper import _tool
    crop_gui = _tool.import_tool()
    data, _ = _layers(n=1)
    saved = crop_gui._napari_may_close
    crop_gui._napari_may_close = lambda window: True
    try:
        for how in ("X", "Close Window"):
            tool_viewer, panel = _tool_window_with_image(crop_gui, data)
            image = weakref.ref(panel.image.levels[0].array)
            window = tool_viewer.window._qt_window
            if how == "X":
                event = _Event()
                assert panel._close_guard.on_close(window, event, True)
                assert event.ignored          # napari's close comes next
            else:
                window.close()                # not spontaneous
            del panel
            _pump(1.0)                    # napari's close, then Qt's delete
            gc.collect()
            assert not crop_gui.is_tool_viewer(tool_viewer), how
            del tool_viewer, window
            for _ in range(10):           # (napari lets go step by step)
                gc.collect()
                if image() is None:
                    break
                _pump(0.2)
            assert image() is None, f"{how}: the image was kept"
    finally:
        crop_gui._napari_may_close = saved


def test_window_not_closed_during_a_crop_without_asking():
    from napari_volume_cropper import _tool
    crop_gui = _tool.import_tool()
    data, _ = _layers(n=1)
    tool_viewer, panel = _tool_window_with_image(crop_gui, data)
    window = tool_viewer.window._qt_window
    guard = panel._close_guard
    answers, stopped = [], []
    saved = crop_gui._confirm
    panel._cancel_crop = lambda: (stopped.append(1),
                                  setattr(panel, "_busy", False))
    try:
        panel._busy = True                 # as while a crop is written
        crop_gui._confirm = lambda *a: answers.append(a[1]) or False
        event = _Event()
        assert guard.on_close(window, event, True)       # the X: asked
        assert answers == ["A crop is running"] and event.ignored
        assert not stopped and crop_gui.is_tool_viewer(tool_viewer)
        # napari quitting (the viewer already torn down): not asked (there
        # is nothing to keep), the crop is stopped
        tool_viewer.layers.clear()
        assert not guard.on_close(window, _Event(), False)
        assert answers == ["A crop is running"] and stopped
        # the X with "Yes": stopped, then closed through Viewer.close()
        panel._busy, stopped[:] = True, []
        crop_gui._confirm = lambda *a: True
        tool_viewer.add_image(np.zeros((4, 4)))     # (layers again)
        event = _Event()
        assert guard.on_close(window, event, True) and stopped
        _pump(0.5)
        assert not crop_gui.is_tool_viewer(tool_viewer)
    finally:
        crop_gui._confirm = saved
        if crop_gui.is_tool_viewer(tool_viewer):
            panel._busy = False
            tool_viewer.close()
            _pump()


def test_use_this_window_while_it_shows_3d():
    """(The 0.17.0 review) "Use this window…" in a window that shows 3D,
    a grid and transposed axes: opening the image there crashed (a 2D
    layer in a 3D canvas); now the window is set back to plain 2D first,
    and "What to load…" works there with a region selected."""
    from napari.utils._proxies import PublicOnlyProxy
    from napari_volume_cropper import _tool
    from napari_volume_cropper._widget import VolumeCropperPanel
    crop_gui = _tool.import_tool()
    viewer, _ = _user_window(n=2)
    viewer.dims.ndisplay = 3
    viewer.grid.enabled = True
    viewer.dims.order = (0, 2, 1)
    try:
        panel = VolumeCropperPanel(PublicOnlyProxy(viewer))
        viewer.window.add_dock_widget(panel, name="Volume Cropper")
        with _Patched(crop_gui, choose="Use this window") as patched:
            panel._use_this_window()
            _pump()
            tool = [w for v, w in crop_gui._WINDOWS if v is viewer][0]
            assert tool.image is not None, patched.messages
            assert viewer.dims.ndisplay == 2 and not viewer.grid.enabled
            image = tool.image
            tool._box_added({a: (lo + 0.3 * (hi - lo), lo + 0.6 * (hi - lo))
                             for a, (lo, hi) in ((a, image.extent(a)) for a
                                                 in image.range_axes)},
                            ("y", "x"), None)
            _pump()
            tool._change_what_to_load()
            _pump()
            assert len(tool.ortho.boxes) == len(tool.boxes) == 1
    finally:
        viewer.close()
        _pump()


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["plugin"]))
