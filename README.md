# Volume Cropper

![Volume Cropper: draw regions at low resolution, save them at full resolution. The XY, YZ and XZ views of a synthetic demo image (embryo-like objects in tissue), with crop regions as solid lines and the polygon outlines of the selected one dashed, and that crop turned in 3D.](docs/images/cover.png)

Crop boxes out of large OME-Zarr images (OME-Zarr 0.6, 0.5 or 0.4, e.g. from
eubi-bridge; also zipped: .ozx, .zip) and save each box as a new OME-Zarr in
the **same format**: the
same OME-Zarr version, data type, chunks, shards, compression and pyramid
levels (cut out level by level, not recomputed), the OME metadata, and a
record of where the crop came from. It runs in napari, with four linked
views (XY, YZ, XZ and 3D), and also as a notebook and on the command line.

**Version 0.23.1 (beta)**, for testing within the group: what is new is in
[PATCH_NOTES.pdf](docs/PATCH_NOTES.pdf), what could not be tested yet in
[NOT_YET_TESTED.pdf](docs/NOT_YET_TESTED.pdf). Testers: please see
[BETA_TESTING.pdf](docs/BETA_TESTING.pdf) for what is most useful to try and
report.

## Start here

**Read [USER_GUIDE_0231.pdf](docs/USER_GUIDE_0231.pdf)**: installing it on Windows, Mac or
Linux (including BARD), a first crop in five minutes, every part of the
window, large crops with the notebook or the command line, and what to do
when something goes wrong.

**A shared copy:** a group can keep the tool in one shared folder on a
network drive, and everyone starts it from there. If your group does, use
it from there, so that you always have the current version, and please
don't change its files. To keep a copy of your own instead (one fixed
version, or working without the drive), see the guide, *Installing it →
Your own copy*: copy the folder, don't move it.

**Everyone else:** download this repository (on GitHub: *Code › Download
ZIP*, or `git clone`), then follow the guide, *Installing it → Your own
copy*, with the downloaded folder as the tool folder.

## In short, for conda users

1. Create the environment once:
   `conda env create -f <tool folder>/env/environment.yml`
   (it is called `volume_cropper`).
2. Start it: double-click `volume_cropper.bat` (Windows), or type
   `bash <tool folder>/volume_cropper.sh` (Mac, Linux, BARD).
3. Large crops: *Save regions…* in the window, then use the notebook
   `examples/volume_cropper.ipynb` (copy it into your own folder first and
   set `TOOL_DIR` in its first code cell) or the command line, from any
   folder:
   `python <tool folder>/volume_cropper/core.py <image>.zarr boxes.csv <output folder> --dry-run`
   (see the guide, *Large crops*).

Save crops in a folder of your own: the tool refuses to write inside the
source image.

## Inside napari (the plugin, 0.13.0)

Volume Cropper can also be started from napari's menus, for example to
crop image layers that napari opened with another plugin's reader. Install
it once (double-click `install_napari_plugin.bat`, or
`bash <tool folder>/volume_cropper.sh --install-plugin`), then start
napari in the `volume_cropper` environment (on BARD:
`bash <tool folder>/volume_cropper.sh --napari`) and use *Plugins ›
Volume Cropper* or *Layers › Data › Crop selected layers…*. Volume Cropper
opens in a window of its own and leaves yours as it is.
`volume_cropper.bat --uninstall-plugin` removes it again.

## Files

| Path | What it is |
|---|---|
| `README.md` / `README.pdf` | this page (the PDF to read without a Markdown viewer) |
| `LICENSE` | the licence (BSD 3-Clause) |
| `volume_cropper.bat` / `volume_cropper.sh` | start the window: Windows / Mac and Linux (BARD); also `--install-plugin`, `--uninstall-plugin`, `--napari` |
| `install_napari_plugin.bat` | double-click to add Volume Cropper to napari's menus (the plugin) |
| `volume_cropper/` | the program: `core.py` crops (no window needed; also the command line), and the window and its parts (`crop_gui.py`, `ortho_views.py`, `load_dialog.py`, ...) |
| `napari_volume_cropper/`, `pyproject.toml` | the napari plugin (napari's manifest and the functions it calls) |
| `env/environment.yml` | conda recipe with everything the tool needs, in tested versions |
| `examples/volume_cropper.ipynb` | notebook for large crops (on a JupyterHub or a cluster node), with preview and checks |
| `docs/` | the user guide (`USER_GUIDE_0231.pdf`, its source and build script), `CHANGELOG.md` (what changed in each version), `PATCH_NOTES.pdf` (what is new, for users), `NOT_YET_TESTED.pdf` (what could not be tested yet), `BETA_TESTING.pdf` (for beta testers); each PDF next to its Markdown source |
| `tests/` | the tests: `python tests/run_all_tests.py` runs them all |
| `dev/` | for maintainers: `DEVELOPERS.md` (how the code is organised and tested, and the napari details it relies on) and the check scripts |

## Maintainer and licence

Maintainer: Matthew Benton (EMBL). Problems and wishes: please tell the
maintainer (from outside the group: through the GitHub repository).

Copyright (c) 2026 EMBL (European Molecular Biology Laboratory), Matthew
Benton. Volume Cropper is open source under the BSD 3-Clause licence: see
[LICENSE](LICENSE). The packages it uses (napari, Qt, zarr and the others
in `env/environment.yml`) come with their own licences when the
environment is installed.
