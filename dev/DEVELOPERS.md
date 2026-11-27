# Volume Cropper: notes for maintainers

For using the tool, see the user guide (`USER_GUIDE_<version>.pdf`). This
file is for whoever fixes or extends it: how the code is organised, how to
test it, how it was tested, and the napari 0.9.2 details it relies on.

Branches, tags, commits and the `reviews/` folder named below are in the
maintainer's development repository, which is not public; the real images
mentioned were the maintainer's own test data and are not part of the tool.

## Code

| File | What it holds |
|---|---|
| `volume_cropper/core.py` | the crop itself, without any window: opening an image (OME-Zarr 0.6, 0.5 or 0.4: `open_image`, `_ome_metadata`, `_compose_transforms`, `_coordinate_system`; the rest of the code reads the OME metadata through `OmeZarrImage.ome`, whatever the version; (0.21.0) zipped ones: `is_zipped`, `_open_zip_group`, `pack_zip`, `_crop_zipped`; collections: `collection_images`, `ImageChoiceNeeded`), planning a box (µm → voxels, snapping to the pyramid), writing a crop in the source's format and version (`_same_format`, `_copy_block`) and verifying it, boxes files (CSV), the command line. Its module docstring explains the method. |
| `volume_cropper/__main__.py` | starts the window. The launchers run it as a file (`python <tool folder>/volume_cropper/__main__.py`), without changing the working folder, so relative paths and the *Open* dialog stay the user's; run as a file it puts the tool folder on `sys.path` in place of the package folder (`if not __package__`), as `core.py` does for the command line, which then runs the package's copy of itself (`volume_cropper.core.main`, not `__main__`: one copy of its classes). From the tool folder, `python -m volume_cropper` does the same (2026-11-24) |
| `volume_cropper/crop_gui.py` | the panel on the right and the main window: boxes, view settings, running crops in a background thread |
| `volume_cropper/load_dialog.py` | the *What to load* dialogs: `LoadDialog` for OME-Zarr images, `FileLoadDialog` for files read into memory (0.12.0) |
| `volume_cropper/file_images.py` | (0.12.0) image files other than OME-Zarr: readers (`TiffReader`, `TiffSequenceReader`; since 0.15.0 `LifReader`, `CziReader`, `Nd2Reader`, `OirReader`; napari layers: `LayerReader`; one per format, each giving `scenes` and `read(scene, key)`), `load()` (the chosen part into zarr arrays in a `MemoryStore`, plus the pyramid; returns an `OmeZarrImage` with `source` set), the memory estimate |
| `volume_cropper/tile_stitching.py` | (0.17.2) stitching a tile scan without blending: `register` (stage places, pair matching by masked NCC on z projections, `fit_model` of the camera's angle and scale, `solve_places` by least squares, dropping matches that disagree), `assemble` (each voxel from the tile whose centre is nearest); used by `LifReader` (`_TileScan`, `prepare_scene`). No file access of its own |
| `volume_cropper/ortho_views.py` | the four linked views: layers, levels, projections, 3D masking, and the napari workarounds listed below |
| `volume_cropper/levels.py` | (0.18.0) the *Levels* histogram without a window: `histogram` (bins of whole numbers for integer data, covering the data and the contrast limits, or with `margin` (0.22.0) only the limits and room around them: `axis_range`, `ZOOM_MARGIN`; statistics), `read_values` (the whole loaded image or a crop region as cropped, inside its outlines, at the finest level with at most `MAX_VALUES` voxels, at a timepoint), gamma (`midtone_value`, `gamma_for_midtone`, `GAMMA_RANGE`) and the boxes' decimals and steps |
| `volume_cropper/levels_panel.py` | (0.18.0) the panel's part *Levels (histogram)* (`LevelsPanel`): draws it with matplotlib, the pointer readout, dragging the contrast lines and the gamma (midtone) line, the min / max / gamma boxes, (0.22.0) *zoom to min–max* (`_bin`, `_refit`); told of changes by the panel and `OrthoViews.on_xy_data` |
| `volume_cropper/box_handles.py` | (0.18.0) the geometry of the dots that change a crop region in 3D (no window): `handle_points`, `pick_handle` (nearest to the mouse's ray on screen), `along_axis` (a face follows the point of its axis nearest the ray), `on_plane` (the centre follows the ray across a camera-facing plane), `move_face`, `move_region`, `shifted_outlines`; used by `OrthoViews._handle_drag` |
| `volume_cropper/channels_panel.py` | (0.18.0) the panel's part *Channels* (`ChannelsPanel`): a row per channel (show tick box, name, colour menu with swatches); a click chooses the channel for Levels (through Levels' hidden channel combo); fed by the panel's `_on_display_changed` |
| `volume_cropper/qt_text.py` | (0.23.0) text from files never shown as HTML: `QLabel` and `QMessageBox` that show plain text (used by every module instead of Qt's own), `literal()` for tooltips and the labels of Qt's dialogs, `escape()` for names in the tool's own HTML, `defang()` for names napari shows; see *Files from elsewhere* |
| `volume_cropper/crop_projection.py` | (0.18.0) the crop preview's projection without a window: `crop_projection` (max or mean along each spatial axis of exactly what a region's crop keeps, via `levels.read_block`, at `projection_level`: at most `MAX_VOXELS` per channel); shown by `OrthoViews` ("the crop preview's projection": a thread of its own, one layer per channel and 2D view above the preview's black) |
| `volume_cropper/autocrop.py` | (0.14.0, experimental) finding a bright object and the crop region around it (`find_object`, `choose_level`, `object_in`), no window needed; (0.18.0) one's own threshold, seed modes, `search_volume` (the search's view of a timepoint, for the own threshold's slider and outline) |
| `volume_cropper/autocrop_panel.py` | (0.14.0, experimental) the panel's group *Experimental: find a bright object* (`ObjectFinder`) |
| `napari_volume_cropper/` | (0.13.0) the napari plugin: `napari.yaml` (napari's manifest), `_widget.py` (*Plugins › Volume Cropper*, a small panel), `_commands.py` (the menu commands), `_sample.py` (the synthetic sample), `_tool.py` (imports the tool from the tool folder) |
| `pyproject.toml` | (0.13.0) makes the plugin installable; only `napari_volume_cropper` is installed |
| `install_napari_plugin.bat` | (0.13.0) double-click: `volume_cropper.bat --install-plugin` |
| `examples/volume_cropper.ipynb` | the notebook (uses only `volume_cropper/core.py`) |
| `docs/build_user_guide.py` | makes the guide PDF from docs/USER_GUIDE.md, named after the version (`USER_GUIDE_0172.pdf` for 0.17.2: a new name for each new guide, so a release never overwrites a PDF someone has open), and points README.md at it (pandoc, then a headless Edge or Chrome); then README.pdf. Also the other PDFs (`<file>.md`, `--all`), each Markdown's fingerprint noted in `docs/pdf_sources.json`; `--check` lists PDFs out of step (0.22.0) |

Version 0.20.0 was developed as 0.18.0.dev0 (the user chose 0.20.0 to
mark a major step): notes "(0.18.0)" here and in the code mean 0.20.0.

Changes are made and tested in a separate copy; the shared tool folder is
updated only for a release. With each release, update `__version__` in
`volume_cropper/core.py`, docs/CHANGELOG.md and the guide's subtitle and authorship
line. The guide's pictures show only the synthetic demo image. When the
guide changes, rebuild it (a new `USER_GUIDE_<version>.pdf`) and `git rm`
the older one. In the shared folder, the release then deletes the old PDF:
if someone has it open, git only warns ("unable to unlink") and leaves it
behind as a stray file, to be deleted once it is closed (tested on a local
disk, 2026-11-11). Before 0.17.2 the guide was always `USER_GUIDE.pdf`, and
a release had to overwrite it: when it was open somewhere, the releases of
0.6.0 and 0.17.0 stopped halfway.

## Environment and tests

Everything runs in the `volume_cropper` environment made from
`env/environment.yml`. Its versions are pinned because `volume_cropper/ortho_views.py`
relies on napari 0.9.2 details (below), and it also pins the parts of
napari the tool depends on (Qt, vispy, superqt, psygnal, numcodecs and
others) to the versions all tests passed with. Qt is pinned only to the
6.8 series: 6.8.4 needs macOS 13.3 or newer on Apple Silicon, and older
Macs get 6.8.3. On 2026-10-28 the recipe was checked with `conda env
create --dry-run` for Windows, Linux (glibc 2.28) and Macs (Intel; Apple
Silicon with macOS 11 and 14), with `CONDA_SUBDIR` and
`CONDA_OVERRIDE_GLIBC`/`CONDA_OVERRIDE_OSX` set to stand in for those
systems. It has not yet been installed on a real Mac or on BARD. The crop code and the
notebook need only zarr, numpy and scikit-image (plus matplotlib and tqdm
for the notebook); they have also been run under Python 3.12.

Run the tests from the tool folder, with the environment active:

```
python tests/run_all_tests.py          # all tests (136 in 0.15.0, ~8 min); one line per
                                 # test, a summary, exit code 1 on failure
python tests/run_all_tests.py crop     # only the crop code (no napari needed)
python tests/run_all_tests.py plugin   # only the napari plugin (0.13.0)
```

Each test script also runs on its own (`python tests/test_volume_cropper.py`,
`test_ortho_views.py`, `test_napari_clipping.py`, `test_file_images.py`,
`test_plugin.py`), but stops at its first failure.
`test_napari_clipping.py` and `test_plugin.py` use hidden napari windows.
Installing the plugin is checked outside this folder, by
`dev/plugin_install_test.py` (a copy of the tool folder
as the "shared folder", installed into a clone of the environment).

Checks of the window itself (scripted runs in a visible window) are not yet
part of this folder. Run them in a **visible** window: an offscreen run once
missed a real crash (see *Removing layers* below).

## How it was tested

- **OME-Zarr 0.4 (0.8.0):** 9 more crop tests (46 in all) on synthetic 0.4 images
  (Zarr v2, nested chunk files): opening (voxel sizes, positions, channel
  names); a crop that stays 0.4, with every level's `.zarray` equal to the
  source's except the shape, the OME metadata directly in the attributes,
  the right positions, channels and voxels, read the same way by
  ome-zarr-py (an independent reader); with channels (czyx), a
  time-lapse (tczyx) and without a channel axis (zyx; 0.5 too, which was
  never tested before); an outline crop copied in many small blocks by 4
  workers (every voxel right; chunks outside the shape not written);
  cancel, overwrite, a foreign folder, a crop of a crop; no compression
  and "." chunk names copied; the copy blocks; and the clear refusals
  (not OME-Zarr, 0.1–0.3, axes without a type, a scale in a separate
  array, a bioformats2raw collection). The basic crop test runs for both
  versions. On a real eubi-bridge 0.4 image
  (zyx): a crop shaped by 3 polygons with a margin passed
  `verify_crop`, all levels identical voxel for voxel to an independent
  recomputation, read the same by
  ome-zarr-py; the source and the crops pass the official OME-Zarr 0.4
  JSON schemas (`image` and `strict_image`); local disk, 450 MB crop:
  2.3 s in 32 MB blocks against 4.0 s with one chunk per task. The
  window was checked on it too (14 checks at 0.8.0).
- **Crop code:** 37 tests on synthetic data, among them the outlines in
  3D (0.7.0): the shape through polygons in three views against a known
  object (a curved tube: at least 88 % kept, at most 5 % extra; measured
  91.7 % and 2.3 %), each polygon reproduced on its own slice, the shape
  ending at the outermost polygons, the same polygon far apart giving a
  straight prism, the margin against k-d tree distances (not scipy's
  distance transform), a written crop with a margin on every level,
  silhouettes (through their depth range, in several views, with a
  shape), holes filled, crossing points and disagreements, and boxes files
  with slices, silhouettes and margins. These cover pixel identity on
  every level, alignment between levels, format, metadata, clipping, a time
  axis, channel subsets, outlines (against an independent point-in-polygon
  calculation), safety guards (outputs inside the source, foreign folders,
  another box's crop), cancelling, an error stopping the queued work, the
  check finding a wrong position or voxel, boxes files (exact numbers,
  Excel's format, mistakes), the command line (every box checked first,
  messages and exit codes), and a crop of a crop.
- **View code:** 19 tests (since 0.8.1 also the voxels outside the box,
  which the clipped 3D view shows as 0, against an independent
  calculation, with box edges on and off the voxel grid): polygons per
  view and slice, the dashed
  outline of the shape on a slice (with a margin), the views' coarser
  drawing of the shape against the crop's (within 1 voxel of its edge),
  geometry, projections, dashed lines (which must
  follow corners), the masked region in 3D (against an independent
  calculation), the 3D masking reading each volume only once, loaded
  regions (the same voxels at the same position on every level), what the
  views read, and the level choice (the same as napari's own, over 96
  combinations of view, levels and zoom).
- **Real lightsheet data** (read over the group drive): crops up to 1.3 GB
  (uncompressed) took about 12 s on a Windows PC, about 100 MB/s, and passed
  verification. Two crops made with 0.1.0 were checked: level 0 is full
  resolution and pixel-identical to the source.
- **The window (scripted runs in a visible window on a Windows PC):**
  - 2026-10-23: drawing in XY and XZ, editing in YZ, the J key, display and
    tool syncing, clipping, level cap/exact, 3D off, projections, lasso and
    polygon outlines, real key presses, and crops checked against the
    source.
  - 2026-10-23, evening, on real and synthetic images: box lines (solid,
    dashed, faded, re-dashed when zooming); all 3D masking modes (switching
    takes about 1 s on real data, editing an outline also about 1 s);
    polygons in all three views, then every masking mode with and without
    clipping, a 3D level change, 3D off/on, deleting boxes and reopening
    the image (after the crash fix); "Open last crop" opening a full second
    window that cropped the crop again; a synthetic time-lapse (shared time
    slider, t range, crop identical to the source, 3D masking across
    timepoints).
  - 2026-10-26, on real data in a 1366 × 768 window: the level shown in the
    titles and in "Auto" while zooming (levels 3 → 0); level changes keep
    slices, zoom and pan (2D and 3D); dragging view borders (rows stay
    aligned); *Focus* and *Show all*; napari's panels hidden; 3D off keeping
    its cell.
  - 2026-10-26, later, on synthetic images: switching XY to 3D (napari's
    2D/3D button or Ctrl+Y) is undone, with no drawing errors (the version
    before this fix failed there, as in the session where the problem was
    reported); transposing axes and grid mode are undone too; the border of
    the right panel can be dragged narrower, wider, closed at the window's
    edge and opened again (before, it could not be dragged; plain napari's
    right docks behave the same).
  - 2026-10-27, *What to load* (0.5.0), on a synthetic 3-channel image with
    an offset (33 checks): the dialog's choices are what opens; every view
    shows exactly the region's voxels at the right place, and the whole
    region; the predicted levels are the ones napari shows; *Add* stays
    inside the region; a channel shown later gets its contrast then; 3D and
    its masking on a region; a crop after a restricted load is identical to
    one after a whole load; *Selected box* keeps the boxes and channels.
  - 2026-10-27, memory cache (7 checks, demo image): a 0.1 GB cache holds at
    most 100 MB; showing a channel again reads 0 MB; *Clear cache* empties
    it, and the channel is then read again (19 MB); 0 GB switches it off;
    napari's saved setting follows the field.
  - 2026-10-27, outlines combined with "any" (7 checks): a box with an XY
    polygon and a YZ polygon shorter in y (a reported case) now encloses
    both, and its crop holds exactly the voxels inside either polygon; a box
    saved the old way is fitted on *Load boxes…*. The same check on 0.4.0
    shows the old behaviour.
  - 2026-10-27, view options and cropping (20 checks, an image with two
    objects at different depths that overlap in XY): whole-depth MIP shows
    both; a 10 µm slab shows only the object at the slice and follows the
    slider to the other; a chosen z range shows only its object and keeps
    the slider inside; crosshairs and crop boxes hide and come back in all
    views; a crop's progress counts bytes and reaches 100 %, with speed and
    time left; with *Keep position* off a crop of an image with a stitching
    offset starts at 0 µm.
  - 2026-10-27, a terabyte-sized lightsheet image read over Wi-Fi
    (about 1.3 MB/s), one channel, 3D off, in a 1283 × 783 window: 0.4.0
    read 128 MB in 133 s; 0.5.0 read 126 MB (whole image, 91 s), 78 MB (a
    third of z, 58 s; moving the cursor then read 28 MB instead of 71 MB)
    and 117 MB (a third of z, y and x, 80 s: the smaller region is shown at
    finer levels). The estimate (before compression) was 240, 113 and
    137 MB; its predicted levels matched napari's every time.
  - 2026-10-30 (0.7.0, outlines in 3D and crop regions; 27 checks):
    polygons on two XY slices and in XZ go into the selected region (a new
    one if none); Finish region (pan/zoom, none selected), Delete, New
    region (polygon tool; removed again if nothing is drawn), a click on an
    empty part of the list; the × marks where the XZ polygon crosses an XY
    slice and the warning that the views disagree; a polygon drawn on a MIP
    is a silhouette, edited there; on a key slice the polygon is the
    editable shape, in
    between the rectangle and the shape's dashed outline, after the last
    polygon none; XZ off its polygon's slice, and YZ without any polygon,
    show the shape's outline too; the shading (kept / set to 0, colours,
    opacity, nothing outside the crop); the outline margin (box, kept
    voxels and plan text grow; the views update in about 0.3 s); the
    polygon list (go to, delete); editing and deleting single polygons; a
    crop with its record (dev/outlines_0_7_gui_test.py). The
    0.6.0 checks below passed again (23 checks).
  - 2026-11-02 (0.8.1, the halo in the clipped 3D view; 4 checks, embryo
    demo, trunk outlined, 3D level 1 with the region's y and x faces a
    quarter voxel off its grid): with clipping, the data given to napari
    are 0 at every voxel centre outside the region (0.8.0: 11.7 million
    voxels were not, so napari's straddling voxels and interpolation
    showed them along the faces); without clipping, the plain data;
    pictures from above and tilted, before and after
    (dev/halo_3d_check.py; `VC_TOOL` = an older copy shows
    the failure). The 0.7.0 (27), 0.6.0 (23) and 0.8.0 (14) checks passed
    again.
  - 2026-11-02 (0.8.0, OME-Zarr 0.4; 14 checks): on a real 0.4
    image without a channel axis, the *What to load* dialog (format named,
    no channel boxes), the image summary ("OME-Zarr 0.4, level 0: …"), no
    channel controls, the views and the 3D view drawing, a region from two
    XY polygons and one XZ polygon, its crop (0.4, checked, recorded) and
    *Open last crop* (asks what to load, opens the 0.4 crop); on a
    synthetic 3-channel 0.4 image, the channel boxes, hiding a channel and
    a crop of two channels (dev/v04_gui_test.py). The 0.7.0
    (27) and 0.6.0 (23) checks passed again on 0.5 images.
  - 2026-10-31: the 3D view's Auto level follows the channels shown
    (embryo demo: 3 channels level 1, 1 channel level 0, 3 again level 1; a
    chosen level stays; 4 checks), and the time-lapse notice appears for a
    time-lapse and not for a still image (2 checks); scratch scripts,
    visible window.
  - 2026-10-28 (0.6.0 fixes): box names that give one file name, boxes files
    with mistakes or from Excel, an output folder inside the source image,
    and *Overwrite* refusing another box's crop; the report of the check
    after a crop, and its warning when the check fails; single boxes hidden
    and shown (2D shapes, lines and names, 3D frames; an edit or a delete
    reaching the right box while another is hidden); the questions before
    boxes are discarded or replaced; the disk-space question; Cancel (23
    checks).
- **Notebook:** run end to end on real data (2026-10-23), and on a synthetic
  image under Python 3.12 (2026-10-28).
- **Review:** a full review of 0.5.0 (functions, design, bugs,
  organisation, explanation, code quality) on 27–28 October 2026, with 89
  black-box checks of the library and command line on the demo image (no
  wrong voxel, scale or position found).
- **Not yet tested:** the window on BARD since version 0.2; a real
  time-lapse; the notebook on the JupyterHub or the cluster; installing on
  a Mac; opening crops in Fiji/BigDataViewer; 0.4 crops written over the
  network (many small files: the speed is not measured; the user chose
  not to measure it, 2026-11-02).

- **Stitched LIF tile scans (0.17.2, 2026-11-11; the user's choice of
  2026-11-10):** `test_stitching.py` cuts tiles with known places out of a
  synthetic volume (camera turned 0.8 degrees, stage errors, z moves, an
  empty channel, background tiles, a reversed stage) and checks the
  places and the stitched voxels exactly. On real tile scans
  (`dev/stitching_0_17_2_check.py`): tiles of 512 and 1024 px within a
  few px of LAS X's placement, the raw overlaps' NCC at our offsets at
  least that at LAS X's; overview scans with 64-128 px tiles several px
  from LAS X, weak overlaps (uncertain). In z, LAS X moved some tiles by
  a few planes; the raw overlaps match at the same plane, where Volume
  Cropper keeps them. Synthetic thin overlaps (7 px of 64, with noise)
  showed no bias. Placing 8 tiles of 1024 x 1024 px took a few seconds.
  The dialog was checked in a visible window (5 checks).
  Design choices: whole voxels only and nearest-centre overlaps (measured
  values; the user's choice); the stitched scene is placed when first
  chosen (`prepare_scene`, from the dialog, `load` and `memory_needed`), so
  opening a file with several tile scans stays quick; the tiles stay
  available as "(tiles)"; flags other than 0 (never seen): orientation
  found from the overlaps; a failed registration falls back to the stage
  positions with a note.

## Planned

**RFC-8 collections (decision 1a, 2026-11-22): later, once adopted.**
On 2026-11-22 RFC-8 was in state R1 (under review), not implemented
anywhere, with the placeholder version "0.x"; it would also replace the
multiscales/datasets metadata with "multiscale"/"singlescale" nodes and
the bioformats2raw, labels and HCS layouts. When it is adopted: add it to
`collection_images` (a collection's nodes → the image chooser; follow
only relative paths inside, as its security section requires) and read
its image nodes. Also still open from 0.21.0: zip reading speed over a network drive
(zarr's ZipStore reads one piece at a time), and the notebook using
`crop_box` for zipped crops.

**All of 0.9.0-0.17.0 below was released together as 0.17.0 (2026-11-09,
the user's request), with the user guide written for all of it,
`docs/PATCH_NOTES.md` and `docs/NOT_YET_TESTED.md` (PDFs made by
`docs/build_user_guide.py <file>.md`).** The notes below ("guide still to
write", "awaiting the user's test") describe the state before that release;
what is still untested is in `docs/NOT_YET_TESTED.md`. *Find a bright object* is
now called the Autocropper (code: `volume_cropper/autocrop.py`, `volume_cropper/autocrop_panel.py`, the
panel part key "find").

The user's choices of 2026-11-06, one release each, in this order:

- **0.9.0 Crop preview** (done on experimental; guide still to write):
  *Preview crop* under *View*: *off* or *in the main views*: black
  wherever the selected region's crop has nothing on the slice shown
  (black rectangles outside the region with its exact edges, and the
  voxels set to 0 on a grid of the level shown, at most `PREVIEW_POINTS`);
  the whole view when the slice or timepoint is outside the region; on a
  projection only the rectangle. *Zoom to region* fits the 2D views.
  The user's decisions after testing it (2026-11-06): the preview shows the
  channels the user has switched on (also channels the region does not
  keep), and it leaves the shading as the user set it.
  *Side by side* (a coarse real crop in a temporary folder, shown in a
  second window that followed the main window) was built and tested
  (commit 1736359), then dropped by the user after testing it
  (2026-11-06: "we don't need the side-by-side") and reverted; the code is
  in that commit if it is ever wanted.
- **0.10.0 TIFF export** (done on experimental, awaiting the user's test in
  Fiji; guide still to write; TIFF only, no HDF5): see *Design decisions*.
- **0.11.0 Cropping in time** (done on experimental, awaiting the user's
  test; guide still to write): `CropBox.t_step`; the plan has a step per
  axis (`CropPlan.steps`, output index j = source start + j × step), used
  by `_source_selection` (stepped slices; zarr's orthogonal indexing reads
  them), the crop's OME scale (× step), the check and the record. The t
  range ends at the last kept timepoint (`plan_crop` trims the stop), so
  `snapped_ranges` and the plan text show what is kept. Levels that also
  shrink time end the kept levels when n > 1 (their timepoints are not
  level 0's). Window: the t row has whole timepoint numbers (QSpinBox,
  1-based, with the time as suffix) and *every n*; the region keeps times
  (frame centres), so the views, the regions file and the crop code are
  unchanged. `OrthoViews.time_kept(i)` decides the fading (2D lines, 3D
  frame line colour on every slider move via `_fade_3d_frames`) and makes
  the crop preview black at skipped timepoints. Open question for the
  user: napari's own sliders count from 0, the t fields from 1. Not wanted
  for now: one range for all regions.
- **0.12.0 Other files** (asked 2026-11-06, before the test round of
  0.9-0.11; step A, TIFF, done on experimental; guide still to write):
  the user's decisions: files are read **into memory** (like Fiji without
  a virtual stack; only the part chosen must fit), **never converted to
  OME-Zarr**; a TIFF comes in, a TIFF goes out; TIFF compression = **zstd
  with the predictor** (TIFF has no Blosc); LIF, CZI, ND2 and OIR crops
  are TIFFs too; **no Bio-Formats** for now (rare formats: Fiji → Save as
  TIFF); 2D images show **only the XY view**. Step B (done 2026-11-08 as
  0.15.0, after the user found the formats missing): readers for LIF
  (liffile 2026.10.14), CZI (czifile 2026.10.14 + imagecodecs), ND2 (nd2
  0.12.0) and OIR (oirfile 2026.10.22), installed into the environment with the
  user's OK (no downgrades; they bring pyarrow, xarray, dask and ruff);
  calibration checked against Fiji's Bio-Formats on real files
  (headless Fiji: `fiji-windows-x64.exe --headless --run script.groovy`,
  writing to a file; the macro Bio-Formats importer opens nothing
  headless): identical. The
  readers (`_MicroscopeReader`): a LIF is memory-mapped on the file
  (`LifImage.asarray(out="memmap")`), a CZI read by plane selection in file
  coordinates (`CziImage.start`) and a region (`roi`, absolute: `bbox`)
  and put back in the key's order, an ND2 through `to_dask()` (positions
  P are scenes), an OIR memory-mapped (untested on a real file). Axes are
  T, C, Z, Y, X; any other library axis is "-" if single, else "?" (the
  user names it). Positions start at 0 (stage coordinates are not used:
  they made the region fields read tens of thousands of µm). The CZI's frame
  interval comes from its frame times (median step): czifile's T step
  was a nominal 0.1 ms; Bio-Formats gives none. Research of
  2026-11-06 (readers for Python 3.14 on conda-forge, zstd in Fiji):
  pylibCZIrw has no Python 3.14 build; bioio's `BioImage` reads whole
  scenes for subsets and silently keeps only view 0 / illumination 0 of
  light-sheet CZIs, so each format gets its own small reader instead.

- **0.13.0 napari plugin, at first for use within the group** (asked 2026-11-07,
  after the prototype on the branch `plugin-prototype`, see its
  `PLUGIN_PROTOTYPE.md`): done on experimental, awaiting the user's test
  (Windows, then BARD and a Mac); guide still to write (with 0.9-0.12).
  Not for PyPI or the napari hub (classifier *Private :: Do Not Upload*).

- **0.14.0 Find a bright object (experimental)** (asked 2026-11-07 for
  later; started while the user was away, as an experimental option off
  by default): done on experimental, awaiting the user's test on real data
  (a single embryo in a time-lapse); guide still to write. Ideas if it
  is kept: a per-timepoint report in the window; a stricter search where
  the object touches bright tissue (a higher threshold). (Silhouette
  outlines and seeds from clicks: done in 0.16.0.)

- **0.15.0 from the user's test round (2026-11-07):** *Save as TIFF…*
  (a button) instead of *Also save as TIFF*; the wheel no longer changes
  values while the panel is scrolled; *Smallest level*, *Parallel workers*
  and *Keep position* removed from the window. Done on experimental.

- **0.16.0 from the user's test round (2026-11-08):** OME-TIFF companion
  files; × marks in their view's colour; *Find a bright object*: no
  growing on repeated fits, *Fit all regions*, seeds (*Add seeds*, *Fit to
  seeds*) and *Outline (silhouettes)*; collapsible panel parts, *Settings…*
  remembered per user, the tool's own layers out of napari's layer list.
  Done on experimental (see CHANGELOG), awaiting the user's test; guide
  still to write.

- **0.17.0 Regions that move over time** (asked 2026-11-08 after testing
  *Find a bright object* on time-lapses; the user chose "one moving
  region, one movie" on 2026-11-09): done on experimental, awaiting the
  user's test on a real time-lapse; guide still to write. Outlines per
  timepoint (the user's choice "option 2", for auto-cropping): done too.
  Not yet: drawing new polygons on a moving region; smoothing the track
  (positions are the object's box centres, as found).
- **0.18.0 Levels (histogram)** (asked 2026-11-12: napari's histogram has
  no numbers; the user chose all the numbers, draggable limit lines, and
  all three sources): done on experimental (`test_levels.py`; a visible
  check, `dev/levels_0_18_gui_test.py`, 9 of 9), awaiting
  the user's test; guide still to write. matplotlib (already in the
  environment for the notebook) draws it, so colleagues need no update.
  The same day, during the user's test, they asked for gamma (a dashed
  midtone line, dragged or double-clicked) and boxes for min, max and
  gamma: done (the visible check now 13 of 13).
- **0.18.0, three smaller wishes of 2026-11-12** (the user approved the
  plan; for the preview they chose the exact projection): several regions
  and polygons deleted at once; the crop preview's projection of exactly
  what the crop keeps (`volume_cropper/crop_projection.py`); a "Loading…" box while
  napari reads (also when an image opens). Done on experimental
  (`test_panel`, `test_crop_projection`; visible check
  `dev/panel_0_18_gui_test.py`, 11 of 11), awaiting the
  user's test; guide still to write.
- **0.18.0, changing a region in 3D** (asked 2026-11-12; the user chose
  face dots plus a centre dot, no corners): done on experimental
  (`test_box_handles`; visible check with real Qt mouse events
  `dev/handles_0_18_gui_test.py`, 6 of 6), awaiting the
  user's test. Not done: corner dots (hard to aim at in 3D); dragging
  while looking straight along an axis. Second round (the user's choices,
  the same evening): Shift / Ctrl drags, Up / Down nudges, live clipping,
  click to select in 3D, Zoom to region in 3D, undo / redo of all region
  edits (visible check 13 of 13). Declined for now: hover feedback, a
  readout while dragging, faces in the views' colours.
- **0.18.0, Autocropper: own threshold and seed modes** (asked
  2026-11-13: brighter parts of an embryo apart from fainter ones; the
  user chose one fixed outline colour, *most stable* as the seeds'
  default, and the own threshold for all three fits): done on
  experimental (`test_autocrop`: core / body / tissue, the outline, the
  panel, the timepoint; visible check
  `dev/own_threshold_0_18_gui_test.py`, 13 of 13),
  awaiting the user's test; guide still to write. Known: *whole object*
  takes in tissue that touches the object (the demo's embryos), because
  the tissue is a sparse network below `SEED_MAX_SHARE` of the volume.
- **0.18.0, splitting touching objects, gap lines** (asked 2026-11-13
  after the own threshold still fused two embryos; the user chose the
  ilastik-like seeded watershed with drawn gap lines over a geometric cut,
  which as a wall through the depth would slice tilted objects' corners;
  freehand drawing; white dash-dot lines): done on experimental
  (`test_autocrop`: tilted spheres, gap wall, time, panel; visible check
  `dev/gap_split_0_18_gui_test.py`, 7 of 7), awaiting the
  user's test; guide still to write. Then the same evening (the user's
  choices): outlines per slice (always for split parts, margin stopping
  halfway to the other part), outlines in 3D as surfaces (default the
  selected region, beside the 3D dimming), and the Settings' start
  options for the Autocropper and View (new windows only): visible check
  `dev/outlines_3d_0_18_gui_test.py`, 5 of 5. Open: the
  × marks in XZ / YZ (where other views' polygons cross the slice) form
  a dense chain with 40 per-slice polygons; the user was unsure about
  holes in existing polygons (left as they are).
- **0.22.0 (2026-11-25, the user's wishes): Levels "zoom to min–max"
  and README.pdf.** The zoom's room is 25 % of max - min per side (the
  user's choice), clipped to the unzoomed axis (data and limits: nothing
  to see beyond); the bins are recomputed over it (zooming 16-bit data to
  a few hundred values would otherwise show 1-2 coarse bins). It never
  rescales during a drag: the axis moving under the pointer would make the
  dragged line run away; `_refit` runs when a drag ends, from `on_display`
  (boxes, auto-contrast, napari) and `set_marker`, and only rebins when
  `axis_range` changes (gamma alone does not). README.pdf and
  BETA_TESTING.pdf have plain names (the user's choice: "GitHub style"),
  so like PATCH_NOTES.pdf a release overwrites them: nobody may have them
  open in the shared copy then. `tests/test_docs.py` keeps those two in step
  (fingerprints in `docs/pdf_sources.json`); the guide, PATCH_NOTES and
  NOT_YET_TESTED are still made before a release (`--check` lists them).

## Design decisions

- **OME-Zarr 0.6, zipped OME-Zarr, collections (0.21.0; the user's
  decisions of 2026-11-22).** Specs read that day: 0.6 final
  (2026-09-14, RFC-5), RFC-9 (zip) state R4, RFC-8 (collections) state
  R1 "not implemented yet" (it also replaces the image metadata itself
  with "nodes", version "0.x").
  - 0.6: `_compose_transforms` applies a level's transformations in order
    (scale, translation, identity, sequence; others refused) to one scale
    and one translation, for all versions. The axes come from the
    coordinate system all levels lead to (`OmeZarrImage.coordinate_system`).
    Crops: one transformation per level (`{scale, input, output}` or a
    `sequence`), the same system name. Multiscale-wide links (decision
    5a): kept as they are with *Keep position*; without it,
    `_shifted_link` adds a translation (into the crop: +shift first; out
    of it: -shift last), sequences flattened. Links that need another
    group (input/output `path`) or arrays (a `path` parameter) are left
    out; unused coordinate systems too; `scene` is dropped. Record:
    `coordinate_system` {cropped_in, links_kept, notes}. `is_v06` also
    accepts pre-release names ("0.6.dev4": ngff_zarr 0.43 writes it; its
    own validator accepts only that, the official schema only "0.6" and
    "0.6rc0"); crops keep the name as written.
  - Zip reading: zarr's `ZipStore` (one handle, reads serialised under a
    lock). Measured on this PC with the demo (105 MB, sharded): the .ozx
    read faster than the folder (level 0, 8 threads: 1.4 vs 2.0 s; 10
    planes 0.22 vs 0.36 s). Not measured over a network drive yet. A zip whose metadata
    sit in one folder inside (Explorer, Finder) is read there; a zip in a
    zip, an empty zip and a damaged or split zip get clear errors.
    Compressed (Deflate) zips work but warn: a byte range in a compressed
    entry is read from its start. `OmeZarrImage.close()` closes the file
    (Windows cannot replace an open file); the window closes the old
    image after showing the new one.
  - Zip writing (decisions 2a, 3a, 4a): `output_suffix` → `.ozx` for v3,
    the source's ending for a zipped 0.4. `_zip_work_folder`: the crop's
    folder goes to `tempfile.gettempdir()` when its drive has the crop's
    uncompressed size plus `ZIP_WORK_MARGIN` free, else next to the
    output (`SCRATCH_PREFIX`). `_crop_zipped`: write, verify (stored in
    the record), TIFF from the folder (`export_tiff(tiff_path=...)`),
    `pack_zip` (sorted metadata first, STORED, `force_zip64` per entry as
    ozx-tck's valid files, comment `{"ome": {"version", "zipFile":
    {"centralDirectory": {"jsonFirst": true}}}}`; written as `.part`,
    then `os.replace`), then `_zip_problems` (same files and sizes, layout,
    `_crop_problems` read through the zip; a failure is written into the
    zip's record by `_rewrite_zip_record`). `write_crop` to a zip path
    packs too; `verify_crop(record=True)` on a zip rewrites it. The CLI
    uses `crop_box` for zipped crops; the notebook still calls
    `write_crop` + `verify_crop` (correct, but the zip is written twice).
  - Collections: `collection_images` reads the bioformats2raw layout
    (OME group "series", else 0, 1, ...) and 0.6 scenes (paths named by
    the scene's transformations); only plain inside paths (`_LEVEL_NAME`).
    `OmeZarrImage.path` stays the collection; `inner` names the image;
    `image_stem` adds it; `_image_identity` and the record
    (`source_image`, `source_collection`) carry it, so Overwrite never
    replaces another image's crop. The window asks with `QInputDialog`
    (`_choose_inner_image`), before *What to load* (not inside it, as the
    plan said: `LoadDialog` is built around one opened image).
  - Independent checks: `dev/formats_0_21_check.py` with
    `formats_0_21_ngff_side.py` (an environment with ngff_zarr 0.43.0
    and jsonschema 4.26.0), the official 0.6 JSON schemas, ozx-tck, and two
    public .ozx samples (see its docstring; results in CHANGELOG/notes).

- **Moving regions (0.17.0).** `CropBox.track` = {time: {axis: shift
  µm}}; `ranges` are the region at shift 0 (its size). `track_shift`
  interpolates linearly between entries (a missed timepoint lies between
  its neighbours) and keeps the first/last entry outside them;
  `ranges_at` / `set_ranges_at` are the region at a time and an edit
  there (same size: that timepoint's shift; another size: every
  timepoint keeps its centre). `plan_crop`: the moving axes (the spatial
  axes with ranges, `_moving_axes`) are not clipped to the image; they
  are widened by half a grid step of the coarsest kept level on each side
  and snapped outward, then each output timepoint's shift is rounded to
  that grid (`_shift_voxels`), so every kept level stays exact and the
  region asked lies inside the crop at every timepoint (rounding moves it
  by at most half a step). The plan's `starts`/`stops`/`snapped_ranges`
  are the region at the first output timepoint, and `CropPlan.shifts`
  (per kept level, output timepoints × axes, whole voxels of that level)
  are relative to it: so the OME translation (the first timepoint's
  place) and its check needed no change. Levels that also shrink time end
  the kept levels (as with every n-th timepoint). `_read_source` reads
  each output timepoint of a block at its place, the part outside the
  source as the fill value; the check's `_expected_moving_block` gets the
  places from `snapped_ranges_at` (µm → level-0 voxels → the level's), not
  from `plan.shifts`, so the two are independent. Positions per timepoint:
  `moving_origins` → the record's `moving_region` and
  `<crop>_positions.csv` (`write_positions_csv`; a fixed crop deletes an
  old one of the same name). Views: `OrthoViews.box_ranges` is a property:
  the ranges as cropped at the timepoint shown (`box_moves` from the
  panel: `functools.partial(snapped_ranges_at, image, plan)`), so every
  drawing reads the right place; on a timepoint change `_follow_time`
  also redraws the rectangles, 3D frames, clipping and masking, and calls
  `on_time_changed` (the panel's number fields and plan). Edits go
  through `set_ranges_at` at `time_shown()`; the number fields send only
  the axes whose (rounded) values changed. `autocrop.moving_region`: size
  = the largest box found + 2 × margin (cut to the image / region
  searched), centres = the boxes' centres, moved back inside where needed;
  the track has the timepoints found only.
- **Outlines per timepoint (0.17.0).** The user chose (2026-11-09) a
  different outline per timepoint over one shape that travels with the
  region, and to keep the whole region where the object was missed.
  `CropBox.timed_outlines` = {time: outlines}, at the object's positions
  then (absolute µm: moving or resizing the region does not move them).
  The crop: `_timed_block_mask` calls `outline_mask` per output timepoint
  with that timepoint's outlines at the region's place then (the writer's
  places from `plan.shifts`, the check's from `_place_from_ranges`); no
  outlines then = all kept. Silhouettes: `find_object(timed_silhouettes=
  True)` runs `_silhouette_polygons` (split out of `silhouettes`) on each
  timepoint's own mask and keeps only the polygons; `timed_silhouettes`
  gives each the region's depth range as cropped then (what a "selected
  region" projection shows, so that it can be edited there). Views:
  `OrthoViews.boxes` are `box_at(box, time)` copies while a box has
  outlines per timepoint (`_boxes_at`, renewed in `_set_cursor` before
  anything is drawn), so all outline drawing is unchanged; the panel's
  edits go to that timepoint's outlines (`_edit_outlines`). Regions file:
  a JSON file beside the CSV (Excel cuts cells at 32767 characters).

- **Seeds and silhouettes (0.16.0, Find a bright object).** The user
  chose (2026-11-08) "click seeds" over drawing a rough region per object,
  and silhouettes in three views over a plain box. Seeds are clicks, not a
  napari Points layer one edits: the four views are separate napari
  models, and a Points layer would only take clicks in XY. Each 2D view
  has a helper Points layer (`View.seeds`, crosses kept `SEED_SIZE`
  screen pixels wide in `_update_widths`) and a mouse handler
  (`OrthoViews._seed_click`, a generator as napari's drag handlers are)
  that reports a click without a drag (`CLICK_SLOP` 4 px), with the
  pan/zoom tool only, through `on_seed_click`: the view's two axes from
  the click, the slice and timepoint from the cursor. Every seed is drawn
  in every view, whatever the slice (it marks which objects have one).
  `find_object(seed=…)` picks, at the seed's timepoint, the thresholded
  piece under the seed or the nearest one of `MIN_VOXELS` or more within
  `SEED_REACH` = 6 voxels of the level searched (`_chosen_piece`, on a
  window around the seed), and at the other timepoints, outwards from the
  seed's (forwards, then backwards from the seed's mask), the piece that
  overlaps the neighbouring timepoint's object most (or the nearest within
  its size); where it is lost, the timepoints beyond are not searched.
  **The seed's threshold** (`_seed_threshold`): the visible check found a
  seed on the demo's embryo 5 giving nothing: the whole volume's Otsu
  threshold (355 on the smoothed nuclei) held 98 % of the bright embryo 6
  but 13 % of embryo 5, in specks of single nuclei. Halfway between the
  background and the brightness at the seed (206-222) found it but joined
  the tissue (its 99th percentile is 215) and the other embryos (regions
  6-20 × the embryo). The size of the seed's piece against the threshold
  (`dev/seeds_threshold_curve.py`, 2026-11-08; the
  regions per embryo: `seeds_all_embryos.py`) showed why:
  embryos 1, 4 and 6 grow 2-9 % per step until they join the tissue, then
  50-500 % in one step; embryos 2, 3 and 5 join the tissue or a neighbour
  before they are whole, so no single threshold separates them. Chosen:
  MSER's idea (Matas et al. 2002, "Robust wide baseline stereo from
  maximally stable extremal regions"): of `SEED_LEVELS` = 12 levels from
  min(automatic, 90th percentile within SEED_REACH of the seed) down to
  the growing floor, the one where the piece's size changes least
  relative to it ((below − above) / size), leaving out pieces over
  `SEED_MAX_SHARE` = 25 % of the volume (without that limit the "all
  joined" piece at the floor won). The followed timepoints use the seed's
  threshold relative to their own background median (0) and the 90th
  percentile inside the followed mask (1) (`relative`, never below the
  floor nor above the automatic threshold), since each timepoint's own
  automatic threshold can lie above a dim object. Result on the demo
  (growing off, margin 10 µm): embryos 1, 4, 6 held 100 % in regions 1.5-
  1.65 × their true box; 2 cut (40 %), 3 and 5 with neighbours. *Fit to
  seeds* searches the loaded part once per seed (so N seeds read N times,
  ~3 s each on the demo) and gives two seeds one region only if their
  objects' masks (all timepoints) overlap by more than half of the two
  together **and** each holds the other's seed (or comes within
  SEED_REACH; `mask_holds`): with overlap alone, embryo 5's piece (mostly
  embryo 6) was reported as "the same object" as embryo 6's region, which
  held none of embryo 5. A region whose object holds another seed whose
  own object differs gets a note: no threshold separates them; draw a
  rough region around each and use *Fit all regions*. `silhouettes` projects the mask
  (union over time, kept by `keep_mask`) along z, y and x, grows it by
  the margin + one voxel of the level (as the box), takes the convex hull
  if it falls apart (an object that moved a lot), and its contour
  (`find_contours`, `approximate_polygon` at 0.5 voxel) as a silhouette
  polygon whose depth range is the region's range (the crop code's
  silhouettes, `_silhouette_mask`). They replace the region's polygons;
  *Undo fit* keeps ("changed", region, ranges, outlines) or ("added",
  region) and puts back or removes.
  After the review of 2026-11-08 (8 agents, 15 findings, all confirmed and
  fixed): silhouettes map voxel j of the level searched to level 0's
  voxels f*j … f*j + f − 1 (centre f*j + (f − 1)/2, level 0's
  translation), as the box and the crop do, never the level's own
  translation (eubi-bridge levels repeat level 0's, which shifted outlines
  by (f − 1)/2 voxels); and they are the contour of the continuous
  distance field (EDT scaled so that 1.0 = margin/voxel + 1.5 voxels per
  axis, plus `tolerance`/min(g) so that approximate_polygon never cuts in),
  not of a thresholded mask (which rounded the margin down to whole
  voxels). Undo entries carry the region's state right after the fit
  (`_state`: name, ranges, outlines, channels, margin, t_step); a region
  that differs at Undo is left alone. A click becomes a seed after
  `QApplication.doubleClickInterval()` (`OrthoViews.seed_delay_ms`; 0 in
  tests), and a double-click (`mouse_double_click_callbacks`) cancels it;
  seeds are refused while a projection is shown; `Section.on_show` ends
  *Add seeds* when its part is closed.

- **Save as TIFF… (0.15.0).** The user wanted a button that saves the
  selected region as a TIFF without an OME-Zarr crop.
  `vc.save_region_tiff` plans the region down to the chosen level
  (`tiff_plan`: `plan_crop(n_levels=level + 1)`, so min_size does not
  apply and a small region has the level too), writes only that level
  with the crop code (`_write_crop_into` with a one-level plan: outlines,
  channels and every n-th timepoint apply) into a temporary folder in the
  output folder (not memory: a full-resolution region can be larger than
  the memory), checks it against the source, writes the ImageJ TIFF
  (`write_tiff`), checks sample slices, and deletes the folder. The name is
  that of *Also save as TIFF* (`<image>_<region>_level<k>.tif`), so the
  overwrite rules are the same. `crop_box(tiff_level=…)` stays for the
  notebook and the command line.
- **Wheel (0.15.0).** `crop_gui.guard_wheel` gives every spin box and combo
  box of the panel `Qt.StrongFocus` (the wheel no longer gives them the
  focus) and an event filter that passes wheel events to the panel's
  scroll bar unless the field has the focus (called after `_build_ui` and
  after each `_load`, for the fields made per image).

- **Find a bright object (0.14.0, experimental).** `autocrop.find_object`
  works on a coarse level (`choose_level`: the finest with at most
  `MAX_VOXELS` = 20 M voxels per timepoint in the search area), one
  timepoint at a time: Gaussian smoothing (1 voxel), an automatic
  threshold (scikit-image's Otsu, Triangle or Li), connected pieces
  (`scipy.ndimage.label`), and the piece with the most signal (sum of
  value minus threshold; pieces under `MIN_VOXELS` ignored). Each
  timepoint has its own threshold (bleaching, changes in brightness). An
  automatic threshold always splits a volume in two, even pure noise, so a
  piece counts only if its mean is at least `MIN_CONTRAST` = 3 robust
  standard deviations (1.4826 × MAD) of the voxels below the threshold
  above their median; found in the tests, where pure-noise timepoints
  otherwise gave an "object". The threshold only finds the object: with
  `grow` (the window's *Dim parts too*, on at first) its extent comes from
  growing it (`binary_propagation` in the smoothed data) down to the
  background's median + 3 raw noise levels (a floor at the smoothed noise
  followed the blur several voxels out). The review of 2026-11-07 showed
  why: Otsu put the threshold inside an object whose light fades with
  depth and cut off 17-64 % of it, and kept only one of two bright parts
  joined by dimmer tissue (54 % lost). Growing takes in everything above
  the background that touches the object, so it is an option: off for
  objects in bright tissue (the embryo demo). Exact zeros in quantity
  (> 5 %; the padding of fused volumes) are left out of the threshold,
  the background and the growing, else the recorded area itself became
  the object. The box is padded by one voxel of the level searched (edge
  voxels are only partly object) and put on level 0's grid as crops are
  (`downsampling_factors`), and the search area goes the other way;
  levels that shrink time are never searched (`usable_levels`). The
  region is the union of the timepoints' boxes plus the margin, within the
  image, with the t range of the timepoints where the object was found
  (frame centres). Measured on the embryo demo (`embryo_tissue_demo`,
  true labels; growing off): whole image, level 1, 0.7 s, the brightest
  embryo; searched around each embryo (true box + 40 µm), sides clear of
  the tissue within 2-16 µm of the truth, sides touching tissue out to the
  search region, and the hollow "cyst" cut short on channel 0 (with
  growing it is held). The window (`autocrop_panel.ObjectFinder`)
  searches all of the selected region (its own ranges within the image,
  not only the part loaded for the views: the search reads the image
  itself) and shrinks it, with *Undo fit*, or the loaded part and adds a
  region; it reads with a progress window and Cancel, in the GUI thread
  (as reading files into memory does), ignoring a second *Fit* meanwhile.

- **The napari plugin leaves the user's window alone (0.13.0).** The
  prototype's dock turned the window it was opened in into the tool:
  opening an image removed the window's layers, napari's 2D/3D, roll,
  transpose and grid buttons were hidden, and nothing was undone when the
  dock was closed. The user chose (2026-11-07) a small panel that opens
  Volume Cropper in **its own window** (`crop_gui.open_window`), plus *Use
  this window…*, which warns first and then does what the prototype did.
  napari gives plugins a "public-only" proxy of the viewer (and of
  layers), which warns on every use of napari's private parts ("no
  earlier than 0.10.0"; still only a warning in 0.10.0). The panel and
  the commands use only public parts (layers, selection); the tool's own
  windows are made by the tool, so they are not proxied. Only *Use this
  window…* unwraps the proxy (`__wrapped__`, a dunder, which gives no
  warning). A command gets the viewer through its type annotation
  (`viewer: napari.viewer.Viewer`, injected by napari's app-model); the
  widget must be a class whose `__init__` takes `napari_viewer` (a plain
  function was not given the viewer in napari 0.9.2). napari's
  layer-list context menu is not open to plugins in 0.9.2, so the command
  is in *Layers › Data* (`enablement: num_selected_image_layers >= 1`).
- **The plugin is installed from the tool folder, package only (0.13.0).**
  `pip install -e` with setuptools maps each installed module to its file
  by name (a static "finder"); the prototype installed the tool's flat
  modules that way, so a release adding a module would have broken the
  plugin until everyone reinstalled, and `crop_gui` etc. became importable
  names everywhere in the environment. Now only `napari_volume_cropper` is
  installed; `_tool.import_tool()` puts the tool folder (the package's
  parent, not resolved: on Windows that would turn a drive letter into a UNC path) at
  the front of `sys.path` when the plugin is first used, as the launchers
  do, and checks that `crop_gui` came from there. npe2 reads `napari.yaml`
  from the folder, so manifest changes need no reinstall either (changes
  to `pyproject.toml` do). The version napari lists is the one at install
  time; the panel reads the current one from `volume_cropper/core.py`.
  `--no-build-isolation --no-index`: with build isolation pip downloads
  setuptools and writes `napari_volume_cropper.egg-info` into the project
  folder (the shared folder!); without it, everything goes to temporary
  folders (checked by `dev/plugin_install_test.py`).
  Bytecode: the launchers set `PYTHONDONTWRITEBYTECODE`; the installer
  sets it as a conda environment variable (`conda env config vars`), so a
  plain `napari` in the activated environment writes nothing into the
  tool folder; `_tool.import_tool` also sets `sys.dont_write_bytecode`
  (too late only for the plugin package's own two small files, when
  Python is started without activating the environment).
- **Layers in memory (0.13.0).** `file_images.LayerReader` reads napari
  layers like a file: axes from axis labels that name t/c/z/y/x (napari's
  default labels "-1", "-2" … say nothing); the others are guessed by
  position (from the end x, y, z, then t; with five axes c, t as in Fiji's
  TCZYX), and the guesses other than y and x stay changeable in *What to
  load* (`Scene.axes` has "?", `Scene.axes_guess` the guess, which
  `default_axes` uses): a 4D layer may be TZYX or CZYX, a 3D one ZYX or a
  2D movie. Voxel size from the layers' scale even when their unit is
  napari's default "pixel" (then only the unit is unknown, micrometre
  assumed; a scale of all 1 counts as unknown); position (translate /
  scale, as Fiji's origin); names, colours and contrast limits. Several
  layers must agree in shape, data type, scale and translate (channels
  must lie on one another as napari shows them), and rotated, sheared or
  affine-transformed layers are refused. It holds the
  layers by **weak reference** (a layer deleted in napari is freed; reading
  it then raises `LayersGone`, and *What to load…* says so), except for
  *Use this window…* (keep=True), where the tool itself removes them.
  Lazy (dask) data are read a **whole chunk along z** at a time and the
  last block is kept: `load` asks for 16 planes at a time, and a reader
  behind a chunk of a whole z-stack would otherwise read it again for each
  (measured in `test_plugin.py`: one read per chunk); the block is let go
  before the next one is read and once the full resolution is read
  (`release`, called by `load`), and *What to load*'s memory estimate
  includes it (`extra_memory`). The default path
  (crop names, *Save regions*) is the layers' file if they share one,
  otherwise `~/<layer name>`, never the current folder (which may be the
  shared tool folder). `fi.source_file` offers to open the layers' own file
  when the tool can (OME-Zarr: lazy, no copy).
- **Windows are forgotten when closed (0.13.0).** `_WINDOWS` used to only
  grow, so each closed window kept its image (possibly gigabytes read into
  memory). Removing the window from `_WINDOWS` is not enough: a napari
  window closed with its X (or File › Close Window) leaves the viewer, its
  layers and the slicer alive; only napari's `Viewer.close()` lets them
  go (found by the review of 2026-11-07, checked in `test_plugin.py`).
  `keep_window` therefore watches each Volume Cropper window
  (`_CloseGuard`, an event filter on the main window): a close that does
  not come from `Viewer.close()` is ignored and `Viewer.close()` is
  scheduled instead (napari's own "Close window?" question is asked first
  for the X, as napari would). During a crop, the X asks whether to stop
  it ("No" keeps the window and the crop); when napari closes the window
  itself (quitting: the viewer is torn down already) nothing can be kept,
  so the crop is stopped without asking, and waited for. When Qt deletes
  the window, `_forget_window` drops it. `attach_panel` makes a napari
  window a Volume Cropper window (views, panel in the splitter,
  `keep_window`), for `open_window` and the plugin's *Use this window…*
  (whose panel must not stay in a napari dock: closing the dock deleted
  it). `_READING` stops a second read into memory while one runs (its
  progress dialog lets other windows work, e.g. napari's own).
- **One copy of the tool per napari session (0.13.0).** The launchers run
  `volume_cropper/__main__.py` (as `python -m volume_cropper` does), which
  imports `volume_cropper.crop_gui` normally, so the running window is the module
  the plugin finds. Without care, the plugin's own import (napari's menus
  are in every window, the tool's own too) could load a second copy that
  knew none of the windows. `_tool.import_tool` reuses the running
  `volume_cropper.crop_gui` from the tool folder; it refuses any tool module
  (`TOOL_MODULES`) that came from another copy of the tool (e.g. a local copy
  started with its own launcher while the plugin points to the shared folder).

- **Files in memory (0.12.0).** `file_images.load` reads the chosen part
  (scene, channels, ranges) one z chunk at a time into zarr arrays in a
  `MemoryStore` (uncompressed; chunks 1 × 1 × 16 × 512 × 512: measured on a
  1024 × 1024 × 200 stack, a slice in any view took 2-5 ms, one plane per
  chunk made XZ/YZ 5-10 times slower), then the pyramid from it
  (`plan_levels`: x and y halve until both are ≤ 256; z halves once its
  voxels would otherwise be thinner than x's; `mean_downsample` adds
  strided views in a wider integer type, rounding halves up: 0.2 s instead
  of numpy's reshape-mean 1.1 s for 160 MB). It returns an `OmeZarrImage`
  with OME-Zarr 0.5 metadata in memory and `source` (the file, the scene,
  what was read, the calibration and where each value came from), so the
  views, regions and crop code are unchanged. Axes are always (t), (c), z,
  y, x: a 2D image gets one z plane, as thick as a pixel is wide, which
  `OrthoViews.flat` turns into the XY-only view. Positions are the file's:
  voxel 0 of the part read sits at (first index + origin) × voxel size,
  the origin from Fiji's xorigin/yorigin/zorigin (the inverse of what
  `_imagej_metadata` writes), so a crop TIFF opened again sits where it
  was cut. Axes the file does not name (tifffile's I, Q) are "?" and
  chosen in the dialog (default z, then t, then c); RGB samples become
  channels. Calibration missing from the file: the dialog says so, and 1
  is assumed (recorded as "assumed").
- **Crops of files (0.12.0).** `crop_box` → `_crop_file_image`: the plan
  keeps full resolution only (`n_levels=1`, so the region is not widened
  for coarser levels); the crop is written with the OME-Zarr code into a
  `MemoryStore` (`_write_crop_into`), checked there (`_crop_problems`, the
  same checks as `verify_crop`), then saved with `write_tiff` (kind
  "imagej" = Fiji hyperstack, or "ome" = OME-TIFF with voxel size, frame
  interval, channel names and colours, and the record in the Image's
  Description) and compared voxel by voxel with the crop in memory
  (`tiff_problems(..., n_planes=None)`). The kind follows the file
  (`output_suffix`). zstd (`tiff_options`): compression "zstd" with
  predictor for integers; BigTIFF when the uncompressed size passes 4 GB
  (tifffile does not switch by itself for compressed files and fails at 4
  GiB, tested). Without imagecodecs tifffile writes zstd through Python
  3.14's `compression.zstd`. Tested in the user's Fiji (ImageJ 1.54p,
  Bio-Formats 8.5.0, headless): ImageJ's own reader opens uncompressed and
  deflate TIFFs but not zstd (File › Open then shows the Bio-Formats
  dialog); Bio-Formats reads zstd ImageJ TIFFs and OME-TIFFs with the right
  pixels and calibration; every .ome.tif goes through Bio-Formats quietly.
  On 210 MB of synthetic data, deflate + predictor and zstd + predictor
  were both 1.85× smaller; Fiji opened them in about 4.5-7.5 s instead of
  0.7 s.

- **TIFF export (0.10.0).** One level of the *written* crop is read back
  (`export_tiff`), so the TIFF is exactly what was cropped and checked. It
  is an ImageJ hyperstack (tifffile `kind="imagej"`, TZCYX), the format
  Fiji opens natively; ImageJ hyperstacks are uncompressed and never
  BigTIFF. Above 4 GB tifffile writes ImageJ's own layout (one image
  directory, data contiguous; it warns "truncating ImageJ file", which is
  silenced): checked on 2026-11-06 with a 4.6 GB synthetic file, written
  in 5 s and read back fully by tifffile; Fiji (*File › Open*) is to be
  checked by the user, other readers may see one plane only, so the
  window asks first. Calibration as ImageJ's `FileOpener` reads it
  (checked in its source, 2026-11-06): `unit` ("um", shown as µm),
  `spacing`, X/YResolution, `finterval` with `tunit` (the OME time unit;
  units ASCII-only, other characters as `\uXXXX`, which ImageJ's
  properties parser reads; "µm" becomes "um"). Origin (a convention
  choice, 2026-11-06): ImageJ maps pixel coordinate p to (p − origin) ×
  size and pixel i spans i…i+1 (its centroids and ROIs use i + 0.5), so
  `xorigin`/`yorigin` = 0.5 − translation / size and `zorigin` =
  −translation / size: Fiji's *measurements* equal the OME/napari
  positions; Fiji's status bar shows pixel corners (half a voxel less in
  x and y), as for every ImageJ image. The other choice (−translation /
  size: status bar = OME, measurements +½ voxel) was rejected because
  measured numbers go into analyses. Channels: the IJ metadata tags
  `Labels` (one per plane, up to `TIFF_MAX_LABELS`), `LUTs` (OMERO
  colours; ImageJ uses them only with several channels, so one 16-bit or
  float channel also gets the TIFF ColorMap tag, as ImageJ saves it; one
  8-bit channel stays grey, because a ColorMap makes it "8-bit Color"),
  `Ranges` (OMERO windows; one channel: also `min`/`max`) and `Info`
  (marker line + the crop's `crop_provenance` as JSON). Reading: whole
  slices of all channels, at most `TIFF_READ_BYTES`, in whole z chunks;
  a chunk thicker than that is read whole up to `TIFF_READ_MAX` without
  reading ahead (thinner reads would read each chunk up to 8 times: the
  review measured 8× for 9-slice reads of 64-slice chunks). Written to
  `<name>.tif.part`, renamed when complete. Overwrite: only TIFFs whose
  *Info* starts with `TIFF_MARK`, and of the same box (as for crops);
  overwriting a crop first deletes that box's TIFFs of any level
  (`remove_old_tiffs`; exact name pattern plus the record's box name), so
  no stale TIFF remains even if the new export is off, at another level,
  or cancelled. A region with fewer levels than asked gives its coarsest
  (`tiff_level_for`), reported in the plan text, the report and the file
  name, rather than refusing: one level choice serves all regions. Only
  uint8, uint16 and float32 (`check_tiff_image`, which also checks the
  axes, before anything is written). int16 is refused although ImageJ
  could hold it: ImageJ stores it shifted by 32768, so its display ranges
  would need shifting too (not needed so far). The TIFF check
  (`verify_tiff`) compares data types by name (a big-endian 0.4 crop gives
  a little-endian TIFF with the same values) and NaN as equal; its result
  is `CropResult.tiff_problems`, reported apart from the crop's check.
  Reviewed 2026-11-06 (four reviewers, each checked by a skeptic): the
  fixes above came from that review.

- **OME-Zarr 0.4 (0.8.0).** The user's decisions (2026-11-02): a crop
  keeps its source's version (0.4 gives 0.4, the "same format" rule);
  only 0.4 is added, not 0.1–0.3; an axis without a type is refused with
  a message rather than guessed from its name (guessing could mix up axes
  unnoticed; eubi-bridge always writes the types). The version is told by
  the storage (Zarr v2 with "multiscales" in the attributes = 0.4; Zarr v3
  with "ome" = 0.5), because 0.4 only recommends its "version" field
  (OME-Zarr 0.4 specification §3.4, <https://ngff.openmicroscopy.org/0.4/>).
  Zarr v2 has no shards, so a copy task writes a block of whole chunks
  (`BLOCK_BYTES`, 32 MB; never two tasks in one chunk file). eubi-bridge's
  `OME/METADATA.ome.xml` (the full image's size) is not copied, as for 0.5
  crops: only the bioformats2raw layout (§3.2) defines that file, and the
  crops pass the official 0.4 schemas without it (the user asked,
  2026-11-02). Other software's root attributes (e.g. `eubi_bridge`) are
  not copied either; `crop_provenance` records the source.
- **Outlines in 3D (0.7.0): one shape through all polygons.** The user's
  aim: crop out an irregular object, and only that object, from polygons
  drawn on a few slices of the XY, XZ and YZ views. A first draft
  interpolated each view on its own (signed distance maps, Raya and Udupa,
  IEEE Trans. Med. Imaging 9: 32-42, 1990) and kept a voxel inside all
  views' shapes: on a synthetic curved tube that kept only 24 % of the
  object, because each view's shape ends at its own first and last polygon
  and cuts the others. Now all polygons define one function of (z, y, x),
  a variational implicit surface (Turk and O'Brien, Proc. SIGGRAPH 1999,
  335-342; ACM Trans. Graph. 21: 855-873, 2002): points on each polygon
  (value 0, every 3 grid steps along its edge) and one grid step inside
  and outside it, along its outward normal in its own plane (values -1 and
  +1 steps), fitted with scipy's `RBFInterpolator` (polynomial degree 1).
  The shape is where the function is negative.
  - **Cubic kernel (r³), not linear (r):** with r, the same 20-voxel
    square on two slices 40 and 60 voxels apart kept 50 % and 0 % of its
    area in the middle (the surface sagged like a soap film between two
    rings), and polygons side by side on two slices left a gap. With r³
    the square stays a straight prism at any distance and such polygons
    are joined. On the tube (3 XY, 2 XZ, 2 YZ polygons) both did about
    equally well: r 90.6 % kept and 2.8 % extra, r³ 91.7 % and 2.3 %; with
    hand-drawn wobble (±1 voxel) 89.8 % kept for both; with views that
    disagree by 10 % in size, r 84.3/8.0 %, r³ 82.5/9.2 %. Weak spot of r³:
    a square shifted by its full width from one drawn slice to the next
    (20 slices apart) keeps 64 % of a square's area in the middle (r:
    90 %); sharp corners are rounded between drawn slices (a 40-voxel
    square by up to 2.5 voxels). Adding inside/outside distances deeper
    in each polygon's plane did not help (it made the sag worse).
  - **Polygons in one view only (0.17.1):** the usual case (outlines on a
    few XY slices) does not need the surface, and the review of 0.17.0
    found it fragile there, as any smooth spline: equal circles on two
    slices bulged to 1.45 times their area in between (squares, which r³
    reproduces exactly, stayed straight); a slightly smaller circle drawn
    1 µm from another (r 17 vs 20 µm) emptied the shape over the next
    50 µm (the steep change between the two close slices carried on), a
    larger one inflated it 1.5-2.2 times. Now such polygons get
    shape-based interpolation after all (`_one_view_field`; Raya and
    Udupa 1990, above): the signed distance to the filled polygons of each
    drawn slice, on a grid 4 times finer (`SLICE_REFINE`), blended
    linearly between neighbouring drawn slices. Where two neighbouring
    slices' filled areas hardly overlap, both maps are first moved so that
    the areas' centres line up (the centre moves linearly; an object that
    shifts sideways by more than its width would otherwise pinch off):
    fully at no overlap, not at all from an overlap of half the smaller
    area (always lining up pinched a part that stays put while another
    part of the polygon ends: 10 % of it kept midway; now 99 %). A slice's
    only filled area is never removed as a speck; a polygon that covers no
    point of the fine grid makes the shape narrow to nothing there. The
    views' preview, on its coarser grid, can differ from the crop by about
    a voxel at the edge (4.6 % of a small object's voxels in one test).
    The cut-off problem of the first draft does not arise
    within one view. Measured (`reviews/volume_cropper_0.17.0/`): equal
    circles 1.00 everywhere; the close sloppy pair 0.77/0.86/0.94 of the
    disc at 10/25/40 µm after it; a lasso overshooting by 5 % of a turn,
    a figure-8 and a shaky lasso match Fiji's fill on their own slices
    within 5 pixels.
  - **The surface's points come from the filled polygon (0.17.1):** each
    polygon is filled (`fill_polygon`: as Fiji fills it, even-odd, plus what
    the outline goes around twice, since 0.17.2), holes and specks smaller
    than a grid cell are removed, the distance map is smoothed by half a
    grid step, and its 0-line gives the points (value 0) and the points
    one step to either side (value: their distance). Taken along the drawn
    line with one normal direction per polygon, as before, an overshooting
    lasso kept 43 % of itself, a figure-8 half, and a shaky lasso (600
    points, 0.7 µm tremor) on the tube kept 49 % with 68 % extra in 5
    pieces; now 90.0 % and 2.7 % in one piece (clean outlines: 91.9 % and
    2.3 %, as before). A polygon less than about two grid steps thick has
    no inner side: it gets the old points (its line, and one step to
    either side along its normals: a slab), so the shape still passes
    through it. At most `MAX_FIT_POINTS` (4000) points are fitted
    (spaced further apart if needed: the fit's time grows with their
    number cubed; a 1500-point lasso with 1.5 µm tremor gave 16000 points
    and never finished, now 17 s).
  - **It ends at the outermost polygons** (the user: smoothing beyond the
    drawn outlines "is non-scientific"): nothing outside the polygons'
    extent along any axis is kept, and the crop is fitted to it
    (`outline_limits`, `_effective_ranges`). A box's only polygon, or one
    without a slice (boxes from before 0.7.0), reaches through the box, as
    before; such polygons and the shape combine with "inside all".
  - **Speed:** the function is evaluated on a grid of `FIELD_POINTS` (128)
    points along the polygons' longest extent, first 4 times coarser and
    then exactly only near the edge of the shape, and interpolated
    linearly to each voxel (`_shape_field`, cached per box). A 1.3 mm
    object with 12 polygons of 200 points: 6 s once per crop
    (`prepare_shape`, before the parallel copying), then about 0.3 s per
    192³ block. The views use the same function on a coarser grid
    (`VIEW_DETAIL`, 48 points; 1.3 s for that object): on the tube it
    differs from the crop in 0.6 % of the kept voxels, all within 1 voxel
    of the edge.
  - **Silhouettes (user's report 2026-10-30):** the user had drawn on a
    MIP, where an outline is the object's silhouette; taken as the outline
    on one slice, it disagreed with the other views and the shape got holes
    and dents. Now a polygon drawn on a projection keeps the projected
    depth range as its key and reaches through it (`_silhouette_mask`;
    "inside all" with the rest: the classic shape from silhouettes, or
    "visual hull"). On the embryo demo, three silhouettes kept 99.9-100 %
    of each object plus 13-39 % extra, seven slice polygons 85-99 % plus
    1-4 % (`dev/score_demo_embryos.py`).
  - **Views that disagree:** with consistent outlines the shape had no
    holes at any stage of drawing; with side views drawn 12 % too small or
    shifted by 12 µm, 3-12 slices per object had holes (the shape must pass
    through every polygon). RBF smoothing (up to 1000) barely helped. Now:
    holes are filled on the shape's grid (closed hollows, and holes on the
    grid's planes along every axis; filled points get the depth they were
    outside, on a grid 2 times finer, `HOLE_REFINE`); holes between the
    grid's planes can remain (about half as many; up to 20-40 µm across in
    those tests), so the window marks where the other views' polygons
    cross a slice (`crossing_points`) and warns when they do not meet by
    more than 3 voxels (`outline_disagreements`).
  - **Margin:** in voxels of the finest axis of level 0 (`pixel_size`);
    each block is computed with a rim as wide as the margin, then grown or
    shrunk with exact Euclidean distances between voxel centres in µm
    (`distance_transform_edt` with the voxel sizes), so blocks agree at
    their edges and anisotropic voxels count correctly.

- **Boxes are axis-aligned and never rotated.** A rotation by an arbitrary
  angle moves voxels off the grid, so the data would have to be resampled
  (interpolated, or nearest-neighbour, which duplicates and drops voxels).
  Only 90° turns and flips are exact, and they don't make an axis-aligned
  box any smaller. Masking with outlines already gives most of the disk
  saving without changing a single voxel value.
- **Snapping:** each box is enlarged by at most one voxel of the coarsest
  pyramid level kept, so that every level lines up exactly. Levels are kept
  while the crop is at least 64 voxels along its longest axis.
- **Writing:** one output shard per task (no two threads write the same
  shard); the OME metadata are written last, so an interrupted crop is
  recognisable; *Overwrite* replaces only a crop of the same box that this
  tool wrote.

## Files from elsewhere (0.23.0)

From the pre-publication review (2026-11-27): a crafted image or regions
file must not make the tool read other files, nor make Windows connect
to another computer (a network name such as `\\server\share` sends it the
user's login). Rules to keep:

- **Text from files is never shown as HTML.** Use `volume_cropper/qt_text.py`:
  its `QLabel` and `QMessageBox` (plain text; the message box's warning,
  critical, information and question answer as Qt's) instead of Qt's
  own; `literal()` for tooltips and the labels of Qt's dialogs
  (QInputDialog, QProgressDialog); `escape()` for names inside the tool's
  own HTML (`load_dialog._label`, the loading notice); `defang()` for
  names napari shows (layer names). Checks patch `crop_gui.QMessageBox.*`,
  the tool's class. `tests/test_qt_text.py`.
- **Files that name other files** are checked before they are opened:
  OME-TIFF sets (`_refuse_files_elsewhere`); Leica files
  (`_refuse_leica_names`; .xlef: `_open_experiment`, `_check_frames`,
  which fill liffile's caches `children`, `_lof_reference` and
  `_tif_reference` with checked files: inner workings of liffile
  2026.10.14, so check them again when liffile is updated); TIFFs never
  as NDTiff (`is_ndtiff=False`), folders file by file (`_multifile=False`).
  The tests in `test_microscope_files.py` and `test_file_images.py` watch
  every file opened (an audit hook).
- **Axis names and units** (`core._check_axis_text`): no < > & " or
  control characters, no name starting with = + - @. **Regions CSV:**
  names that start like a formula are written with a leading '
  (`_csv_text`, read back by `_from_csv_text`).
- Not changed (low; noted by the review): a path with "&" dropped onto
  the .bat; `.part` files in an output folder others can write to; a
  zip of compressed members (read whole); links inside image folders are
  followed on Linux and Mac. Each crop records its source's full path
  (provenance; the user's choice to keep it).

## napari internals used

`volume_cropper/ortho_views.py` relies on napari 0.9.2 details, which is why the
environment is pinned. After upgrading napari, run the tests and check
these:

- **The window:** it reaches into the main window's centre widget and
  `viewer.window._qt_viewer` to build the grid.
- **OpenGL query:** it asks OpenGL about its capabilities once through a
  hidden canvas before creating the extra views (otherwise napari's first
  query fails).
- **Hidden layers:** image layers are added hidden and shown only after the
  views are zoomed to fit. Otherwise napari loads a large region of the
  finest level at its default zoom.
- **Exact level:** it uses a one-level pyramid, not napari's
  `locked_data_level`, which loads the whole plane at once.
- **3D clipping:** plane positions are given in µm (world coordinates), not
  voxels as napari's docs say. `test_napari_clipping.py` checks this.
- **Key bindings:** these must be named functions. napari reads the name of
  every handler on each key press, and a `functools.partial` (which has no
  name) broke all shortcuts.
- **Axis labels:** every layer gets `axis_labels`, because napari derives
  the slider labels from the layers.
- **Overlays:** they use `translucent_no_depth` blending, so that data can't
  hide the 3D wireframes and lines can't hide each other.
- **Box lines:** these are napari *Vectors* layers (many short line pieces),
  which redraw fast. Their width is in µm, so it is reset when the zoom
  changes.
- **Slider feedback:** changing a display layer makes napari re-send the
  slider positions. The views ignore those events while they update their
  own lines (`_display_only`); otherwise one change set off about 80
  redraws (25 s instead of 1 s on real data).
- **Removing layers (crash fix, 2026-10-23):** napari deletes a removed
  layer's OpenGL objects at once, in whatever OpenGL context happens to be
  active. With four canvases that is often another view's (or none), and on
  Windows the program then crashed without a message ("access violation").
  It showed up when the 3D masking was switched to *hide* with clipping on.
  All removals therefore go through `_remove_layers`, which first makes that
  view's canvas the active one, as vispy itself does. The crash was
  reproduced in a visible window and is gone with the fix; without it, it
  returns at the same step.
- **Closing windows (0.13.0):** an event filter on
  `viewer.window._qt_window` turns closes into `Viewer.close()` (see
  *Windows are forgotten when closed*); for the X it shows napari's own
  `napari._qt.dialogs.confirm_close_dialog.ConfirmCloseDialog` if napari's
  setting *Confirm window or application closing* is on. The plugin's
  *Layers › Data* entry is greyed out by napari only when the *Layers*
  menu is opened (app-model updates the enablement on `aboutToShow`).
- **The XY view's data (0.18.0):** the *Levels* histogram of "XY view (as
  shown)" reads `layer._slice.image.raw` (what napari loaded for the
  view: the visible part, at the level shown, projected if a projection
  is on; napari's own histogram reads the same) when the layer's
  `set_data` event fires (`OrthoViews.on_xy_data`). A hidden layer loads
  nothing, so the panel then asks for the channel to be shown.
- **A layer added becomes the active one (0.18.0):** napari selects a
  layer inserted into a view's layer list and makes it active. The crop
  preview's projection layers are made when first needed, so they put the
  previous active layer back (the regions' shapes layer, which the drawing
  tools and keys act on). Seen in the visible check: the layer controls
  showed the new image layer instead of *crop regions*.
- **napari's panels replaced (0.18.0):** the panel's Channels part,
  Levels' auto-contrast (`layer.reset_contrast_limits()` for *once*,
  `layer.auto_contrast` for *continuous*, as napari's
  `AutoScaleButtons`; *reset* = `channel_contrast` and gamma 1) and the
  drawing tool buttons (`DRAWING_TOOLS`, napari Shapes modes; push
  buttons, because napari's style draws `QToolButton`s flat and shows a
  checked `QPushButton` highlighted). The setting is `napari_layer_panels`
  (default off); the old key `napari_panels` (default on) is ignored, so
  that saved settings do not keep the panels shown. `CropWidget.show_checks`
  stays (a property: the Channels rows' tick boxes) for the older checks.
- **Undo of the regions (0.18.0):** snapshot-based, in
  `CropWidget._record_regions`, called by `_refresh` (every change to the
  regions ends there). A deep copy and its `repr` are kept; a different
  `repr` pushes the previous snapshot. Not recorded while a 3D dot is
  dragged (`_drag_3d`) or while restoring; changes less than
  `UNDO_MERGE_S` apart merge. Each region carries `_vc_uid` (an attribute
  outside the dataclass fields, so not in `repr`) and `_box_objects` keeps
  every region object: `_restore_regions` writes old values into the same
  objects (`__dict__.update`) and the same list, because the Autocropper
  keeps references to regions (`seed_results`, its own Undo fit).
- **napari key bindings need named functions (0.18.0):** napari's key
  handling reads `__name__` of every function in a keymap
  (`_get_repeatable_shortcuts`); a `functools.partial` made every key
  press in that view raise. Bind closures (with a distinct `__name__`).
- **Points of a fixed size on screen in 3D (0.18.0):** not with
  `canvas_size_limits`. A huge size clipped by the upper limit made the
  3D view's home button / `reset_view` zoom out to nothing (napari counts
  a point's size in the scene's extent); a tiny size raised by the lower
  limit drew the markers mostly in their border colour (the seeds came
  out as black discs). Now the size in µm is pixels / the 3D camera's
  zoom, updated on its zoom event (`_size_3d_markers`), as the 2D seeds
  do. Points added to an empty Points layer get napari's default size,
  not the one the layer was made with: set the size after the data.
- **Opening at a chosen level (0.18.0):** `_zoom_to_level` after
  `reset_cameras` in `set_image` zooms the 2D views to 1 / the level's
  in-plane voxel (napari's rule, `level_for_pixel`, then picks that
  level), never further out than the fit.
- **Autocropper progress (0.18.0):** `find_object`'s `progress` gets
  fractions of a timepoint from `object_in(step=...)`, called per
  threshold in `_seed_threshold` (5 of 8 s of a seed's search on the
  demo); `step` also checks `cancel`. The panel shares one
  `QProgressDialog` across a Fit to seeds / Fit all regions.
- **Splitting an object between seeds (0.18.0):** after Fit to seeds'
  per-seed searches, `_fused_groups` links seeds whose objects are the
  same or hold each other's seed; `split_objects` then, per timepoint
  (from the first seed's, forwards, then backwards): the object = the
  pieces above the lowest of the group's thresholds (one's own, if set)
  under the seeds or overlapping the neighbour's parts, grown as
  `find_object` grows; markers = a ball (`MARKER_RADIUS`) at each seed at
  its timepoint, else the neighbour's part eroded by a voxel; landscape =
  -(0.5 brightness + 0.5 depth inside (EDT in µm)), both 0-1, plus
  `GAP_HEIGHT` x the gap lines' ridge (`gap_walls`: the wall on the drawn
  slice, fading over `GAP_REACH` slices); skimage `watershed` in the
  object's box (the wall's voxels removed, then given to the nearest
  part); each part through `_box_of` / `_finish` like a search. Gap lines
  live in the panel (`ObjectFinder.gap_lines`), drawn by `OrthoViews`
  (`_gap_draw`: a drag generator next to `_seed_click`; `mouse_pan` off
  while drawing, so a drag does not pan).
- **The Autocropper's own threshold (0.18.0):** `find_object(threshold=)`
  reaches `_object_in(fixed=)`, which uses it instead of the automatic,
  seed or followed threshold, at every timepoint, and never grows (that
  would undo it); the background, contrast and piece choice stay as
  before. Seed modes (`_seed_threshold(mode=)`) pick among the stable
  plateaus (local minima of the size change) the first from the top or
  the last; for them the levels start at the brightness near the seed
  even above the automatic threshold (where a bright core lies). The
  preview reads `search_volume` (same level, area and smoothing as a
  search of the loaded part, via the shared `_prepared`) once per
  image / channel / timepoint / loaded part; `SearchVolume.contours`
  (skimage `find_contours`, the 400 longest, the slab's maximum for a
  projection) feeds `OrthoViews.level_outline`, drawn by
  `_draw_level_lines` as dotted `Vectors` (`THRESHOLD_COLOR`, opaque:
  napari's default 0.7 was hard to see) on cursor, projection and zoom
  changes. Another timepoint gives no outline until the panel's 0.3 s
  timer has read it, so playing through time is not slowed. A fit of a
  smaller selected region may search a finer level than the preview's.
  The Levels marker (`set_marker`, `MARKER`) wins over another line
  within `MARKER_FIRST` px: matplotlib's events carry whole pixels, and
  1.4 px from the gamma line it could not be grabbed.
- **napari's `visible` setter always redraws (0.18.0):** in napari
  0.9.2 `Layer.visible = x` refreshes the layer (re-slicing it when x is
  True) and updates the scene graph even when the value stays the same,
  ~15 ms a time. The views' updates set it many times per region edit, so
  dragging a region in 3D took 0.5-1.2 s per mouse move (profiled with
  `dev/handles_lag_profile.py`). Use `set_visible(layer,
  x)`, which only sets a changed value. Likewise the crosshairs (a key in
  their metadata) and the 3D slice planes are only redrawn when they
  move, and `update_boxes` skips the 2D shapes, 3D clipping and masking,
  shading and preview while a 3D dot is dragged (`_drag_3d`): ~0.12 s per
  move since, of which ~60 ms is napari drawing the four canvases.
- **Dragging in the 3D view (0.18.0):** a viewer-level
  `mouse_drag_callbacks` generator on the 3D view (`_handle_drag`); napari
  gives each event `position` (µm, on a plane through the camera's
  centre), `view_direction` (the ray through the pixel, from
  `_calculate_view_direction`) and `camera_zoom` (pixels per µm), and
  updates the same event object as the drag goes on. Setting the camera's
  `mouse_pan` and `mouse_zoom` to False makes the vispy view
  non-interactive, so the camera does not turn while a dot is dragged
  (restored on release). The dots are a Points layer with
  `canvas_size_limits` = (14, 14) px and huge sizes, so they keep their
  size on screen; adding it restores the active layer (see above).
  A face goes where the mouse is (`drawn` = the box as cropped) while the
  other faces keep the panel's values (`held`): adding the mouse's
  movement to the held value failed for regions with polygons, whose
  crop is fitted to the polygons (held 2-12 µm, drawn 2.9-11.1 µm).
- **The window can't paint while napari reads (0.18.0):** napari 0.9.2
  slices (reads) in the main thread, during the call that changes a level
  and in later draw events. The "Loading…" box (`_LoadingNotice`, a
  frameless Qt.Tool window) is shown and painted (processEvents without
  user input) before the change, and `_check_loading` closes it when a
  100 ms timer has come on time for 0.6 s without bytes read (psutil), or
  for 2 s. Screen grabs over napari's OpenGL views come out black on this
  PC, so the check saves the box with `QWidget.grab()`.
- **Gamma's range (0.18.0):** napari's layer controls connect their gamma
  slider (0.2-2, `qt_contrast_limits.py`) both ways to `layer.gamma`, so
  a gamma set outside that range comes back clamped by the slider. The
  *Levels* gamma box and dashed line therefore stay within
  `levels.GAMMA_RANGE` = (0.2, 2). napari shows a value v at brightness
  ((v - low) / (high - low)) ** gamma; the dashed line sits where that is
  0.5 (`levels.midtone_value`).
- **Right panel:** the panel is not a napari dock but sits next to the
  views in a Qt splitter. The border of a napari dock on the right could not
  be dragged (napari 0.9.2 on Windows, also in plain napari).
- **2D views stay 2D:** switching XY to 3D made napari 0.9.2 fail on every
  redraw: it builds a wrongly shaped placeholder for an empty *Vectors*
  layer (the box lines) in a 3D-displayed view. The views therefore undo
  such changes (`_keep_layout`), and the line layers are hidden while
  empty.
- **Changing levels:** replacing a view's image layers makes napari reset
  its sliders and zoom. New layers are therefore added before the old ones
  are removed, the place is saved and restored around it (`_keep_place`),
  and the new layers are shown (and read) only afterwards.
- **Level shown:** napari fires a layer's `set_data` event whenever it
  loads data, also after switching level with the zoom; the titles and the
  "Auto" entry update from it.
- **Crash reports:** `volume_cropper/crop_gui.py` switches on Python's `faulthandler`, so a
  crash inside napari/OpenGL prints where it happened. `volume_cropper.bat`
  keeps its window open after any failure, including crashes (negative exit
  codes).
- **3D masking:** the 3D layers get a small array-like object
  (`MaskedVolume`) instead of an array. napari only needs `shape`, `dtype`,
  `ndim` and indexing. Removing a layer is slow, because napari then runs
  Python's garbage collection, so the masking layers are reused where
  possible.
- **Crop worker:** its handlers (`returned`, `errored`, `finished`) are
  passed when the worker is made (`_connect`). Connected afterwards,
  superqt also re-raises every error in the main thread, even a Cancel,
  and napari shows it as a second, red error message (a test script
  without napari's notification hooks is ended by it).
- **Hidden boxes:** a hidden box has no shape in the 2D shapes layers, so
  shape numbers differ from box numbers; `View.shape_boxes` maps them, and
  every shape event goes through it.
- **Projections:** these use napari's thick slices, set through
  `layer.projection_mode` together with `dims.margin_left` and
  `dims.margin_right` (in µm).
