# Volume Cropper: check scripts (not part of the tool)

Scripts used during development to check the GUI, kept
in git in this folder (`dev/`). They use the tool folder they are in (the
repository) and the synthetic demo data, which lives beside the repository
in a folder `volume_cropper_demo_data` (made by `make_demo_embryos.py` and
`make_demo_tails.py`; elsewhere: set the `VC_DEMO` environment variable).
Run them in the tool's environment (`volume_cropper`, made from
`env/environment.yml`; the plotting scripts also need matplotlib), from a
scratch folder: outputs go to `v03_out\` there (create it first), and
read-out image data stay there. An `OUT_DIR` must be a scratch folder
too: the scripts delete and rewrite their own outputs there (e.g. its
`crops` folder) without asking. Branches, tags and commits named here
are in the maintainer's development repository, which is not public.

**GUI checks.** These open a VISIBLE napari window and drive it; an
offscreen window once missed a real OpenGL crash. Since 0.15.0 the window
has no `min_size_spin`, `workers_spin`, `keep_position_check` or
`tiff_check` (use `crop_gui.MIN_LEVEL_SIZE`, `WORKERS`, `KEEP_POSITION`,
and `_crop(False, tiff_only=True)` for *Save as TIFF…*): the older scripts
that set them (e.g. `tiff_0_10_gui_test.py`, `view_options_gui_test.py`,
`fixes_0_6_gui_test.py`) check the versions they were written for.

**The plugin's environment** (default `napari-plugin-test`) is a clone of
the tool's environment with the plugin installed. Keep it in step when the
tool's environment gains packages: a clone made earlier lacked the ones
added later (matplotlib for *Levels*, the microscope readers), so the
plugin's tool window could not load. Copying the exact versions from
conda's cache fixes it: the lines of `conda list -n <tool env> --export`
that the other environment lacks, into
`conda install -n <plugin env> --offline --file LIST`.

| Script | Checks |
|---|---|
| `gui_v03_test.py [IMAGE]` | box lines, 3D masking modes, "open last crop in a new window", crop of a crop |
| `gui_timelapse_test.py` | a synthetic time-lapse; saves `v03_out/timelapse_XY.png` and `timelapse_3D.png` |
| `layout_test.py` | splitters, Focus, napari panels, the Auto level label |
| `fix_check.py TOOL_DIR` | the 2D/3D button error, dragging the right panel |
| `repro_hide.py` | the old crash when hiding blacked-out voxels with clipping |
| `channels_test.py IMAGE.zarr 0` or `all` | opening a large multichannel OME-Zarr with some channels: time, bytes read |
| `view_options_gui_test.py OUT_DIR` | projections through a slab (follows the slider) and chosen ranges, on two objects at different depths; hiding crosshairs and crop boxes; crop progress in bytes with a time estimate; *Keep position* off → crop starts at 0 (20 checks) |
| `cache_gui_test.py` | *Memory cache* (demo): size limit kept, showing a channel again reads 0 MB, *Clear cache* empties it, 0 = off; puts the user's napari cache setting back afterwards (7 checks) |
| `polygons_any_gui_test.py OUT_DIR` | outlines combine with "any" (the user's 2026-10-27 case: XY polygon longer in y than the YZ one): the box encloses both, the crop = voxels inside either (6 checks; on 0.4.0 two fail) |
| `outlines_0_7_gui_test.py OUT_DIR` | 0.7.0 crop regions and outlines in 3D: drawing goes into the selected region; New region, Finish region, Delete, deselecting; dashed outlines in every view, × marks, the disagreement warning, a silhouette on a MIP; shading, margin, polygon list, editing and deleting single polygons, a crop; saves `shading.png`, `margin.png` (27 checks) |
| `fixes_0_6_gui_test.py OUT_DIR` | the 0.6.0 review fixes (packages 1, 3, 4), grown package by package: names giving one file name, CSV mistakes and Excel CSVs, output inside the source, Overwrite vs another box's crop, the check report and its warning, single boxes hidden/shown (shapes, lines, 3D frames, edits reach the right box), questions before boxes are discarded or replaced, disk space, Cancel; saves `box_two_hidden.png` (23 checks) |
| `open_cost_real.py IMAGE old/whole/zthird/small` | one opening of a real image (channel 0, 3D off) in a fresh window: time and MB read, the estimate and the predicted vs shown levels, then a cursor move and 3D on. `old` = 0.4.0 behaviour (set `VC_TOOL` to a 0.4.0 copy). Run each choice in its own process (napari keeps what it read). |
| `raw_vs_decode.py IMAGE LEVEL CHANNEL` | network vs decompression: reads one channel's shard files as raw bytes, then decodes them from memory (no GUI) |
| `open_dialog_gui_test.py OUT_DIR` | 0.5.0 "What to load": the dialog, loaded regions shown at the right voxels and place, levels predicted = shown, Add inside the region, lazy contrast, 3D and masking on a region, crop after a restricted load = after a whole load, "Selected box"; 30 checks on a synthetic image |
| `halo_3d_check.py OUT_DIR [LABEL]` | 0.8.1, visible window, embryo demo with the trunk outlined: with *Clip 3D to selected region* the data given to napari are 0 at every voxel centre outside the region, without clipping the plain data (modes hide and dim; 4 checks); saves 3D pictures from above and tilted (`3d_<LABEL>_<mode>_<clip>.png`). `VC_TOOL` = 0.8.0 or older shows the halo bug (2 of 4 fail) |
| `v04_gui_test.py OUT_DIR` | 0.8.0 OME-Zarr 0.4, visible window: on a synthetic 3-channel 0.4 image the summary, the channel boxes, hiding a channel and a crop of two channels, checked (4 checks). Saves `synthetic_window.png`. `VC_TOOL` default: the tool folder |
| `preview_0_9_gui_test.py OUT_DIR` | 0.9.0 *Preview crop: in the main views*, visible window, embryo demo with the trunk outlined: *Zoom to region*, black bands end at the region's edges, inside black = the crop's own rule (`outline_mask`), slider timing with and without, a slice outside the region, a projection, preview off (12 checks); saves `preview_*.png`. `VC_TOOL` default: the tool folder |
| `tiff_0_10_gui_test.py OUT_DIR` | 0.10.0 *Also save as TIFF*, visible window, embryo demo with the trunk outlined: the level menu, the TIFF line in the plan text, *Crop selected* writes crop + TIFF (voxels = the crop's level, ImageJ calibration, channels), progress to the end, the report; refused without *Overwrite*; a level the crop lacks gives its coarsest; *Overwrite* deletes the replaced crop's old TIFF and says so; a TIFF over the limit is asked about first (12 checks; dialogs answered by the script); saves `tiff_window.png`. `VC_TOOL` default: the tool folder |
| `time_0_11_gui_test.py OUT_DIR` | 0.11.0 cropping in time, visible window, synthetic time-lapse (9 timepoints, written into OUT_DIR): t fields as timepoint numbers with times, *every n*, the plan text, *here*, fading of 2D lines and the 3D frame at timepoints not kept, the crop preview black there, *What to load* counting from 1, the crop (timepoints 2, 5, 8; 90 min apart; checked) (8 checks); saves `time_window.png`. `VC_TOOL` default: the tool folder |
| `files_0_12_gui_test.py OUT_DIR` | 0.12.0 TIFF files in memory, visible window, synthetic TIFFs written into OUT_DIR (Fiji hyperstack, a 2-image OME-TIFF with a 2D movie, a plain stack): the file's *What to load* dialog (images in the file, an unnamed axis as t, calibration fields, memory line, a 2D image without z/3D), a part read into memory at its place, TIFF level hidden and crops .tif/.ome.tif, a zstd crop checked OK, *What to load…* reading again with the region kept, a 2D movie XY-only (no z row, no 3D) and its crop (every 2nd frame, OME-TIFF), four views again for a 3D image, *Open last crop* on a TIFF (14 checks); saves `files_window.png`, `flat_window.png`. `VC_TOOL` default: the tool folder |
| `plugin_gui_test.py OUT_DIR` | 0.13.0 napari plugin, visible windows, run in the environment with the plugin installed (`napari-plugin-test`): npe2 finds it, napari starts without importing the tool, the sample, *Layers › Data › Crop selected layers…* greyed out without a selection, *Plugins › Volume Cropper* (a small panel; the window's layers and buttons untouched), *Crop the selected layers* → a new window with one 3-channel image, a zstd OME-TIFF crop checked, closing forgets the window, the Layers › Data command through napari's app-model (napari passes the viewer), *File › IO Utilities › Open in a new window*, no private-API warnings, *Use this window…* in a second window (warning answered); saves `plugin_panel.png`, `plugin_tool_window.png` (synthetic data) |
| `plugin_install_test.py TOOL_DIR WORK_DIR [ENV] [--keep]` | 0.13.0 installing the plugin (no windows shown): a copy of TOOL_DIR in "WORK_DIR/vc copy (with spaces)" stands in for the shared folder; `volume_cropper.bat --install-plugin` into ENV (default `napari-plugin-test`): only `napari_volume_cropper` mapped, `PYTHONDONTWRITEBYTECODE` set in the environment, nothing written into the copy; npe2 finds it and napari starts with it; a new module and a changed manifest work without a reinstall; napari starts with the folder gone; `bash volume_cropper.sh --install-plugin` (Git Bash); `--uninstall-plugin`. Afterwards ENV has the plugin from TOOL_DIR again (unless `--keep`) |
| `readers_0_15_check.py` | 0.15.0, no window: the readers on LIF and CZI files of your own (`readers_0_15_check.py FILE …`): scenes, axes, calibration, a part through `load()` equal to the library's own read |
| `readers_0_15_bioformats.groovy` | Fiji's Bio-Formats calibration of the same files, headless (`fiji-windows-x64.exe --headless --run …`; files in `BF_FILES`, it writes the text file `BF_OUT`, because headless Fiji's printed output does not reach the terminal on Windows). On real LIF and CZI files it gave the same calibration as the readers |
| `autocrop_0_14_gui_test.py OUT_DIR` | 0.14.0 *Experimental: find a bright object*, visible window: off at first; on the embryo demo (OME-Zarr) a new region around the brightest embryo (holds ≥ 95 % of embryo 1's true voxels), a rough region around embryo 4 shrunk on every side and still holding it, *Undo fit*; on a synthetic single-embryo time-lapse (Fiji TIFF written into OUT_DIR; missing at the first timepoint, drifting) the region holds it at every timepoint from the second, and its crop is checked OK (6 checks); saves `autocrop_demo.png`, `autocrop_movie.png`. `VC_TOOL` default: the tool folder |
| `panel_0_16_gui_test.py OUT_DIR` | 0.16.0 panel, visible window, embryo demo, settings file of its own in OUT_DIR: the parts open and close (the panel gets shorter), *Memory cache* first in *View*, napari's layer list shows only channels and *crop regions*, the *Polygons* part only for a region with polygons, × marks of XZ and YZ polygons in green and red in XY, the *Settings* dialog; saves `panel_open.png`, `panel_closed.png`, `marks.png`, `settings.png`. `VC_TOOL` default: the tool folder |
| `seeds_0_16_gui_test.py OUT_DIR` | 0.16.0 *Find a bright object*, seeds and outlines, visible window, embryo demo: real mouse clicks on the XY canvas put seeds (cross in all 2D views; a click on a seed removes it); seeds on two embryos that are not the brightest (one twice) with *Outline* → *Fit to seeds* gives two regions holding ≥ 90 % of their embryos, with XY/XZ/YZ outlines keeping ≥ 90 %; the crop of one is checked OK with zeros outside; *Undo fit* removes them; then, for the record, what the window says for a seed on embryo 5 (no threshold separates it); saves `seeds.png`, `seeds_fitted.png`, `seeds_not_separable.png`. `VC_TOOL` default: the tool folder |
| `seeds_threshold_curve.py` | (no window) the size of each demo embryo's seed piece, and its share of own voxels, at 24 thresholds from the automatic one down to the growing floor: why 0.16.0 chooses a seed's threshold the MSER way (embryos 1, 4, 6 stable until they join the tissue; 2, 3, 5 join before they are whole) |
| `seeds_all_embryos.py [TOOL]` | (no window) a seed on each demo embryo, growing off and on: share held, region volume against the true box, other embryos in it, time |
| `levels_zoom_0_22_gui_test.py OUT_DIR` | 0.22.0 Levels *zoom to min–max*, visible window, embryo demo, settings file of its own in OUT_DIR: the tick box fits its row (also in Settings); 25 % room, finer bins, all voxels still counted; real drags beyond the edge (axis still while dragging, follows on release, a second drag goes further); typed min and auto-contrast move the axis; the time to follow on *whole image*; unticking; saves `levels_zoom_*.png`, `settings_dialog.png` |
| `plugin_prototype_gui_test.py OUT_DIR [TIFF]` | the plugin prototype (branch `plugin-prototype`, 2026-11-07; superseded by `plugin_gui_test.py`) |

`gui_v03_test.py`, `gui_timelapse_test.py`, `repro_hide.py` and
`open_dialog_gui_test.py` test the folder in the `VC_TOOL` environment
variable (default: the tool folder they are in); `channels_test.py` and
`guide_screenshots.py` do too. Since 0.5.0 the 3D
view starts off and new windows ask "What to load": the scripts tick 3D and
press Load themselves.

**The magic wand's checks** left this folder on 2026-11-24 (the user's
choice; in the release after 0.21.0): the wand is archived (branch
`archive/magic-wand-prototype`), and these 14 scripts need it, either
`wand.py` or the wand and ruler parts of its window: `wand_gui_test.py`,
`wand_toggle_test.py`, `wand_outlines_gui_test.py`,
`ruler_help_gui_test.py`, `review_fixes_gui_test.py`,
`real_data_gui_test.py`, `wand_stress_necks.py`, `wand_stress_seeds.py`,
`wand_bumpy.py`, `scenario_check.py`, `read_pos.py`, `final_real.py`,
`render_labels.py`, `proj_final.py`. They are in the maintainer's
development history (release 0.21.0).

**Checks on the maintainer's own files** left this folder in 0.23.0:
`readers_0_15_gui_test.py` (LIF and CZI files) and
`stitching_0_17_2_gui_test.py` (a LIF tile scan) checked properties of
those particular files, and `v04_gui_test.py` and `files_0_12_gui_test.py`
lost their parts for a real image. They are in the maintainer's
development history (release 0.22.1).
`make_demo_tails.py` (here since the archive) makes the touching-tails
demo image and gives its tails' geometry.

`guide_screenshots.py`: the user guide's pictures (synthetic demo only:
the embryo demo, its trunk outlined from the true shape; the tails demo for
a wand copy), written into `docs/images` of the copy in `VC_TOOL` (default:
the tool folder it is in): `load_dialog.png`, `main_window.jpg`, `outlines_xy.jpg`
(plus the wand pictures for a wand copy); then rebuild the PDF with
`python docs/build_user_guide.py` there. Once (2026-10-27), on the wand
version, Python crashed with an access violation while exiting after all
pictures were saved; the pictures were fine.

`make_cover_image.py OUT.png`: the GitHub repository's cover picture
(*Settings › Social preview*; 1280 × 640 px, under 1 MB) from the embryo
demo only: the name and the tool's idea ("Draw regions at low
resolution. Save them at full resolution."), three linked views (maximum
projections of level 1, half resolution) with crop regions as solid lines
and silhouette polygons (traced from the demo's true labels) as dashed
lines, and the crop's 3D preview at full resolution (level 0): the voxels
the tool's `outline_mask` keeps, the others dimmed as in the window. Stops
if the words would reach the pictures. No window. Its picture is also
`docs/images/cover.png`, at the top of the README (0.23.1; the PNG holds
no metadata): `python dev/make_cover_image.py docs/images/cover.png`.

`bard_gl_check.sh`: which OpenGL route works on a BARD desktop.

**Embryo demo (2026-10-30):** a more realistic synthetic test image than
the tails: six embryo-inspired objects (trunk and tail with somites and
PSM, blastocyst-like cyst, branching bud in mesenchyme, looped heart-like
tube, limb bud, gastruloid-like ovoid), some pressed together, in loose
tissue with sparse cells, fibres, vessels and debris; 3 channels (nuclei,
membranes, marker), light-sheet look (shadows, stripes, depth, blur,
noise). 0.8 × 1.0 × 1.0 mm at 4 × 2 × 2 µm, 100 MB, made in about 70 s.

| Script | Does |
|---|---|
| `make_demo_embryos.py OUT_DIR [--voxel Z Y X] [--seed N]` | writes `embryo_tissue_demo.zarr` and the true labels `embryo_tissue_demo_truth.npz` (object and tissue type of every voxel); `--layout` only builds the labels and says which objects touch (for placing them) |
| `preview_demo_embryos.py DEMO_DIR OUT_DIR` (needs matplotlib) | `layout.png`, `slices.png`, `outlines.png` (true edges dashed), `detail.png` |
| `score_demo_embryos.py DEMO_DIR [--margin N] [--views "xy xz"] [--steps]` | outlines traced from each object's true shape (3 XY, 2 XZ, 2 YZ slices); the tool's 3D shape: % of the object kept, % extra, other objects' voxels, loose pieces, closed hollows, slices with holes; `--steps` after each polygon |

`make_demo_tails.write_ome_zarr` takes the name, the channels and extra
metadata since 2026-10-30 (the tails demo is written as before).
If Windows refuses to replace the demo folder because a folder in it is
read-only, `make_demo_embryos.py` clears that flag first.
