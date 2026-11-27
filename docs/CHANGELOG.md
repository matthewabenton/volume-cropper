# Changes

The crop code's version is recorded in each crop's provenance
(`crop_provenance` in its `zarr.json`).

## 0.23.1 (2026-11-27, beta)

The cover picture at the top of the README (the user's wish: on the
GitHub repository's front page) and of README.pdf:
`docs/images/cover.png`, made by `dev/make_cover_image.py` from the
synthetic demo only, now without metadata in the PNG (matplotlib wrote
its name and version). The tool works exactly as 0.23.0; the guide is
the same apart from its version line, now `USER_GUIDE_0231.pdf`.

## 0.23.0 (2026-11-27, beta)

Ready to be public: the fixes of a pre-publication review (a crafted file
could make the tool read other files, or make Windows connect to another
computer and send it the user's login), and no folders of the group's
network drive in the public files. The environment's pins are those of
0.17.0. The guide's PDF is now `USER_GUIDE_0230.pdf`.

- **Leica files:** liffile opens the files a Leica file's XML names, at
  any path. A .lif or .lof holds its images itself: one that names other
  files is refused (real ones name none). An .xlef experiment names
  thousands of files, climbing with "..": each name must stay inside the
  experiment's folder. Its files are opened in liffile's own order, each
  after its name was checked, and liffile uses these (`_open_experiment`
  fills its `children` cache); an image's frame files are checked just
  before liffile reads them (a .lof frame must name no files; an OME-TIFF
  frame is opened as one plain TIFF: `_check_frames`). A real experiment
  of a few thousand files opens as before (in about the same time, the
  same images) and reads the same pixels, through both kinds of frames.
- **TIFF:** never read as a Micro-Manager NDTiff (`is_ndtiff=False`): its
  index names files that tifffile would open wherever they are. A folder
  of TIFFs is read file by file (`_multifile=False`), as only its first
  and last file are checked when it is opened.
- **Names shown as text, never as HTML** (new `qt_text.py`): Qt shows text
  that looks like HTML as HTML, and an `<img>` tag in a name made Qt read
  the file it names. The tool's labels and message boxes show plain
  text; its formatted labels escape the names in them; tooltips and the
  labels of Qt's dialogs show names exactly as they are (`literal`);
  napari's layer names get an invisible space after every "<"
  (`defang`). OME-Zarr axis names and units with < > & " or control
  characters are refused when the image is opened (they also reach
  napari's own labels).
- **Regions files:** a region name that starts like a spreadsheet
  formula (= + - @) is saved with a leading ' (Excel shows it as text)
  and read back unchanged; axis names that start so are refused (they
  become column names).
- **Launcher (Windows):** Python and findstr are called by their full
  paths (cmd.exe would run a python.bat in the current folder first).
- **dev/:** the two checks that tested the maintainer's own files are
  removed, two others lost their parts for a real image, and the scripts
  no longer describe the maintainer's files, computer or drives.
- **Tests:** new `test_qt_text.py` (`run_all_tests.py text`), and tests
  of the Leica, NDTiff, TIFF-folder, CSV and axis rules; the file tests
  watch every file opened (an audit hook) and fail if one outside is.
- README, user guide, `env/environment.yml`, the notebook and two
  change-log entries no longer name the network drive's folders: they say
  `<tool folder>`; the README's paragraph on a shared copy is general. The
  folders are in a note beside the shared copy, outside git.
- **Notebook:** it finds the tool only in its own folder or one level up
  (in the tool's `examples` folder). A copy kept elsewhere stops with a
  message asking for `TOOL_DIR` in its first code cell; before, it fell
  back to the shared folder's path. Checked in place, as a copy without
  `TOOL_DIR` (the message) and as a copy with it.
- **User guide:** how to make the two demo images without a shared copy
  (`dev/make_demo_embryos.py`, `dev/make_demo_embryo_timelapse.py --step
  16`).
- New `tests/test_public_files.py` (`run_all_tests.py public`): fails when
  a file names a folder on a network drive letter, a cluster, home-folder
  or network-volume path, a Windows network path or an email address. It
  looks for the form of such paths only, so it names no drive or person
  itself; it fails on 0.22.1 (14 such lines).
- `dev/make_cover_image.py`: the GitHub repository's cover picture, from
  the synthetic demo only (polygon outlines in three views, their 3D
  preview, "Draw regions at low resolution. Save them at full
  resolution.").

## 0.22.1 (2026-11-27, beta)

Ready for publication. The tool works exactly as 0.22.0 (no change to its
code, only to comments); the environment's pins are those of 0.17.0. The
user guide's text is unchanged apart from its version line and dates; its
PDF is now `USER_GUIDE_0221.pdf` (named after the version, which
`build_user_guide.py --check` expects before a release).

- **Licence: BSD 3-Clause** (`LICENSE`; the user's choice, 2026-11-27),
  "Copyright (c) 2026 EMBL, Matthew Benton".
- `pyproject.toml`: `license = "BSD-3-Clause"`, `license-files`,
  maintainer; setuptools 77 or later (the environment pins 84); no longer
  "for use within the lab only".
- README: a heading for the group's shared copy, *Everyone else*
  (download, then the guide's *Your own copy*), and *Maintainer and
  licence*.
- `dev/`: the check scripts no longer contain paths of the maintainer's
  computer. They use the tool folder they are in (`VC_TOOL` still
  overrides) and the demo folder beside it (or `VC_DEMO`); the scripts
  that read the maintainer's own microscope files take them as arguments
  (`channels_test.py`, `readers_0_15_check.py`,
  `stitching_0_17_2_check.py`) or environment variables
  (`readers_0_15_bioformats.groovy`: `BF_FILES`, `BF_OUT`).
- Two comments and a change-log entry no longer name a project's data.

## 0.22.0 (2026-11-25, beta)

The user checked the histogram zoom in the window and asked for the release
the same day. The environment's pins are those of 0.17.0 (unchanged).

- **Levels: "zoom to min–max"** (the user's wish, 2026-11-25): a tick box
  next to *log* spreads the contrast limits across the histogram, with
  room on either side (`levels.ZOOM_MARGIN` = 25 % of their distance, the
  user's choice) to drag them further out, never beyond the data
  (`levels.axis_range`). The 256 bins are computed again over that range
  (`levels.histogram(margin=...)`); the statistics still count all values.
  The axis follows when a drag ends (not during one), and when the limits
  are typed, set by auto-contrast or by napari; the Autocropper's
  threshold stays in view. Its start value is in *Settings…*
  (`view.levels_zoom`, off at first). Tests: `test_axis_range_zoom`,
  `test_histogram_zoomed`, `test_levels_zoom_in_window`, settings.
- **`README.pdf` and `docs/BETA_TESTING.pdf`** (the user's wish,
  2026-11-25: a PDF of the README for colleagues who don't read Markdown;
  plain names, "since we're moving to GitHub style"). `build_user_guide.py`
  makes README.pdf after every guide build (and alone:
  `python docs/build_user_guide.py README.md`), prints links to files as
  their path (a PDF would point into the temporary build folder), refuses
  a PDF that links files on this computer, and notes each Markdown's
  SHA-256 in `docs/pdf_sources.json`; `--check` lists PDFs out of step,
  `--all` makes them all. New `tests/test_docs.py`: README.pdf and
  BETA_TESTING.pdf must match their Markdown; README's links must lead to
  files. The README links BETA_TESTING.pdf.
- `dev/`: the magic wand's 14 check scripts removed (the user's choice,
  2026-11-24, "drop them next time"): they need the archived wand
  (`wand.py`, or the wand and ruler parts of its window), which is not
  part of this release; `dev/README.md` names them.

## 0.21.0 (2026-11-24, beta)

New formats, asked by the user on 2026-11-22 (plan approved the same day:
decisions 1a, 2a, 3a, 4a, 5a), and a tidier tool folder (2026-11-24). The
environment's pins are those of 0.17.0 (unchanged).

- **OME-Zarr 0.6** (final since 2026-09-14; RFC-5). The axes live in named
  coordinate systems; each level has one transformation (a scale,
  identity, or a sequence of a scale and a translation) to the image's
  own ("intrinsic") system. The tool reads them in order, shows and crops
  the image in that system (as the specification recommends), and writes
  0.6 crops the 0.6 way. Links to other coordinate systems (e.g. a
  registration) are kept, corrected when *Keep position* is off, and left
  out when they point at something a crop does not include (label images,
  fields stored in arrays); the record says which (decision 5a). A 0.6
  `scene` is not copied into a crop (a crop holds one image). Pre-release
  names such as `"0.6.dev4"` (ngff_zarr 0.43 writes it for 0.6) are read
  with the 0.6 rules, and crops keep the name as written.
- **Zipped OME-Zarr**: RFC-9 `.ozx` files and other zips (`.zip`,
  `.ome.zarr.zip`, also with the image in a folder inside, or compressed:
  slower, and the tool says so) open like folders, read piece by piece
  (*Open… › Zipped OME-Zarr*, the command line, the notebook). RFC-9 is
  still a proposal (state R4, 2026-11).
- **Crops of zipped images are zipped** (decisions 2a, 3a): `.ozx` for
  OME-Zarr 0.5 and 0.6, in RFC-9's layout (metadata first, breadth-first;
  no zip compression; ZIP64; the version in the zip's comment); a zipped
  0.4 keeps its own ending (`.zip`, `.ome.zarr.zip`). A crop is written
  as a folder in this computer's temporary folder (next to the output if
  that has no room: decision 4a), checked there, its TIFF made from it,
  then packed (a *packing* step in the progress bar), and the zip is
  checked too (its files, and the crop's checks read through it).
  Overwrite replaces only the tool's own zipped crop of the same region.
- **Collections of images**: a bioformats2raw layout (also zipped) or an
  OME-Zarr 0.6 scene with several images asks which one to open (a small
  chooser before *What to load*; `open_image(path, inner=...)`;
  `--collection-image` on the command line; `--image` stays the image
  of a microscope file). One with a single image opens it
  (before: refused). Crops are named `<collection>_<image>_<region>` and
  record the image (`source_image`).
- RFC-8 (collections) is not read yet: it is a draft under review (state
  R1, "not implemented yet"; decision 1a). See DEVELOPERS.md, Planned.

### A tidier tool folder (2026-11-24)

Plan approved by the user on 2026-11-24: a base folder with only what a
colleague needs, the Python code in one package, one clone of the
repository (switching branches) instead of worktrees.

- **Base folder:** `README.md` (where to start), the launchers
  (`volume_cropper.bat`, `volume_cropper.sh`, `install_napari_plugin.bat`),
  `pyproject.toml`, and the folders `env/` (the environment recipe),
  `volume_cropper/` (the program), `napari_volume_cropper/` (the plugin),
  `docs/` (user guide, patch notes, this file, the not-yet-tested list,
  beta testing; the guide's builder and pictures), `examples/` (the
  notebook), `tests/` (`python tests/run_all_tests.py`) and `dev/`
  (`DEVELOPERS.md` and the check scripts, kept outside git until then;
  their demo movies stay out of git).
- **One package:** the modules are the package `volume_cropper/`, and
  `volume_cropper.py` is now `volume_cropper/core.py`. With the tool folder
  on `sys.path`: `from volume_cropper import core as vc` (was `import
  volume_cropper as vc`) and `from volume_cropper import file_images` (was
  `import file_images`). No compatibility layer (the user's decision):
  copies of the notebook made before stop in their first code cell, so
  colleagues copy `examples/volume_cropper.ipynb` again (patch notes).
- **Starting it:** the launchers keep their names and their behaviour.
  They run `volume_cropper/__main__.py` as a file, without changing the
  working folder, so relative image paths and the *Open* dialog stay the
  user's; run as a file, it puts the tool folder on `sys.path` itself.
  `python -m volume_cropper` (from the tool folder) starts the window too.
- **Command line:** `python <tool folder>/volume_cropper/core.py
  <image>.zarr boxes.csv <output folder>` from any folder (was
  `<tool folder>/volume_cropper.py`); run as a file it finds the package
  itself and runs the package's copy of its code (one copy of its
  classes). `python -m volume_cropper.core` works from the tool folder.
- **Plugin:** finds the package in the tool folder and reads the version
  from `volume_cropper/core.py`. The plugin package itself did not move,
  so an install made before keeps working without reinstalling. Checked at
  the release: the plugin installed from 0.20.0 in a scratch copy, which
  was then updated with `git pull --ff-only` as the shared folder is; npe2
  still finds it, the tool then comes from `volume_cropper/`, nothing was
  written into the copy, and `plugin_gui_test.py` passed 13/13 on it. pip
  and napari's plugin list keep the version of the install (0.20.0); the
  panel shows the tool's.
- New tests: the command line from another folder with relative paths
  (`test_command_line_runs_from_any_folder`) and the launchers' way of
  starting the window (`test_launchers_start_the_window_from_the_users_folder`).
- **Notebook** (at the release): it looks for the package in its own folder
  and one level up before it uses the shared folder's path, so the
  notebook in `examples/` uses its own tool folder again, as it did in the
  tool folder before the move (in place it had fallen back to the shared
  folder's path: on the hub the shared copy, whatever its version; on
  Windows an error). It prints where the tool came from. Its
  introduction names 0.6 and zipped images, and its notes the speeds of
  2026-11-22 (hub ~120 MB/s with 8 workers, cluster job ~300 MB/s with
  16). Run end to end on the embryo demo, in place and as a copy elsewhere
  (a crop and its TIFF, both checked OK).

After the release, the same day: the guide was rebuilt under the same name
(`USER_GUIDE_0210.pdf`), with its authorship line in the wording approved
before (the dates to 24 November).

## 0.20.0 (2026-11-17, beta)

Developed as 0.18.0.dev0 and released as 0.20.0, the user's choice, to mark
a major step: there is no 0.18 or 0.19 release. Notes "(0.18.0)" in the
code and in DEVELOPERS.md mean this release. The environment's pins are
those of 0.17.0 (only the recipe's comments changed).

- **Levels (histogram)**, a new part of the panel under *View* (asked
  2026-11-12: napari 0.9.2's own histogram shows no numbers). For one
  channel it shows the histogram with its values and counts (log
  optional), the contrast limits as white lines with their values, and
  napari's brightness mapping (yellow, with gamma). Pointing at it gives
  the values in a bin and how many voxels it holds; dragging a white line
  sets that limit in all views. Below: minimum, maximum, mean, the share
  of voxels at the maximum (saturation) and below / above the limits.
  Its source is the user's choice: the XY view as shown (it follows the
  slider, zoom and projection), the whole image (as loaded), or the
  selected crop region (the voxels its crop keeps, inside its outlines),
  at the timepoint shown. A channel selected in napari's layer list is
  shown. Code: `levels.py` (no window), `levels_panel.py`.
- **Gamma and number boxes in Levels** (asked 2026-11-12, during the
  user's test). A dashed line marks the value shown at half brightness;
  dragging it sets gamma (left brightens the dim values, right darkens
  them), and double-clicking it sets gamma 1, like the grey slider of
  Photoshop's Levels. Boxes under the histogram take exact values for
  min, max (the contrast limits) and gamma. They apply when typing ends
  (Enter), in all views, and follow napari's own controls. Gamma stays
  within 0.2-2, the range of napari's slider, which would put any other
  value back into it. A min typed at or above the max stops just below
  it. The statistics line now starts "data:" to tell the data's own
  minimum and maximum apart from the boxes. Gamma changes only the
  display; crops keep the original values.
- **Several regions or polygons deleted at once** (the user's wish,
  2026-11-12). In the *Crop regions* and *Polygons* lists, Ctrl+click,
  Shift+click and Ctrl+A select several; *Delete* (regions) and *Delete
  selected* (polygons, was *Delete polygon*) delete them all, as does the
  Delete key in the list. With several, the panel asks first (*Undo*
  brings them back). The region clicked last is the selected one (in bold): drawing,
  the preview, shading and Levels use it. The region list is now updated
  in place, so a refresh keeps the selection.
- **The crop preview's projection is the crop's own** (the user's wish,
  2026-11-12; they chose the exact version). With *Preview crop* on and a
  max or mean projection, each 2D view shows the projection of exactly
  what the selected region's crop keeps: only its depth, with the voxels
  its polygons set to 0 at 0 (a mean projection counts them, as Fiji's
  would on the saved crop), black elsewhere. It is read in the background
  at the timepoint shown, at the finest level with at most 16 million
  voxels per channel, and follows the region, its polygons, the channels
  and their colours; a line under *Preview crop* names the level. Before,
  the views showed napari's projection through the whole depth (or the
  range chosen), with only the region's rectangle. Code:
  `crop_projection.py`; `levels.read_block` (shared with Levels).
- **A "Loading…" box while the views read** (the user's wish,
  2026-11-12): napari reads in the window's own thread, so the window
  froze without a word. A box in the middle of the window, and the busy
  pointer, now say what is loading when an image opens (after *What to
  load*), for a new 2D resolution or *exact level*, the 3D view and a
  projection; they go once the window has been free for 0.6 s (2 s if
  something keeps reading in the background). The line under *Image*
  counts what is read.
- **Changing a crop region in the 3D view** (the user's wish,
  2026-11-12; they chose face dots plus a centre dot). The selected
  region's wireframe has a dot in the middle of each face and a white one
  in its centre. Dragging a face's dot moves that face along its axis, to
  where the mouse is (the other faces keep their values exactly);
  dragging the centre's moves the region and its polygons across the
  screen (in the plane facing the camera), keeping its size. The camera
  stays still while a dot is dragged; elsewhere the mouse turns the view
  as before. The 2D views, number fields and plan follow (a face as it
  goes, the centre on release). Polygons still shape a region: a face
  can't go past them in their own plane (it springs back to them).
  Looking straight along an axis, its faces can't follow the mouse (turn
  the view a little). Code: `box_handles.py` (the geometry, no window),
  `OrthoViews._handle_drag`.
- **New region with the 3D view on** (the user's wish, 2026-11-12):
  the region starts as the middle half of the image along each axis, so
  its dots lie inside the image, ready to drag. Drawing into it first
  makes it the whole image again, as before (one polygon then still
  reaches through the whole depth); a region shaped by its dots stays
  after *Finish region*.
- **More for the 3D region** (the user's choices, 2026-11-12):
  - *Shift*-drag a face's dot: the region resizes around its centre (the
    opposite face moves as far the other way). *Ctrl*-drag: the face (or
    the moved region) lands on round values, in steps of about 1 % of the
    image (1, 2, 5, 10, 20 ... µm; 10 µm for the 1 mm demo).
  - *Up* / *Down* in the 3D view move the face last dragged (or clicked)
    one voxel out / in; with *Shift* ten voxels.
  - With *Clip 3D to selected region*, the clipping follows a drag live.
  - A click on another region's frame in 3D selects it.
  - *Zoom to region* also centres the 3D view on the region and fits it
    (its angle stays).
- **Undo and redo for all changes to the crop regions** (the user's
  wish, 2026-11-12): *Undo* / *Redo* under the region list, *Ctrl+Z* /
  *Ctrl+Shift+Z* (or *Ctrl+Y*) anywhere in the window (a number field
  being typed in keeps its own). Drawing, dragging in 2D and 3D, typed
  ranges, margin, deleting (also several at once), renaming, loading
  regions, the Autocropper: every change ends in the panel's refresh,
  which compares the regions with the last snapshot. A drag in 3D is one
  step, and so are changes less than 0.5 s apart (keys or arrows held).
  50 steps; another image starts a new history. Undo gives the same
  region objects their old values back, so the Autocropper's seeds keep
  their regions.
- Found by the visible check, fixed: the dots' huge point size (to keep
  them 14 pixels on screen) made napari's home button zoom the 3D view
  out to nothing, because napari counts point sizes in the scene's
  extent; and Up / Down bound with `functools.partial` made every key
  press in the 3D view fail (napari reads a bound function's name).
- **The panel's own controls instead of napari's left panels** (the
  user's wish, 2026-11-13; napari's layer list and controls took much
  room and offered little more). napari's panels are hidden at first now
  (the View tick *napari's layer panels (left)* and Settings bring them
  back; the setting is new, `napari_layer_panels`, so a settings file
  that kept them shown under the old key starts with them hidden too).
  - **Channels**, a new part above Levels: a row per channel with a
    tick box to show it (replacing *Show channels* under View), its name
    and a colour menu (the file's colour or one of napari's). A click on
    a row chooses the channel for Levels (its name shown in bold);
    napari's layer list still does too.
  - **Levels: auto-contrast** *once* and *continuous* (as napari's; a
    limit set by hand stops *continuous*) and *reset* (the contrast the
    channel had when the image opened, and gamma 1; napari's own reset
    differs little from *once*). Levels shows the channel's name instead
    of its channel menu.
  - **Crop regions: the drawing tools** as buttons: Move view, Rectangle,
    Polygon, Lasso, Select, Edit points (napari's modes, with napari's
    keys in the tooltips), the one in use pressed, also when chosen by its
    key; and *Delete shape* (the shapes selected in a view: a rectangle
    deletes its region, a polygon only itself).
- **Opening zooms in to the resolution chosen** (the user's wish,
  2026-11-13): with a 2D resolution chosen in *What to load* (not
  *Auto*), the 2D views open zoomed in to one voxel of that level per
  screen pixel, centred on the image, so that napari shows that level (it
  showed a coarser one for the whole image). If the image fits at that
  level anyway, or with *Auto*, they show the whole image as before. Only
  when an image opens; changing the resolution later keeps the zoom.
- **Seeds in the 3D view** (the user's wish, 2026-11-13): as in 2D, a
  cross in its region's colour with a dark edge.
- **Fit to seeds / Fit all regions: one progress window** (the user's
  wish, 2026-11-13). Its bar now moves within each search (per threshold
  tried with a seed: on the embryo demo a search took 6-9 s and showed
  "timepoint 1 of 1" all the while), names the seed or region, and ends
  with making the regions and their outlines. Cancel stops between those
  steps too. Measured: the searches are 97 % of the time, the outlines
  2 %.
- **Autocropper: one's own threshold, and what a seed finds** (the
  user's wish, 2026-11-13: separate brighter parts of an embryo from
  fainter ones). *Own threshold* (off at first): a slider, a number box
  and *Auto*; it starts at the automatic value (the method chosen, on
  the loaded part at the timepoint shown). While it is on, *Fit*, *Fit
  all regions* and *Fit to seeds* all use that one value at every
  timepoint, without growing (*Dim parts too* is switched off and greyed
  meanwhile, and back as it was afterwards). The 2D views outline what
  lies above it on the slice shown (through the slab with a projection)
  as a dotted orange line, from the volume as the search sees it (its
  level, smoothed); it follows the slider, the slice, the channel and
  the timepoint (another timepoint is read once one stays on it 0.3 s;
  on the embryo demo a read takes 0.2 s, an outline 1-8 ms per view).
  The *Levels* histogram shows the value as a dotted orange line when it
  shows the Autocropper's channel; dragging that line sets the value
  (it wins over a line within 2 pixels of it, e.g. gamma's). The status
  after a fit names the value. New menu *A seed finds the*: *most stable
  part* (as before), *brightest stable part* (e.g. a bright core) or
  *whole object* (the dimmest stable threshold before the piece joins
  too much of the image). On the embryo demo with one seed (nuclei,
  *Dim parts too* off): brightest 152 × 724 × 384 µm (threshold 511),
  most stable 212 × 880 × 408 µm (287), whole 544 × 1000 × 968 µm (172,
  which there took in the tissue: in tissue, use *most stable* or one's
  own threshold). Code: `autocrop.search_volume` / `SearchVolume`,
  `find_object(threshold=, seed_mode=)`, `_seed_threshold(mode=)`;
  `OrthoViews.set_level_outline`; `LevelsPanel.set_marker`.
- **Autocropper: touching objects split between their seeds, and gap
  lines** (the user's wish, 2026-11-13: "as in ilastik, where one paints
  the gap between objects"). *Split an object between its seeds* (on at
  first): when *Fit to seeds* finds that several seeds share one object
  (touching embryos no threshold separates), it splits that object
  between them by a seeded watershed in 3D (as MorphoLibJ's
  *Marker-controlled Watershed* in Fiji, or ilastik's carving): each
  seed's part floods outwards from it until it meets another's, where
  the image is dimmest between them and where the object narrows; so the
  boundary can lie at any angle. *Paint gap*: hold the mouse button and
  draw along the gap in a 2D view (white dash-dot lines on a dark edge;
  dragging no longer pans meanwhile, scrolling still zooms); the lines
  are walls the floods cannot cross on the slice drawn, and raise the
  landscape over 3 slices of the search level to each side; between the
  lines the image decides. A click on a line removes it; *Clear gap
  lines*; width (µm, at first two voxels of the search level). Each part
  then gets its own region, with silhouettes as before. Put one seed per
  object: two seeds on one object now split it (*Split* off: one region,
  as before). Time-lapses: the split starts at the first seed's
  timepoint and each timepoint starts from its neighbour's parts.
  Measured: two touching spheres, contact tilted 30°: each part holds
  99.4-99.5 % of its own and 0.5-0.6 % of the other; two touching
  ellipsoids in the window (visible check 7/7): 97.5 / 100 % of their own,
  0 / 2.5 % of the other (the overlap counted as the first one's), 2.1 s.
  Code: `autocrop.split_objects`, `gap_walls`; `find_object`'s box and
  finishing moved into `_box_of` / `_finish`; `OrthoViews` gap lines
  (`set_gap_lines`, `set_gap_mode`, `_gap_draw`, `pattern_segments`).
- **Autocropper: outlines per slice; parts of a split that do not
  overlap** (the user's wishes, 2026-11-13: generated outlines overlapped).
  *Outline* has a menu: *silhouettes (XY, XZ, YZ)* (as before) or *per
  slice (XY)*: the object's outline on up to 40 XY slices (interpolated
  between them), each its outer edge only (no holes; pieces joined by
  thin bridges), grown by the margin, the end ones repeated a margin
  further out. The crop then follows the object slice by slice, where
  three silhouettes keep a block carved from three sides. The parts of
  an object split between seeds always get outlines per slice, and their
  margin stops halfway to the other part, so that their polygons meet
  instead of overlapping (silhouettes are shadows through the whole
  depth: for parts touching at an angle they overlap, and the margins
  did too). Measured, touching spheres tilted 30°, margin 2 µm: each
  crop held 100 % of its own and 9 % of the other with silhouettes, 99.4-
  99.5 % and 0.5-0.6 % per slice; in the window (ellipsoids, margin 4 µm)
  99.5-99.8 % and 0.2-0.5 %, 177 voxels in both crops. Moving regions keep
  silhouettes at each timepoint. Code: `autocrop.slice_outlines`,
  `_outline_points` (shared with the silhouettes), `_bridged`.
- **Outlines in 3D** (the user's wish, 2026-11-13: "a view of the polygons
  in 3D, maybe as shaded objects"): under *View*, the shape a region's
  polygons make (what its crop keeps, as the 3D dimming computes it at
  the 3D view's level) as a see-through surface in the region's colour:
  *all regions* (at first; was the selected region), *selected region*
  or *off*, with its
  opacity (0.35 at first). It follows the polygons when a change is
  finished, beside *Blacked-out voxels (3D)*. Measured: the surface
  encloses 17,142 µm³ where the crop keeps 17,228 voxels of 1 µm³.
  Code: `OrthoViews.outline_mesh` (marching cubes on the lightly smoothed
  keep mask), `_sync_surfaces`, `set_outlines_3d`.
- **Settings: what new windows start with, for the Autocropper and View**
  (the user's wish, 2026-11-13; new windows only, as the other settings):
  threshold method, Dim parts too, margin, Moving region, Outline and its
  kind, what a seed finds, Split an object between its seeds; slice
  planes, blacked-out voxels (3D), outlines in 3D and their opacity,
  crosshairs, crop regions, preview crop, the Levels histogram's source
  and log. In the Settings dialog's new right-hand column; kept per user
  (settings.json keys `autocropper` and `view`; values of the wrong type
  are ignored). And **what opening an image starts with** (the user's
  wish, key `opening`, every opening from the window): the 2D resolution
  (*Auto*, or the finest level with voxels of at least a size in µm: the
  user's choice, so that images from other set-ups open with the same
  detail; `ortho_views.level_for_voxel`), the 3D view (*Off*, *Auto* or
  by voxel size), the channels shown (all or the first) and *Ask 'What to
  load'* (off: the image opens at once with these choices; a 3D volume
  over 1.5 GB still asks; *What to load...* and napari layers always
  ask). Files (TIFF, LIF ...) take the channels, the 3D view and *Ask*.
  *Outlines in 3D* shows all regions at first (the user's wish: all at
  once).
- **Splitting: overlapping objects too, and "Recompute split"** (the
  user's wishes, 2026-11-13, after a gap line seemed not to work): seeds
  whose objects merely overlap (by 10 voxels and 0.5 % of the smaller),
  without either holding the other's seed, are now split between them too
  (before, they gave two overlapping regions and their gap lines were not
  used). *Recompute split* (beside *Paint gap*) splits again, with the gap
  lines as they are now, the objects of the last *Fit to seeds* that a gap
  line crosses, reusing that fit's searches (quick), and updates their
  regions in place (*Undo fit* gives them back); after the seeds change it
  asks for *Fit to seeds*. The panel's status says when lines were used.
- **The threshold outline has no holes** (the user's report, 2026-11-13: a
  ring inside an object): the dotted outline of one's own threshold traces
  only outer edges (dim patches inside an object are filled first, as
  Fiji's *Fill Holes*).
- **Save as TIFF's scratch folder no longer left behind** (the user's
  report, 2026-11-14: an empty `.volume_cropper_tmp_…` folder on the
  Desktop from a save on 12 November). The removal is tried again a few
  times (Windows may hold a folder open for a moment, e.g. a virus
  scanner); if it still fails, the panel says so. Scratch folders left by
  earlier saves that hold no files and are over an hour old are cleared
  at the next *Save as TIFF…* into that folder. Code:
  `vc.remove_scratch`, `vc.clear_old_scratch`, `CropResult.note`.
- **Settings keep only what you changed** (found 2026-11-13): the file
  stores the values that differ from the built-in defaults, so that later
  versions' new defaults reach you where you have not chosen otherwise
  (saving everything had frozen the defaults of the day).
- **User guide for 0.18.0** (`USER_GUIDE_0180.pdf`): opening (starting
  choices, the Loading box, zoom to the level), the window (Channels,
  Levels, napari's panels, Settings), seeing the data (Levels, gamma,
  auto-contrast, colours, the exact preview projection, outlines in 3D),
  crop regions (tool buttons, several at once, undo / redo, changing a
  region in 3D), the Autocropper (own threshold, seed modes, splitting,
  gap lines, Recompute split, outlines per slice), keys, troubleshooting
  and words; new pictures (Settings, Channels and Levels, a region in 3D,
  a split) from `guide_screenshots.py`.
- **Fixed: opening a second image at a chosen coarse resolution could
  fail** with "IndexError: list index out of range" (the user's report,
  2026-11-13), which also stopped the opening before the views were
  fitted. The new choice (e.g. level 3) was set before the first image's
  layers were replaced; one of them still showing a finer level was then
  looked up in the new list. Each layer now keeps the levels it was built
  with (`vc_levels`; `test_level_shown_while_another_image_opens_at_a_coarse_level`
  fails on the old code with the user's error).
- The region dots and seeds in 3D keep their size on screen by a size in
  µm that follows the 3D zoom: napari's canvas size limit, used at first,
  drew them mostly in their border colour (the seeds came out as black
  discs).
- **Dragging in 3D is 4-10 times smoother** (the user found it laggy,
  2026-11-12). Measured on the synthetic demo, a mouse move took 0.5 s
  (plain region) to 1.2 s (with a polygon), now about 0.12 s. Causes:
  napari redraws a layer whenever its visibility is set, even to the
  same value (~15 ms each; now only when it changes, which also helps
  every other update); the crosshairs and the 3D slice planes were
  redrawn on every update (now only when they move); and each update
  during a drag rebuilt everything for the region (now the lines and the
  3D frame follow live, the 2D shapes, 3D dimming and clipping, shading
  and preview once on the release, 0.3-0.6 s).
- **Fewer × marks where they crowd, and a switch for them** (the user's
  choice, 2026-11-17). The Autocropper's outlines per slice (up to 40 XY
  polygons) put two × marks per polygon into the XZ and YZ views, a solid
  band along the object's edge that hid the dashed outline. Now each
  view's marks are thinned so that none are closer than 30 screen pixels
  (about three marks' widths; `ortho_views.MARK_SPACING`, `thin_points`);
  a view's marks never hide another view's, and zooming in brings the
  others back. On the synthetic touching objects: 84 marks in XZ became
  14, 72 in YZ became 11, and 28 at 3x zoom. *View: Show "× marks"*
  switches them off, also as a start option in *Settings* (key
  `crossing_marks`).

## 0.17.2 (2026-11-11)

The user's choices after the review of 0.17.0:

- **An area a lasso goes around twice is kept** (`fill_polygon`): Fiji's
  even-odd rule left it out (a lasso that overshoots its start had a hole
  where it overlapped itself). Any other polygon is filled exactly as
  before (as Fiji fills it). Used for polygons through a region,
  silhouettes and the shape through polygons on slices.
- **Polygons with (almost) no area are refused** in the window (less
  than one voxel wide on average, `polygon_width`; e.g. three clicks on a
  line): the shape would narrow to nothing there. An edit that flattens a
  polygon is ignored.
- **The Autocropper follows a seed's object past timepoints where it is
  not found**, from the last place it was found (it stopped at the first
  such timepoint, and the region ended there).
- Uncalibrated images stay cropped as 1 µm voxels (the user: keep).
- **Colour (RGB) images from LIF, CZI, ND2 and OIR files open as three
  channels**, named and coloured red, green and blue (as TIFFs already
  did). Their colour axis was a choice, offered as z at first. An image
  with channels as well as colours keeps the colour axis as a choice.
  Checked on a real RGB LIF (a part read equal to liffile's, which
  turns Leica's blue-green-red order around).
- **The guide PDF is named after the version** (`USER_GUIDE_0172.pdf`;
  the user's choice, 2026-11-11). A release used to overwrite
  `USER_GUIDE.pdf`, and when someone had it open in the shared copy, the
  releases of 0.6.0 and 0.17.0 stopped halfway. `docs/build_user_guide.py`
  names the PDF and points
  README.md at it.
- **LIF tile scans open stitched** (the user's choice, 2026-11-10;
  `tile_stitching.py`): "… (stitched)" places each tile where its overlaps
  match (normalised cross-correlation over the shared voxels, then all
  tiles at once by least squares, much as Fiji's Grid/Collection
  stitching), by whole voxels and without blending: each voxel in an
  overlap comes from the tile whose centre is nearest. It is worked out
  when the image is chosen, and the dialog says how it went. "… (tiles)"
  keeps the tiles one after the other. The stage positions alone were
  several pixels off on the tile scans tried (the camera slightly turned
  against the stage). Checked on real tile scans: within a few pixels of
  LAS X's placement for tiles of 512 and 1024 px; overview scans with
  tiny tiles are uncertain. The crop's record keeps every tile's place.
- **LAS X's merged images get a note** in *What to load* (the user's
  choice): LAS X blends the tiles where they overlap, so their voxel values
  are not all the measured ones.

## 0.17.1 (2026-11-10)

Fixes from a review of the whole tool.

- **Polygons (the user's "weird interpolations"):** polygons drawn in one
  view only are now interpolated from one drawn slice to the next
  (shape-based interpolation: the signed distance to each filled polygon,
  blended linearly between neighbouring drawn slices, after lining up the
  polygons' centres). Before, one smooth surface through all polygons
  (since 0.7.0) bulged between equal circles (1.45 times their area), and a
  slightly smaller polygon drawn 1 µm from another emptied the shape 50 µm
  away (a slightly larger one inflated it 1.5-2.2 times). Polygons in
  several views keep the smooth surface, but its points now come from the
  filled polygon: a lasso that overshoots its start by 5 % of a turn kept
  only 43 % of itself on its own slice, a figure-8 half, and a shaky lasso
  (0.7 µm tremor) broke the shape into 5 pieces with 68 % extra (now 90 %
  of the object, 2.7 % extra, one piece). At most 4000 points are fitted
  (a very shaky 1500-point lasso never finished).
- **Security:** an OME-Zarr whose level is named with an absolute path or
  a drive (`C:/...`, `C:0`) was read from outside the image, and its crop
  written outside the output folder: only plain level names are opened
  now. The regions CSV's outlines-per-timepoint file must be a plain file
  name next to the CSV (any path, also a network share, was read). An
  OME-TIFF or companion file that names files outside its folder (`..`,
  absolute, network share) is refused before any of them is touched, and
  a set with a missing file is refused (its planes were zeros, with only a
  log message).
- **Overwrite** replaces only an earlier crop of the same region of the
  same image (pos1/img.zarr and pos2/img.zarr give the same crop name: the
  second replaced the first's crop and TIFFs without a word); a folder
  with an empty or odd crop record is not taken as a crop; a file where
  the crop would go gets a clear message; a `<crop>_positions.csv` this
  tool did not write is never replaced or deleted (it was deleted even
  without Overwrite).
- **Window:** *What to load…* with a region selected crashed (images from
  napari layers: the plugin), and the regions vanished from the views.
  *Use this window…* in a napari window that showed 3D crashed. A moving
  region dragged where it sticks out of the image was clipped there, and
  the smaller size went to all its timepoints. With channel 0 hidden, the
  views' titles and *Auto* showed an old level. The Autocropper in a
  region over one timepoint, with *Moving region* and *Outline* ticked,
  made no outline, and the region (one timepoint) was "outside the image"
  at the next fit: over one timepoint the fit is now a fixed region.
- **Files:** OIR channel names were one channel off (oirfile lists an
  extra "CH1"); a LIF's frame interval from its time stamps (when the time
  axis has no step) was 1000 times too long. The command line now writes
  a moving region's `<crop>_positions.csv`, as the window does.
- **A second review of these fixes** (two more reviewers) found, and
  0.17.1 fixes: an OME-TIFF entry with the file's own UUID followed by a
  space slipped past the file check (tifffile compares exactly; the check
  now follows tifffile's own rule, and checks every name); Overwrite still
  replaced another image's TIFF (Save as TIFF, TIFFs next to a crop),
  a crop made from a relative path, or a crop of another image in the
  same file with the same name (the record now holds the absolute path,
  and which image of a file); lining up the centres pinched a part of a
  polygon that stays put while another part ends (now only where the
  slices hardly overlap); a polygon thinner than the shape's grid among
  polygons in other views got no points (with only thin ones, the fit
  crashed); a LIF without time stamps could not be opened.

## Guide update (after 0.17.0, on experimental)

- The guide's Autocropper picture now shows a multicolour synthetic embryo
  time-lapse (an embryo's trunk and tail that drifts and turns, with a
  gastruloid beside it; made by the maintainer's
  `make_demo_embryo_timelapse.py`), the user's wish of 2026-11-09. No
  change to the code (still 0.17.0).

## 0.17.0 (2026-11-09, beta)

Released together with 0.9.0-0.16.0 below, which were not released on
their own (they were built and tested on the experimental branch). Read
`PATCH_NOTES.pdf` for a summary for users, and `NOT_YET_TESTED.pdf` for
what could not be tested yet. *Find a bright object* is now called the
**Autocropper** (still experimental). The environment needs the new
readers of 0.15.0 (see there).

From the user's test of *Find a bright object* on time-lapses
(2026-11-08): the region was the union of the object's boxes over all
timepoints, so a drifting or growing embryo got a large region. The user
chose "one moving region, one movie":

- **Regions that move over time:** a region can follow an object: one
  size, its place at each timepoint (`CropBox.track`: the shift per time,
  interpolated between them). It is cropped into ONE time-lapse (OME-Zarr
  or TIFF) in which the object stays in place: each timepoint is read at
  the region's place then, on every pyramid level. Its shifts are rounded
  to whole voxels of the coarsest level kept, so that every level stays
  exact; the region is widened by half such a voxel on each side, so that
  it still holds what was asked at every timepoint. Where it reaches
  outside the image, the crop has 0. Every n-th timepoint and *Save as
  TIFF* work with it. Polygons that stay in place are refused on a moving
  region (with a message).
- **Outlines per timepoint** (the user's choice of 2026-11-09, for
  auto-cropping): a moving region can have its own outlines at each
  timepoint (`CropBox.timed_outlines`, at the object's positions then).
  Each timepoint of the crop is shaped by its own (voxels outside are 0);
  a timepoint without one (where the object was missed) keeps the whole
  region (the user's choice: a borrowed shape could cut the object). With
  *Moving region* and *Outline (silhouettes)* both ticked, *Find a bright
  object* makes the object's silhouettes (XY, XZ, YZ) at each timepoint
  from that timepoint's mask. The window shows the outlines of the
  timepoint shown (dashed lines, × marks, shading, preview, 3D, the
  polygon list); reshaping or deleting a silhouette changes that
  timepoint only; drawing new polygons on a moving region stays off.
  A warning in the panel and in the plan says that this is slower on
  large data (the user's request). The regions file keeps them in a JSON
  file next to it (`<regions>_outlines_per_timepoint.json`, named in a
  `timed_outlines_file` column: they are too long for CSV cells, which
  Excel cuts at 32767 characters); the crop's record keeps them too.
  *Fix in place* asks first, as it removes them.
- **Seeds** (the user's requests of 2026-11-09): white with a dark edge
  until *Fit to seeds*, then in the colour of the region they gave (or of
  their object's region), white again if it is removed; a seed whose region
  moves follows it through time (faded where the object was not followed;
  a seed without one is full only at its own timepoint). A **seed table**
  lists each seed's timepoint, position and region ("no object", "same
  object"); a click on a row goes to the seed; *Delete selected seeds*.
- **Fix: the progress bar stopped at 99 % when saving a compressed TIFF**
  (the user's report, 2026-11-09). With compression, tifffile takes
  exactly the planes it needs and never asks for one more, which is when
  the last plane's progress was reported; the end is now reported once
  the TIFF is written (and its check then shows as "checking the TIFF").
- **Where each timepoint came from:** the crop's OME position is the
  region's place at its first timepoint (OME-Zarr has one position per
  image); its record (`moving_region`) and a file
  `<crop>_positions.csv` next to it hold every timepoint's (time and the
  position of the crop's first voxel in the source, in µm), e.g. for
  Fiji or to put measurements back into the source's coordinates.
- **The check** works out each timepoint's place on its own (from the
  region, not from the writer's table of shifts) and compares the voxels
  there; a crop written one step off at one timepoint fails it.
- **Regions file:** a `track` column (only when a region moves), e.g.
  `t=0: z=0 y=0 x=0 | t=5: z=1.5 y=-2 x=12.5` (µm); old files unchanged.
- **Find a bright object:** *Moving region (follows the object)*, on at
  first for time-lapses: a fit makes a region of the object's largest
  extent plus the margin, centred on the object at each timepoint (where
  it was missed: between its neighbours), moved back inside the image or
  the region searched where needed (not cut). With *Outline
  (silhouettes)* too: outlines per timepoint (below); a moving fit
  replaces a region's own polygons (*Undo fit* gives them back, and undoes
  the track and the outlines).
- **In the window:** a moving region is drawn where it is at the
  timepoint shown (lines, the rectangle you edit, 3D frame, 3D clipping,
  preview, shading), and the number fields and the plan show it there.
  Dragging it moves it at that timepoint only; resizing (rectangle or
  number fields) changes its size at all timepoints, each keeping its
  centre. The region list says "· moves", and *Fix in place* turns it into
  one fixed region that holds all its places.

## 0.16.0 (part of 0.17.0)

From the user's test round of 2026-11-08:

- **OME-TIFF companion files:** an OME-TIFF set whose metadata are in a
  companion file (`<name>.companion.ome`, as Viventis and Bio-Formats write
  them for data spread over several TIFFs) opens from the companion, or
  from one of its TIFFs when that names the companion ("BinaryOnly"): one
  image across all the TIFF files it lists, with its calibration and
  channel names. A missing TIFF is named in the message. Crops are named
  after the set (`<name>_<region>.ome.tif`).
- **× marks in their view's colour:** where a polygon of another view
  crosses the slice shown, its × is drawn in that view's colour (blue XY,
  green XZ, red YZ, as the view titles and crosshairs), no longer in the
  region's colour, so that you can see which view each comes from.
- **Find a bright object:** *Fit region to object* no longer makes a
  selected region larger with every click (the fit kept adding the margin
  and padding): a fit stays within the region it searched, and fitting
  again leaves it as it is. When the object reaches the edge of the region
  searched, the result says so (it may be cut there). **Fit all regions**
  fits every region to the object inside it, each on its own (draw a rough
  region around each object first); one *Undo fit* takes them all back.
- **Find a bright object, seeds:** *Add seeds*, then a click on each
  object in XY, XZ or YZ puts a seed (a yellow cross) there, at the slice
  and timepoint shown (a click on a seed removes it; dragging still pans).
  *Fit to seeds* makes one region per seed (`object01`, `object02` …) in
  the loaded part of the image: the object under the seed (or right beside
  it) at the seed's timepoint, followed through time from there (forwards
  and backwards), so a dimmer object or one next to a brighter one gets
  its own region. Each seed's object gets a threshold of its own: of 12
  levels between the image's automatic threshold (or the brightness at the
  seed, if lower) and the background, the one at which the object is most
  stable (it changes least in size; the idea of MSER), so that a dim
  object is found whole rather than as specks, and a bright one stops
  before it joins the tissue around it. Seeds on the same object give one
  region. Objects that no threshold separates (they touch, or join through
  other bright structures) are said so, with the advice to draw a rough
  region around each and use *Fit all regions*. *Undo fit* removes the new
  regions again (also the one *Fit region to object* adds when no region
  is selected). Measured on the embryo demo (growing off): seeds on
  embryos 1, 4 and 6 give regions that hold all of them; embryos 2, 3 and
  5 join the tissue before they are whole at every threshold (cut, or
  with their neighbours, and said so).
- **Find a bright object, outlines:** with *Outline (silhouettes)* ticked,
  a fitted region also gets the object's outline as seen in XY, XZ and YZ
  (over all its timepoints, plus the margin), as silhouette polygons:
  voxels outside them are 0 in the crop, as with drawn polygons, and they
  can be edited like those. They replace the region's polygons; *Undo fit*
  gives those back.
- **Opening another image could end in an error** in napari's layer list
  ("'NoneType' object has no attribute 'height'", the user's report of
  2026-11-08): napari 0.9.2 can draw a row whose layer has just been
  removed, and its loading indicator then fails. The window now skips the
  indicator for such a row (`_guard_layer_list`). Not reproduced here (six
  openings in a row, with and without the hidden rows, gave no error); the
  test recreates the row napari meets and shows napari's own code failing
  on it as in the report.
- **Fixes from the review of 0.16.0 (2026-11-08, 15 findings, all
  confirmed):** silhouette outlines now lie exactly where the region's box
  lies, with the whole margin (on pyramids whose levels record no
  translation, as eubi-bridge writes them, they were shifted by up to 7.5
  voxels, and the margin was rounded down to whole voxels of the level
  searched, so a crop with the outline could cut the object); a seed
  clicked on a projection is refused (its depth is hidden); a double-click
  (napari's zoom) no longer adds or removes a seed (a click counts once
  the double-click time has passed); closing *Find a bright object* ends
  *Add seeds*; *Undo fit* leaves regions changed since the fit alone (it
  could delete regions renamed or drawn on since) and says so; an object
  at the image's own edge is said to be there (no advice to search a
  larger region); *Settings…* OK no longer resets *Link zoom/pan*,
  *Compress TIFFs* and *exact level* of windows with an image (they are
  for new windows), sets the memory cache whatever another window shows,
  and a cancelled question keeps the exact-level tick; OME-TIFF companion
  sets with their TIFFs in a subfolder, or TIFFs with tifffile's own
  description, open as the whole set (they opened as the first TIFF alone,
  silently), a set opened from a TIFF whose metadata are in a master
  OME-TIFF opens again (as in 0.15.0), and a `.companion.ome.xml` set's
  crops are named after the set.
- **A tidier panel:** its parts (*Image*, *View*, *Crop regions*, *Find a
  bright object*, *Output*) open and close with a click on their title,
  and stay as you left them. *Memory cache* is the first row of *View*.
  In *Crop regions* the list is as tall as its regions (2 to 6 rows, then
  it scrolls), the *Polygons* part shows only while the selected region
  has polygons, and the start/stop fields are in a part of their own,
  *Ranges*, closed at first (the text under it shows the ranges anyway).
  *Save regions…* and the region's *Channels to crop* are in *Output*.
- **Settings…** (top right of the panel): which parts are open, *Compress
  TIFFs*, *Link zoom/pan*, *exact level* and napari's layer panels for new
  windows, the memory cache (napari's own setting), and whether
  Volume Cropper's own layers are listed in napari. Kept per user in
  `~/.volume_cropper/settings.json` (never in the shared tool folder).
- **napari's layer list** shows only your image channels and *crop
  regions* (where you draw): the layers the views make for themselves
  (region lines, crosshair, crop preview, shading) are no longer listed;
  they are switched in *View* as before (napari 0.9.2 cannot group
  layers). *Settings…* can list them again.

## 0.15.0 (part of 0.17.0)

- **Microscope files: Leica LIF, Zeiss CZI, Nikon ND2, Olympus OIR**
  (*Open… → Image file (TIFF, LIF, CZI, ND2, OIR)…*), read into memory
  like TIFFs (the part chosen in *What to load*; each image or position in
  the file is one of its images), with their voxel size, frame interval,
  channel names and colours; crops are OME-TIFFs. Read with liffile,
  czifile, nd2 and oirfile (no Bio-Formats, the user's choice): only the
  planes and region asked for are read (a LIF is memory-mapped, a CZI read
  by region and plane, an ND2 frame by frame). Checked on real LIFs and a
  CZI: the calibration is Bio-Formats' (the CZI's frame interval comes
  from its frame times), and parts read through the tool equal the
  library's own reads. ND2 and
  OIR could not be tried on real files yet (none to hand). Axes the tool
  does not know (tiles, views, colour samples) can be chosen as z, t or c
  in *What to load*. **The environment needs the readers:** `conda env
  update -n volume_cropper -f <tool folder>/env/environment.yml`.

From the user's test round of 2026-11-07:

- **Save as TIFF…** (a button, next to the crop buttons) replaces the tick
  box *Also save as TIFF*: it saves the selected region straight to a TIFF
  for Fiji at the **TIFF resolution** chosen above it (full resolution or
  a pyramid level), without making an OME-Zarr crop:
  `<image>_<region>_level<k>.tif`, checked against the image. The region
  is planned down to that level, so a small region has it too (before, a
  small region gave its finest level only). The resolutions have a row
  of their own (they were cut off) and read "full resolution: 2 × 1 × 1
  µm", "level 1: …". For a TIFF or layers read into memory the crop is a
  TIFF anyway, so the button is not shown. The notebook and the command
  line keep `tiff_level` (a crop plus its TIFF).
- **Scrolling the panel** with the mouse wheel or the touchpad no longer
  changes a number field or list that passes under the pointer: they take
  the wheel only after a click into them.
- **Output:** *Smallest level*, *Parallel workers* and *Keep position* are
  gone from the window; crops keep pyramid levels down to 64 voxels, copy
  with 8 threads and keep their position (they overlay the full image).
  The notebook and the command line still have all three.

## 0.14.0 (part of 0.17.0)

Experimental: **find a bright object** (asked 2026-11-07: "auto crops
around a bright object in 3D and across t, like when there's a single
embryo in the image").

- A new group under *Crop regions*, **Experimental: find a bright
  object**, off at first (tick it to use it): a channel, the threshold
  method (Otsu, Triangle or Li, as in Fiji's *Auto Threshold*), a margin,
  and **Fit region to object**. With a crop region selected it searches
  inside that region and shrinks it to the object (*Undo fit* gives it
  back); with none selected it searches the loaded image and adds a new
  region `object01`, `object02` … The region is an ordinary crop region
  afterwards.
- How: at each timepoint, a coarse pyramid level (at most about 20
  million voxels per timepoint) is smoothed a little and thresholded; of
  the pieces above the threshold, the one with the most signal is the
  object (not hot pixels, not a smaller or dimmer object), if it stands at
  least 3 noise levels above the background (otherwise "not found" at that
  timepoint: an automatic threshold splits even pure noise in two). With
  **Dim parts too** (on at first) the object is then grown down to just
  above the background, so that its dim parts are in the region too (the
  far side of an embryo when the light fades with depth, or tissue joining
  two bright parts: the threshold alone cut off up to 60 % of such objects
  in the review's tests). The region holds the object at every timepoint
  where it was found, padded by one voxel of the level searched, plus the
  margin, and its t range runs from the first to the last of them. Exact
  zeros in quantity (the padding of fused volumes) count as no data.
- Measured on the synthetic demos: on the embryo demo (embryos in bright
  tissue) **Dim parts too** takes in the tissue (the region spreads over
  it), so there it is turned off: the whole-image search then picks the
  brightest embryo (the region holds 100 % of it) in under a second, and
  around an embryo that stands clear of the tissue the box edges are
  within 2-16 µm of the true ones; without growing, an object can be cut
  short (one hollow embryo was), so check the region. On a synthetic
  single-embryo time-lapse (missing at the first timepoint, drifting 6 µm
  per timepoint), with growing, the region holds it at every timepoint
  from the second. Not yet tried on real data.
- A review (2 reviewers, 3 findings checked, all real; 11 in all) led to
  the growing, the padding, the zeros, and: the fit searches all of the
  selected region (it was cut to the part loaded for the views, which
  could drop timepoints from the region), a second click on *Fit* during
  a search is ignored, levels that shrink time are not searched, the
  time axis is found by its type.

## 0.13.0 (part of 0.17.0)

Volume Cropper inside napari: a napari plugin, for use within the lab (not
on PyPI or the napari hub).

- **Install it once** into the `volume_cropper` environment, from the tool
  folder: double-click `install_napari_plugin.bat` (Windows), or type
  `volume_cropper.bat --install-plugin` / `bash volume_cropper.sh
  --install-plugin`. It is an "editable" install: the environment points
  to the tool folder, so each release reaches the plugin without
  installing again. Nothing is written into the tool folder, and nothing
  is downloaded. `--uninstall-plugin` removes it.
- **In napari** (`conda activate volume_cropper`, then `napari`; on BARD
  `bash volume_cropper.sh --napari`, which sets up OpenGL as for the tool):
  - *Plugins › Volume Cropper*: a small panel that leaves the window as it
    is: *Crop the selected layers* opens a new Volume Cropper window with
    the selected image layers (several of the same size: the channels of
    one image), *Open Volume Cropper* an empty one, and *Use this window…*
    turns the window itself into Volume Cropper, after a warning (its
    layers are removed when an image is opened; napari's 2D/3D and grid
    buttons stay hidden until the window is closed); the tool's panel then
    sits to the right of the views, as in its own windows.
  - *Layers › Data › Crop selected layers…* (greyed out without a selected
    image layer) and *File › IO Utilities › Open in a new window*.
  - *File › Open Sample › Volume Cropper*: a small synthetic demo.
- **Layers from any napari reader** (e.g. another plugin's CZI or LIF
  reader) are read into memory like a TIFF (*What to load* first): names,
  colours, contrast, voxel size and position come from the layers. Lazy
  layers are read a whole chunk at a time. Their crops are OME-TIFFs,
  checked voxel by voxel. If the layers come from a file Volume Cropper
  can open itself (OME-Zarr, TIFF), it offers to open that file instead
  (an OME-Zarr is then read piece by piece, not copied). *What to load…*
  reads the layers again while they exist.
- Axes that napari does not name are guessed from their position, and the
  guesses can be changed in *What to load* (a 4D layer may be a z-stack
  over time or the channels of one stack). A layer's voxel size is kept
  when napari has no unit for it (micrometre is then assumed). Layers at
  different positions, or rotated ones, are refused as channels of one
  image.
- A closed Volume Cropper window now lets its image go, however it is
  closed (an image read into memory used to stay until napari was
  closed). Closing a window with its X during a crop asks first, then
  stops the crop; quitting napari stops it without asking.
- With the plugin installed, napari's menus also appear in the tool's own
  windows; they use the running tool (not a second copy).
- `env/environment.yml` pins setuptools (it was already there, through
  Qt); the installer needs it.

## 0.12.0 (part of 0.17.0)

Opening other files: TIFF so far (LIF, CZI, ND2 and OIR come next).

- **Open… → Image file (TIFF, OME-TIFF)** or **Folder of TIFF files** (one
  file per plane or per stack, in the order of their names, like Fiji's
  *File › Import › Image Sequence*). Plain TIFFs, Fiji hyperstacks and
  OME-TIFFs (each of their images), compressed or not.
- **Read into memory**, like opening an image in Fiji without "virtual
  stack": a *What to load* dialog for files chooses the image in the file,
  what an axis is when the file does not say (z, t or c), the calibration
  (voxel size and frame interval, filled in from the file and editable, as
  in Fiji's *Image › Properties*; it says which values are missing), the
  channels and the range of each axis to read (numbers 1, 2, 3 …). It
  shows the memory needed and refuses what would not fit (above 80 % of
  the free memory; a warning above half). Reading shows its progress and
  can be cancelled. The part read keeps its place in the file's
  coordinates (and a crop TIFF saved by this tool its place in the
  original image). Smaller copies for the views are made in memory (each
  the mean of 2 × 2 voxels, and of 2 × 2 × 2 once voxels are about cubes).
  Measured on a 160 MB course TIFF: read in about 1 s.
- **Crops of a file are TIFFs**, of the file's kind: a Fiji TIFF (.tif) for
  Fiji and plain TIFFs, an OME-TIFF (.ome.tif) for OME-TIFFs (and for the
  microscope formats to come), with the voxel size, frame interval,
  channel names and colours, and the crop's record (in Fiji: *Image › Show
  Info*). Each is checked voxel by voxel against the image in memory. Its
  record names the file, the image in it, what was read and where the
  calibration came from. Cropping such a crop again keeps the record of
  the first.
- **Compress TIFFs (zstd):** lossless zstd with the predictor, for crops of
  files and for *Also save as TIFF*; on synthetic data the files were about
  half the size. Fiji opens them through Bio-Formats (a Fiji .tif then shows
  the Bio-Formats dialog; an .ome.tif always goes through Bio-Formats).
  A compressed TIFF over 4 GB is a BigTIFF. For a file, the box starts
  ticked when the file itself is compressed.
- **2D images** (one z plane, e.g. a 2D movie) show only the XY view: no z
  fields, no 3D view, no projections.
- Command line: a TIFF as the source (`--image`, `--axes`, `--voxel-size`,
  `--unit`, `--frame-interval`, `--time-unit`), and `--compress`. The
  notebook stays for OME-Zarr images (it says so for a TIFF) and has
  `COMPRESS`.

## 0.11.0 (part of 0.17.0)

Cropping in time.

- **Every n-th timepoint:** each region has an *every n* field in its t
  row: the crop keeps timepoints first, first + n, … of the region's t
  range (which then ends at the last kept one), and its frame interval is
  n times the image's (so Fiji and napari show the right times). The
  check, the crop's record (`t_step`), the regions file (optional column
  `t_step`, written only when a region has a step), the command line and
  the notebook (`CropBox(..., t_step=3)`) all know it. Pyramid levels that
  also shrink time are left out of such crops.
- **Timepoint numbers:** the t fields show timepoints 1, 2, 3 … (like
  Fiji's frames) with their time beside them; the plan text says e.g.
  "t 2–8, every 3rd: 3 kept (30–210 min)". The *What to load* dialog
  counts timepoints from 1 too (it counted from 0). Regions and the
  regions file keep times (frame centres), as before.
- **Fading:** a region's lines (2D and its 3D frame) fade at timepoints its
  crop does not keep, and its crop preview is all black there.

## 0.10.0 (part of 0.17.0)

TIFF export for Fiji.

- **Output → Also save as TIFF, level k.** Each crop can also be saved as
  a TIFF next to it, `<image>_<region>_level<k>.tif`: one chosen level
  (0 = full resolution; a small region with fewer levels gives its
  coarsest, and says so). It is an ImageJ hyperstack (axes TZCYX) that
  Fiji opens with *File › Open*, calibrated: voxel size and unit, z
  spacing, frame interval and time unit, and the crop's position (Fiji's
  measurements, e.g. centroids, give the same µm coordinates as napari;
  Fiji's status bar shows pixel corners, half a voxel less in x and y, as
  always in ImageJ). Channel names, colours and display ranges come from
  the OME metadata (a single 8-bit channel opens grey); the crop's record
  is the image's *Info* (*Image › Show Info*). Voxels set to 0 by outlines
  stay 0.
- TIFFs are not compressed (ImageJ hyperstacks cannot be). The region's
  summary shows the TIFF's size, the disk-space question counts it, and
  TIFFs over 4 GB are asked about first: they use ImageJ's own large-file
  layout, which Fiji reads but other programs may not read fully.
- The TIFF is written a few slices at a time (never the whole level in
  memory), under a temporary name until complete, and then checked against
  the crop (size, voxel size and the voxels of a few slices). Existing
  TIFFs are replaced only with *Overwrite earlier crops*, and only TIFFs
  this tool wrote for the same region. Overwriting a crop also deletes
  that region's earlier TIFFs first (any level), so that no TIFF next to
  it shows the old region. Everything that could refuse (data type, axes,
  existing files) is checked before the first crop is written.
- Fixed (there since 0.3.0): the crop check reported "fill_value differs
  from the source" for float images whose fill value is NaN.
- Also on the command line (`--tiff LEVEL`) and in the notebook
  (`TIFF_LEVEL`). In the library: `export_tiff`, `verify_tiff`,
  `crop_box(..., tiff_level=...)`.
- `env/environment.yml` now pins tifffile 2026.9.20 (already installed
  with napari; nothing to update).

## 0.9.0 (part of 0.17.0)

Crop preview: see a region as it will be cropped before cropping it.

- **View → Preview crop: in the main views.** In XY, XZ and YZ,
  everything the selected region's crop does not contain on the slice
  shown is black: outside the region (exact edges) and the voxels its
  outlines set to 0. A slice or timepoint outside the region is all black.
  On a projection only the region's rectangle is shown.
- **View → Zoom to region** fits the 2D views to the selected region.

## 0.8.1 (2026-11-02, beta)

- **Fixed: a halo of data along the faces of the selected region in the
  3D view** with *Clip 3D to selected region* (reported by the user; there
  since the clipping was added). napari clips the volume exactly at the
  region's faces, but it draws a voxel that straddles a face partly
  inside, and it blends neighbouring voxels when drawing; the voxels just
  outside the region were not blacked out, so a thin layer of them
  showed, at full brightness even in *hide* and *dim*. Now, with
  clipping on, every voxel whose centre lies outside the region is 0 in
  the 3D view (in every mode), so *hide* with clipping shows exactly the
  voxels of the crop. Without clipping nothing changes. Checked on the
  embryo demo: before, 11.7 million voxels outside the region were still
  given to napari; now none.
- Corrected counts in the 0.8.0 notes: 9 new crop tests (not 10).

## 0.8.0 (2026-11-02, beta)

OME-Zarr 0.4 images, as well as 0.5.

- **OME-Zarr 0.4 (Zarr v2)** images open, show and crop like 0.5 images
  (the window, the notebook and the command line). A crop keeps its
  image's version: a 0.4 image gives a 0.4 crop, with the same chunks,
  compression and chunk files, and its OME metadata in the 0.4 place.
  Without shards (0.4), a crop is copied in blocks of whole chunks of
  about 32 MB (measured on a real 0.4 image: 2.3 s instead of 4.0 s with
  one chunk at a time). The check after a crop compares the 0.4 format
  too, and the crop's record names the source's format
  (`source_format`). The panel's image summary and the *What to load*
  dialog name the version. eubi-bridge's `OME/METADATA.ome.xml` is not
  copied into crops (as for 0.5); crops pass the official OME-Zarr 0.4
  schemas.
- **Clear messages** for images the tool cannot open: not OME-Zarr,
  OME-Zarr 0.1–0.3, axes without a type, a voxel size stored in a separate
  array, a bioformats2raw collection (open one of its numbered
  subfolders), and an unfinished (interrupted) crop. They replace the old
  "Only OME-Zarr 0.5 (Zarr v3) is supported".
- Images with only z, y and x (no channel axis), such as single-channel
  eubi-bridge output, are now tested (in 0.4 and 0.5).
- Tests: 65 (9 new for 0.4); a new check of the window,
  `v04_gui_test.py` (14 checks, on a real 0.4 image).

## Guide update (2026-11-02; the code is still 0.7.0)

- New section *Installing it → Your own copy*: how to copy the tool folder
  to your own computer (copy, don't move it), start it from there, point
  the notebook's `TOOL_DIR` at it, and update it. The shared copy stays
  the usual way to run it. The README points to the section.

## 0.7.0 (2026-10-31, beta)

Outlines in 3D, and *crop regions*.

- **Crop regions** (the name "crop box" is gone from the window and the
  guide): *New region* starts one and switches on the polygon tool,
  *Finish region* switches it off and selects none; *Rename…*, *Delete*.
  Everything drawn goes into the selected region (a new one if none is
  selected); click an empty part of the list to select none. *Drawing
  refines the selected box* and the old *Add* button are gone. Shorter
  texts in the panel (details in tooltips and the guide).
- **Polygons on several slices, in all views, make one 3D shape.** A
  polygon or lasso belongs to the slice on which it is drawn. All the
  polygons of a region, from every view, define one smooth shape that
  passes through each of them and is interpolated in between, in all
  directions at once (a variational implicit surface: radial basis
  functions with the cubic kernel). The shape ends at the outermost
  polygons along every axis, and the crop is fitted to it. Holes in the
  shape are filled. One polygon alone still reaches through the whole
  region, so older boxes files still work. 0.5.0 and 0.6.0 kept a voxel
  inside *any* view's outline, which kept too much: regions with polygons
  in more than one view now crop differently (each crop records its rule).
- **Silhouettes:** a polygon drawn on a projection (MIP or mean) is the
  object's outline seen through the projected depth, and reaches through
  that depth; silhouettes of several views keep what lies inside all of
  them. (A polygon drawn on a MIP used to count as the outline on one
  slice, which gave the shape holes and dents.)
- **Help while drawing:** small × marks show where the region's polygons
  in the other views cross the slice shown, and the panel warns when two
  views' polygons do not meet where their slices cross.
- **Margin:** grow (or, with a negative number, shrink) a region's shape by
  a number of voxels in all directions. Saved in boxes files (column
  `outline_margin_voxels`) and in each crop's record.
- **Display:** every 2D view shows the shape's outline on its current
  slice as a dashed line, also views in which nothing was drawn. New
  *Shade selected region* (under *View*): the voxels that will be kept in
  orange, those set to 0 in sky blue (unlike the usual LUT colours, and
  colour-blind safe), with your own colours and opacity. The polygon list
  names each polygon (xy1, xy2, ... per view in slice order; xyP1, ... for
  silhouettes) with its slice, and the views label them the same way
  (*Go to slice*, *Delete polygon*, *Clear polygons*); *Delete* removes
  just the selected polygon.
- Boxes files store each polygon's slice (`z=10: y x; … | z=30: …`), or a
  silhouette's depth range (`z=0..800: …`).
- **3D view:** on *Auto*, its level now follows the channels shown (it
  used to keep the level chosen when the view was switched on).
- **Time-lapse images are marked experimental** (in the window and the
  guide): check the first crops.
- **New demo image** `embryo_tissue_demo.zarr`: six embryo-like objects
  packed in tissue, three channels; the guide's pictures use it.
- `scipy` is now listed in `environment.yml` (it was installed already, as
  part of scikit-image).

## 0.6.0 (2026-10-28, beta)

Fixes from the review of 0.5.0 (27–28 October 2026), plus showing and
hiding single boxes:

- **Box names:** two boxes whose names would give the same file name (for
  example "tail 1" and "tail_1", or "A" and "a") are refused when a boxes
  file is loaded, when a box is renamed and before a crop. Before, the
  second crop could silently replace the first. *Overwrite earlier crops*
  replaces only a crop of the same box.
- **Boxes files (CSV):** mistakes are reported with the row number as Excel
  shows it: a range with only a start or only a stop (before, a stop alone
  silently cropped the whole axis), a box without a name, a cell that is
  not a number. Empty rows are skipped. Files saved by Excel with
  semicolons, decimal commas or Windows text encoding are read. Numbers are
  saved exactly (before: 6 digits, which could move a box edge by one
  voxel).
- **Output folder:** checked before it is created, so nothing is ever made
  inside the source image (before, a refused crop could leave an empty
  folder there).
- **The check after each crop** now also compares every level's position
  and voxel size (recomputed from the plan), the channel names and the
  crop's first corner with the source; before, a crop at the wrong position
  passed. It says what it checked ("checked OK (position, voxel size and
  format of 4 levels, and the voxels of 6 sample blocks per level)"), and
  its result is stored in the crop's record (`verified`). A failed check
  keeps the crop and shows a warning. Images whose empty value is NaN no
  longer fail the check.
- **Errors while cropping** stop the work still queued at once (before, all
  queued parts were still written first).
- **Show or hide single boxes:** each box in the list has a tick box;
  unticked boxes disappear from all views (lines, names, shapes, 3D frames)
  but are still cropped. *Show: crop boxes* still hides all of them.
- **Boxes are not lost without asking:** opening another image offers to
  save the boxes first; *Load boxes…* asks whether to add the file's boxes
  or replace the current ones.
- **Disk space:** if the output drive may be too small for the crops, the
  tool asks before starting; a full disk during a crop is reported in plain
  words.
- **Cancel** shows *Stopping…*, and afterwards says that the crop being
  written is incomplete and how to replace it. A Cancel or an error no
  longer also shows a second, red napari error message.
- The window title shows the version; the start message and the progress
  line use the same total (all levels).
- **Command line:** every box is checked before the first one is written
  (before, a bad box was found only after the boxes before it were
  cropped). Problems are reported in one line instead of a Python error
  report, with exit code 2 (before, an unknown `--names` box ended with
  "success", 0); 1 means a check after a crop failed, 130 Ctrl+C. Help for
  every option, the boxes-file layout in `--help`, `--version`, and a
  warning when the output drive may be too small.
- **Installing:** `environment.yml` also pins the exact Python version and
  the parts of napari the tool relies on (Qt, vispy, superqt and others),
  so that colleagues who install weeks apart get the same tool; checked
  with trial installs for Windows, Linux and Macs.
- **User guide:** the Mac and Linux steps use the same conda command as the
  launchers; more troubleshooting entries (network drive, failed install,
  "already exists", names that give one file name, empty 3D view); glossary
  entries for chunk, shard, translation and worker; the demo folder named;
  new pictures.
- **Notebook:** no example image or project details; it asks for your
  image and uses the `volume_cropper` kernel. It checks every box before
  the first is written.
- **Documents:** this file and DEVELOPERS.md split off the README, which now
  points to the user guide.

## 0.5.0 (2026-10-27, beta for testing in the group)

Built for fast opening.

- A *What to load* dialog appears when an image is opened: it shows the
  channels, timepoints, size and levels, estimates how much opening will
  read, and lets you choose the channels, a range per axis, the 2D level
  and the 3D view. Only that region is read and shown.
- The 3D view starts off; its *Auto* level fits 100 MB (was 300 MB) of the
  loaded region and the channels shown; contrast is measured only for the
  channels shown; a line under the image says how much was read.
- Fixed: the 2D views now start by showing the whole image. Before, the
  last view's zoom was used for all three, which cut off part of large
  images and read finer data.
- *View → Memory cache*: the size of the memory that keeps data already
  read (napari's own setting, remembered), how much is used, and *Clear
  cache*.
- **Outlines combine differently:** a voxel inside *any* of a box's
  outlines is kept (before: inside all of them), and the box grows to hold
  all its outlines, so none is cut off. Crops of boxes with outlines in more
  than one plane therefore differ from 0.4.0, also for boxes from an older
  CSV; crops record `"outlines_combine": "any"`. *Load boxes…* fits each box
  around its outlines (older boxes could be narrower); the notebook and the
  command line use a CSV's ranges as saved. All other crops are made exactly
  as in 0.4.0, except:
- *Keep position* off: the crop now starts at 0 µm (before, it kept the
  source's own offset, e.g. from stitching).
- 2D projections can also go through a *slab around the slice* (a
  thickness that moves with the slider) or *chosen ranges* per view.
- *View → Show*: hide the crosshairs, or the crop boxes, in all views.
- Cropping shows its progress in bytes written (all boxes and levels
  together), the speed, and the time left. `write_crop` has a new
  `progress_bytes` callback for this.

## 0.4.0 (2026-10-27)

*View → Show channels* and the `--channels` option, to show (and read) only
some channels; a user guide (`USER_GUIDE.md` / `.pdf`) replaces
`INSTALL.md`. Crops are made exactly as in 0.3.0.

## 0.3.0 (2026-10-23)

- Renamed to Volume Cropper. Until then the tool was called *omezarr_crop*
  (module `omezarr_crop.py`, launcher `crop_gui.bat`); crops made before
  say `"omezarr_crop"` in their provenance and are unaffected.
- A crop of a crop keeps its source's record (`source_provenance`), so the
  chain back to the original image is never lost.
- "Open last crop in a new window" opens a complete second window; boxes are
  drawn as solid lines, outlines as dashed lines; the 3D view can show the
  voxels that outlines will black out; time-lapse images are handled
  better.

## 0.2.0 (2026-10-23)

- Optional freehand/polygon *outlines*; voxels outside them are written as
  0.
- The window: four linked views, a choice of resolution level, MIP and mean
  projections, slice planes and clipping in 3D, lasso and polygon drawing.

## 0.1.0 (2026-10-21)

Plain boxes.
