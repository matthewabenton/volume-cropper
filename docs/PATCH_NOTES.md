---
title: "Volume Cropper 0.23.0"
subtitle: "Patch notes: what is new since 0.8.1 (beta, 27 November 2026)"
lang: en
---

Everything here is explained in the user guide (`USER_GUIDE_<version>.pdf`). The
full technical list is in `CHANGELOG.md`.

# 0.23.0 (27 November 2026): safer with files from elsewhere

Nothing changes in how you use the window: no new environment is
needed, and the plugin keeps working without installing it again.

- **Safer with files from elsewhere.** A review before publishing found
  ways in which a crafted file could make Volume Cropper read other
  files, or make Windows connect to another computer and send it your
  login. Now:
  - Leica: a .lif or .lof that names other files is refused; an .xlef
    experiment opens only if every file it names lies in its folder
    (tried on a real experiment: it opens as before);
  - TIFF: a Micro-Manager NDTiff index beside a TIFF is ignored (the
    TIFF is read as it is), and a folder of TIFFs is read file by file;
  - names from files (channels, axes, regions ...) are always shown as
    text in the window;
  - an OME-Zarr whose axis names or units hold < > & or " is refused;
  - a region name that starts with = + - or @ is saved with a ' in
    front, so that Excel shows it as text; Volume Cropper reads it back
    unchanged.
- **The notebook:** a copy of it kept in your own folder no longer finds
  the tool by itself. Set `TOOL_DIR` in its first code cell to the tool
  folder; if you forget, it stops and says so. The notebook in the tool's
  `examples` folder finds the tool as before.
- **Folders:** the tool is public now, so the README, the user guide and
  the environment file no longer name the folders of the group's network
  drive; they say `<tool folder>`. A note next to the shared tool
  folder lists them.
- **Demo images without a shared copy:** the guide says how to make them
  with two scripts in the tool's `dev` folder.
- The user guide is now `USER_GUIDE_0230.pdf`.

# 0.22.1 (27 November 2026): an open-source licence

Nothing changes in how the tool works: no new environment is needed, and
the plugin keeps working without installing it again.

- **Open source:** Volume Cropper now has a licence, the BSD 3-Clause
  licence (the file `LICENSE`, copyright EMBL and Matthew Benton): anyone
  may use, change and share it, as long as the copyright notice stays with
  it. This lets it be shared on GitHub with people outside the group.
- **README:** a short section for people outside the group, and who
  maintains the tool.
- The user guide is the same as for 0.22.0, now as `USER_GUIDE_0221.pdf`.

# 0.22.0 (25 November 2026): zoom the histogram to min–max, README as PDF

No new environment is needed (the same as for 0.17.0), and the plugin keeps
working without installing it again.

- **Levels: zoom to min–max.** A new tick box next to *log* spreads min to
  max across the histogram, with room on either side (a quarter of the
  distance between them) to drag them further out. The bins are computed
  again over that range, so a narrow range shows its detail, for example
  when a few very bright voxels squeeze everything else to the left. The
  axis follows when you let go of a line or type a value; to go further
  than the room shown, drag a line to the edge, let go and drag again. The
  numbers below the histogram still count all the voxels. *Settings…* can
  start every new window with it ticked.
- **README.pdf and BETA_TESTING.pdf:** the tool folder's start page and the
  notes for testers now come as PDFs too, for reading without a Markdown
  viewer.

# 0.21.0 (24 November 2026): new OME-Zarr formats, a tidier tool folder

No new environment is needed (the same as for 0.17.0). If you added Volume
Cropper to napari's menus (the plugin), it keeps working without installing
it again; napari's list of plugins may still say 0.20.0, but the panel shows
the right version.

## New formats

- **OME-Zarr 0.6** images open and crop like 0.4 and 0.5 ones; their
  crops are 0.6 too.
- **Zipped OME-Zarr** (`.ozx`, `.zip`): the whole image in one file. It
  opens like a folder (*Open… › Zipped OME-Zarr*), read piece by piece
  without unzipping, and its crops are zipped too. A zip made with
  compression (e.g. Windows' *Send to › Compressed (zipped) folder*) opens
  as well, but more slowly; the tool says so.
- **Collections** (one OME-Zarr holding several images, e.g. made by
  bioformats2raw, or an OME-Zarr 0.6 *scene*): a small window asks which
  image to open. Crops are named `<collection>_<image>_<region>`.

The user guide explains them (*OME-Zarr 0.6, zipped OME-Zarr and
collections*). They have not been tried on real lightsheet data yet (see
`NOT_YET_TESTED.pdf`).

## A tidier tool folder

The tool folder now holds only what you need to start: `README.md` (start
here), the launchers, and the folders `docs` (these notes, the user guide
and the other notes), `examples` (the notebook), `env` (the environment
recipe) and the program itself (`volume_cropper`, `napari_volume_cropper`).
Material for developers is in `dev` and `tests`.

- **Nothing changes** when you start it with `volume_cropper.bat` or
  `volume_cropper.sh`, or use it from napari's menus.
- **The notebook moved** to `examples/volume_cropper.ipynb`. A copy you
  made before stops with an error in its first code cell: copy the
  notebook again and fill in your parameters.
- **Your own Python scripts:** write `from volume_cropper import core as vc`
  instead of `import volume_cropper as vc`, and
  `from volume_cropper import file_images` instead of `import file_images`.
- **The command line** is now
  `python <tool folder>/volume_cropper/core.py <image>.zarr boxes.csv <output folder>`
  (it was `<tool folder>/volume_cropper.py`): change cluster job scripts
  that use it.

# 0.20.0 (17 November 2026): a major step

The version jumps from 0.17.2 to 0.20.0 to mark a big release; there is no
0.18 or 0.19. No new environment is needed (the same as for 0.17.0). On a
Mac it needs macOS 11 (Big Sur) or newer, as 0.17 did; the guide now says
so.

## Opening an image

- **Settings: what opening starts with.** The 2D and 3D resolutions can be
  chosen by voxel size (so that images from other set-ups open with the
  same detail), all channels or only the first, and whether *What to
  load* asks at all (if not, the image opens at once; a very large 3D view
  still asks).
- With a 2D resolution chosen, the views open zoomed in so that they
  really show it.
- A **"Loading…" box** says what is being read while the window is busy.

## The panel

- **napari's left panels are hidden at first.** The panel has its own
  *Channels* part (show, colour, and choose the channel for *Levels*),
  auto-contrast, and the drawing tools as buttons. *View: napari's layer
  panels (left)* brings napari's back.
- **Levels (histogram)** for one channel, with numbers: the values and
  counts, the contrast limits as lines you can drag, gamma (a dashed
  midtone line, as in Photoshop's *Levels*), and boxes for exact min, max
  and gamma. It shows the XY view as shown, the whole image, or what the
  selected crop region keeps. Contrast and gamma change only the display.
- **Settings: what new windows start with** for the Autocropper and the
  *View* part. The settings file now keeps only what you changed, so that
  later versions' new defaults reach you.

## Crop regions

- **Undo and redo** for every change to the crop regions (*Ctrl+Z*,
  *Ctrl+Shift+Z*, or the buttons under the list).
- **Several regions or polygons** can be selected and deleted at once.
- **Change a region in the 3D view:** drag the dot on a face to move that
  face, or the centre dot to move the region; *Shift* resizes around the
  centre, *Ctrl* snaps to round values, *Up* / *Down* nudge the last face
  by a voxel. A click on a region's frame in 3D selects it.
- **Preview crop with a projection** shows the projection of exactly what
  the crop keeps.
- **Outlines in 3D:** what the crop keeps, shown as see-through surfaces
  (all regions or only the selected one).
- **Fewer × marks** where they would crowd (zoom in to see them all), and
  *View: Show × marks* to hide them.

## Autocropper (experimental)

- **Your own threshold:** a slider, a number box and *Auto*, with a dotted
  orange outline of what lies above it in the 2D views, and a line in the
  *Levels* histogram.
- **What a seed finds:** the most stable part (as before), the brightest
  stable part, or the whole object.
- **Touching objects split between their seeds**, as in ilastik: put a
  seed on each object; where no threshold separates them, *Paint gap*
  lines mark the boundary, and *Recompute split* applies new lines.
  Overlapping objects are split too.
- **Outlines per slice (XY)** as an option; the parts of a split always
  get them.
- Seeds are shown in 3D, and one progress window covers a whole fit.

## Fixed

- Opening a second image at a chosen coarse resolution could fail
  ("IndexError: list index out of range").
- *Save as TIFF…* could leave an empty folder `.volume_cropper_tmp_…`
  next to the TIFF. It now tries again, says so if one stays, and clears
  old empty ones at the next save into that folder.
- Dragging in the 3D view is several times smoother.

# 0.17.2 (11 November 2026)

No new environment is needed (the same as for 0.17.0).

- **Leica tile scans open stitched.** A LIF tile scan (a mosaic) is
  listed twice in *What to load*: "… (stitched)", one image with each tile
  placed where its overlaps with its neighbours match best (a few seconds
  when you choose it), and "… (tiles)", the tiles one after the other as
  before. Tiles move by whole voxels only and nothing is blended: where
  tiles overlap, each voxel comes from the tile whose centre is nearest,
  so all values are the measured ones. The dialog says how well the
  overlaps matched; look at the seams before cutting across them. See
  *Tile scans (Leica LIF)* in the guide.
- **LAS X's merged images get a note:** LAS X blends tiles where they
  overlap, so their voxel values are not all the measured ones.
- **Colour (RGB) images** in LIF, CZI, ND2 and OIR files open as three
  channels, red, green and blue (they opened with the colours as z).
- **Lassos:** where a lasso goes over the same area twice (overshooting
  its start), that area is now kept in the shape; it was left out as a
  hole. **Crops with such polygons can differ from 0.17.1.** A polygon with
  (almost) no area, e.g. three clicks on a line, is refused.
- **The Autocropper** keeps following an object past timepoints where it
  does not find it (the crop region used to end there).
- **The user guide is now named after the version** (this one:
  `USER_GUIDE_0172.pdf`), so that it can be updated while someone has the
  old one open.

# 0.17.1 (10 November 2026): fixes

No new environment is needed (the same as for 0.17.0).

- **Polygons follow your drawing more closely.** Polygons drawn in one
  view (e.g. on a few XY slices) are now blended from one drawn slice to
  the next: the same outline on two slices gives a straight shape (it
  bulged before), and a quick extra slice drawn a little off no longer
  bends the shape far away. A lasso that overshoots its start, crosses
  itself or wobbles is filled as Fiji fills it (it could lose half of
  itself before). **Crops with polygons can differ from those made with
  0.17.0** (closer to what was drawn).
- **Safer with files from elsewhere.** An image or regions file that
  names files outside its own folder, or a network share, is refused; a
  multi-file OME-TIFF with a missing file is refused (its planes were
  zeros). *Overwrite earlier crops* replaces only an earlier crop of the
  same region of the same image (two images with the same name in
  different folders used to replace each other's crops and TIFFs).
- **Fixed in the window:** *What to load…* with a region selected
  crashed (always in the plugin); *Use this window…* crashed when napari
  showed 3D; a moving region dragged at the image's edge shrank at every
  timepoint; with channel 0 hidden the level shown was wrong; the
  Autocropper over one timepoint made no outline.
- **Fixed for files:** Olympus OIR channel names were one channel off; a
  LIF's time between frames could be 1000 times too long; the command line
  now writes a moving region's positions file.

# Since 0.8.1 (0.17.0)

# Before you start: update your environment once

Version 0.17.0 reads microscope files with a few new packages. In Miniforge
Prompt (Windows) or Terminal (Mac, Linux), run once:

```
conda env update -n volume_cropper -f "<tool folder>/env/environment.yml"
```

Without them everything else works; opening a LIF, CZI, ND2 or OIR file
then tells you this command.

# New

**Other files: TIFF and microscope files.** *Open…* now also opens TIFF and
OME-TIFF files (also OME-TIFF sets spread over several files), folders of
TIFFs (one file per plane, like Fiji's *Image Sequence*), and Leica LIF,
Zeiss CZI, Nikon ND2 and Olympus OIR files, without Bio-Formats. They are
read into memory: a *What to load* dialog chooses the image in the file,
the channels, the part to read and the calibration (as in Fiji's *Image ›
Properties*), and says whether it fits into memory. Their crops are TIFFs
(Fiji TIFF or OME-TIFF), checked voxel by voxel. 2D images (e.g. a 2D
movie) show only the XY view.

**Save as TIFF… for Fiji.** Saves the selected region straight to a TIFF at
the resolution you choose (full resolution or a coarser level), calibrated
for Fiji (voxel size, frame interval, position, channel names and colours).
**Compress TIFFs (zstd)** makes them about half the size (Fiji then opens
them through Bio-Formats).

**Cropping in time.** The *t* fields count timepoints 1, 2, 3 … like Fiji.
**Every n-th timepoint** keeps only every n-th timepoint of a region's
range (the crop's frame interval is then n times longer). A region's lines
fade at timepoints its crop does not keep.

**Crop preview.** *View → Preview crop: in the main views* shows in black
everything the selected region's crop will not contain on the slice shown.
*Zoom to region* fits the 2D views to the selected region.

**Autocropper (experimental).** Fits crop regions to bright objects by
itself, for example a single embryo, in 3D and over time:

- *Fit region to object* (inside a region, or in the whole loaded image),
  *Fit all regions* (one object per rough region), and **seeds**: click
  on objects, then *Fit to seeds* gives one region per object, each with a
  threshold of its own, followed through time from the seed. Seeds turn the
  colour of the region they gave, follow it through time, and are listed
  in a table (click a row to go there).
- *Dim parts too* keeps an object's dim parts; *Outline (silhouettes)* also
  shapes the region to the object's outline in XY, XZ and YZ.
- **Regions that move with the object** (time-lapses): one region of one
  size that follows a drifting embryo, cropped into one time-lapse in which
  the embryo stays in place. A file `<crop>_positions.csv` says where each
  timepoint came from. With *Outline*, every timepoint gets its own outline
  of the object (slower on large data; the panel warns).
- *Undo fit* takes the last fit back.

It has been tried on synthetic images only: please check its regions on
your data, and tell the maintainer how it does.

**In napari's own window (optional plugin).** Install once with
`volume_cropper.bat --install-plugin` (or `install_napari_plugin.bat`);
napari then has *Plugins › Volume Cropper*, which crops image layers you
opened in napari (with any reader) or opens Volume Cropper windows.

# Changed

- **The panel** has parts that open and close with a click on their title,
  and remembers them. **Settings…** keeps your own choices between sessions
  (only for you, never in the shared folder).
- **Output** is simpler: *Smallest level*, *Parallel workers* and *Keep
  position* are gone from the window (crops keep levels down to 64 voxels,
  write with 8 threads and keep their position; the notebook and the
  command line still have them). *Channels to crop* and *Save regions…*
  are under *Output*.
- napari's layer list shows only your image channels and *crop regions*.
- The × marks (where another view's polygons cross the slice) are drawn in
  that view's colour: blue XY, green XZ, red YZ.
- Scrolling the panel with the mouse wheel no longer changes the number
  fields it passes over.
- The regions file can hold `t_step` (every n-th timepoint) and, for
  moving regions, `track`, with their outlines per timepoint in a file next
  to it (`<file>_outlines_per_timepoint.json`); older files read as before.

# Fixed

- Saving a compressed TIFF: the progress bar stopped at 99 % (the TIFF was
  complete; the last step was not shown).
- Opening another image could end in an error in napari's layer list
  ("'NoneType' object has no attribute 'height'").
- The crop check reported "fill_value differs from the source" for float
  images whose fill value is NaN.
- Many smaller fixes from three reviews of the new parts (see
  `CHANGELOG.md`).

# Not changed

OME-Zarr crops are cut, written and checked as in 0.8.1, and old regions
files and crops work as before. What could not be tested yet is listed in
`NOT_YET_TESTED.pdf`.
