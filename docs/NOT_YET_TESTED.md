---
title: "Volume Cropper 0.23.0"
subtitle: "What has not been tested yet (27 November 2026)"
lang: en
---

Volume Cropper is checked by about 290 automatic tests and by scripted runs
of the real window, on **synthetic images** made for the purpose (with
known true answers). Some parts have also been used on real data (see the
end). Everything below has **not** been tried yet, or only partly. If you
use one of these, please check the result and tell the maintainer how it
went, also when it worked.

# Computers

- **Mac:** never run on a real Mac. The environment recipe was only
  checked to install (a "dry run" for Intel and Apple Silicon Macs, last
  on 17 November 2026); it needs macOS 11 (Big Sur) or newer.
- **Linux and BARD:** the window has not been tried on BARD (EMBL's remote
  desktop) since an early version (0.2); the launcher's drawing set-up
  (`vgl`, `egl`) was made for BARD but needs a new check. Since 0.21.0
  the launcher `volume_cropper.sh` no longer changes folder before it
  starts the tool; that has not been run on Linux or a Mac yet. With
  0.20.0 the command line ran on the cluster and the notebook on the
  JupyterHub (22 November 2026); 0.21.0's command
  (`volume_cropper/core.py`) and the moved notebook have been run on
  Windows only.
- **Windows:** used on one PC only.
- **Several people** using the shared copy at the same time.

# File formats

- **OME-Zarr 0.6, zipped OME-Zarr (.ozx, .zip), collections** (0.21.0):
  tested on synthetic images, on images written by
  another program (ngff_zarr 0.43) and on two public `.ozx` examples, but
  not yet on real lightsheet data, not on large crops (several GB; the
  ZIP64 format for files over 4 GB was only checked on small files), and
  not for reading speed over the network drive (only on a local disk,
  where the zip read faster than the folder). Real 0.6 images may still
  be labelled `0.6.dev4` by their writer; such crops keep that label,
  which the official 0.6 checker does not accept (nor the source).
- **Nikon ND2:** no real file to hand; tested only through the reading
  library's own behaviour. Please compare your first ND2 with Fiji
  (Bio-Formats): voxel size, frame interval, channels, positions.
- **Olympus OIR (0.17.1):** tried on a few real files (z-stacks and a
  2D image): the parts read were identical to the
  reading library's own, and a crop of one (two channels) was read back
  identical. The channel names were one channel off before 0.17.1. Not
  compared with Fiji yet: please compare your first OIR's voxel size and
  channel names with Fiji. Not tried: time-lapses. The reader cannot read
  compressed or tiled (mosaic) OIR files.
- **Zeiss CZI:** tried on one file only, a 2D movie (one scene, one
  channel; in 0.17.1 also cropped in time, every 4th timepoint and as a
  moving region). Not tried: z-stacks, several scenes or channels, tiles or
  mosaics, light-sheet CZIs with several views or illuminations, CZIs split
  over several files.
- **Leica LIF:** tried on seven files (z-stacks with one to five channels,
  2D images, tile scans and an RGB image). Since 0.17.2 an RGB image opens
  as three channels (red, green, blue, in the order liffile gives); please
  compare the colours of your first RGB LIF with Fiji. Not tried:
  time-lapses. LOF and XLEF (0.23.0): one real XLEF experiment was opened
  and read (images whose frames are .lof files and OME-TIFFs); a .lof
  opened on its own was not tried.
- **Stitched LIF tile scans (0.17.2):** tried on several real tile scans
  from one Leica confocal (2D and z-stacks). With tiles of 512 or 1024
  pixels, the tiles sat within a few pixels of LAS X's own placement, and the raw tiles' overlaps matched at least as well at
  Volume Cropper's places as at LAS X's. Overview scans with tiles of 64-128
  pixels differed from LAS X by several pixels, and their overlaps
  match only weakly: their placement is uncertain. Not tried: other Leica
  microscopes (the file's orientation flags were never set in these files;
  when they are, the orientation is worked out from the overlaps, which
  was tested only on synthetic tiles), tile scans that are time-lapses
  (the tiles are placed once, from the first timepoint), colour (RGB) tile
  scans, very large mosaics (hundreds of large tiles: the time to place
  them and the memory). Please look at the seams of your first stitched
  images.
- **Colour (RGB) CZI, ND2 and OIR images (0.17.2):** open as three
  channels (red, green, blue) like RGB LIFs; not tried on a real file.
- **OME-TIFF sets** made by microscopes (e.g. Viventis): tried on
  synthetic sets only.
- **Very large files** read into memory (tens of GB): only the memory
  check was tested, not reading such a file.
- **Fiji opening the TIFFs:** TIFF output was checked by reading it back
  with Python's TIFF library, and with Fiji on test files during
  development; not yet by a person on real crops: uncompressed and
  compressed (Bio-Formats), Fiji TIFF and OME-TIFF, and TIFFs over 4 GB.
- **Other programs reading the OME-Zarr crops** (Fiji with BigDataViewer
  or n5-viewer, QuPath, Imaris): only napari and Python's OME-Zarr library
  have read them.

# The Autocropper and time-lapses

- **Real time-lapses:** only a 2D movie (CZI, several hundred timepoints) has been
  cropped (0.17.1 review): in time, every 4th timepoint (read back
  identical to the source) and as a moving region (checked by the tool
  only). No real 3D time-lapse yet.
- **The Autocropper on real data:** tried on synthetic embryos (an embryo
  demo and a drifting, growing synthetic embryo), and by the maintainer
  while testing 0.20.0. Not known yet: how well it finds embryos in other
  people's images, how often it loses one over time, how seeds behave on
  other tissues.
- **Splitting touching objects between seeds (0.20.0):** checked on
  synthetic touching objects (two ellipsoids touching at an angle: each
  crop kept 99.5 % of its own object and 0.5 % of the other); gap lines
  and *Recompute split* by the automatic tests and scripted window runs.
  Not known yet: how it does on real touching embryos of uneven
  brightness, and how long it takes on large images.
- **Outlines per timepoint on large data:** on a small synthetic
  time-lapse (78 MB, 8 timepoints) the fit took 4.5 s instead of 2.8 s and
  the crop took about as long as without outlines. How much slower it is on
  large data is not known yet.
- **A moving region with every n-th timepoint, Save as TIFF, the
  notebook and the command line:** tested by the automatic tests only.

# Other parts

- **Changing a region in the 3D view (0.20.0):** used with a mouse on
  Windows; not tried with a touchpad, on a Mac or on BARD (where 3D
  drawing goes over the network and may be slower).
- **Levels, the preview's projection and outlines in 3D (0.20.0) on large
  images:** timed on the synthetic demos only (there, Levels recomputes in
  under 0.1 s). How long they take on very large images over the network
  is not known yet.
- **The napari plugin:** tested on Windows only (installing, the menus,
  cropping layers; for 0.21.0 also a plugin installed from 0.20.0, which
  kept working after the update); not on Mac or BARD, and not by a person.
- **The notebook** (`examples/volume_cropper.ipynb`): run end to end for
  0.21.0 on the synthetic demo, on Windows (a crop and its TIFF, both
  checked); with 0.20.0 it was also used on the JupyterHub.
- **The regions file in Excel:** reading and writing Excel's own CSV
  formats was tested; regions that move (the `track` column, the JSON file
  next to it) have not been opened and saved in Excel.
- **Very large crops:** with 0.20.0, four crops of real lightsheet images
  were written on a cluster (the largest about 0.5 TB uncompressed, in
  under half an hour), all checked OK by the tool. Not tried
  yet: very large crops saved as TIFF, or zipped (0.21.0).
- **Slow networks:** speeds were measured over Wi-Fi only; a network cable
  is expected to be faster.

# What has been used on real data

For comparison: OME-Zarr 0.5 lightsheet images of fixed samples (opening,
crop regions, outlines, crops checked voxel by voxel at several levels), an
OME-Zarr 0.4 image (crops identical to the source at every level), and LIF
and CZI files (calibration identical to Fiji's Bio-Formats, and the parts
read identical to the reading library's own).
