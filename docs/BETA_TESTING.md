# Testing Volume Cropper (beta)

Thank you for trying Volume Cropper. It has been tested on Windows with
synthetic and real lightsheet images. These checks have **not** been done
yet, and your help with them is most useful:

1. **Mac:** install it and start it as the user guide (*Installing it →
   Mac*) describes. Does the window open? Can you open the demo image and
   make a crop?
2. **BARD (EMBL's remote Linux desktop):** install and start it as in
   *Installing it → Linux*. Does the window open (note the "vgl" or "egl"
   line in the terminal)? Do the 3D view and a crop work?
3. **Fiji:** open one of your crops in Fiji (or BigDataViewer), with the
   OME-Zarr reader you normally use. Does it open, and does it look like
   the same region in the full image (voxel size, position, channels)?
4. **Your own data:** crop a region you know well, then check the crop
   against the source, for example in napari with *Keep position* on (the
   crop should lie exactly on the full image).
5. **OME-Zarr 0.4** (new in 0.8.0): open a 0.4 image, for example one
   made by eubi-bridge, crop a region, and open the crop in the software
   you normally use for 0.4 images. Crops of 0.4 images stay 0.4.
6. **OME-Zarr 0.6 and zipped OME-Zarr** (new in 0.21.0): if you convert
   images to OME-Zarr 0.6 or to a zipped file (`.ozx`), for example with
   EuBI-Bridge, open one, crop a region, and open the crop in the software
   you normally use. Crops of a zipped image are zipped too.

**When something goes wrong or is unclear,** please tell the maintainer:

- what you did, what you expected, and what happened;
- a screenshot of the window and of the terminal window (Windows: the black
  window that opens first; Mac and Linux: Terminal), including its last
  lines;
- the version: it is in the window's title, e.g. "Volume Cropper 0.7.0".

Wishes and "this was confusing" are just as welcome as error reports.
