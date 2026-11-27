"""Tests that names from files are shown as plain text in the tool's
windows (0.23.0, the pre-publication review).

Qt shows text that looks like HTML as HTML, and an <img src="..."> tag in
it makes Qt read the file it names (on Windows also from a network share,
which makes Windows send the user's login to that computer). Names from
image and regions files reach labels, message boxes, tooltips and napari's
layer names. Run with ``python run_all_tests.py text`` (or this file alone;
no window is shown). They check that:
- Qt's own label does read the picture an <img> tag names (the premise),
  while the tool's QLabel and QMessageBox show such text as it is;
- the tool's message boxes answer as Qt's do (the button clicked);
- literal() shows any text exactly as it is, and defang() gives a name
  that Qt never takes for HTML but that looks the same;
- the "What to load" summary escapes what it takes from the file.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile

from napari.qt import get_qapp       # (napari's: its OpenGL settings first)
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QImage, QTextDocument

from volume_cropper import qt_text


def _picture_tag(folder: Path) -> str:
    """An <img> tag naming a 300 x 300 picture: a label showing it is
    tall."""
    path = folder / "big.png"
    image = QImage(300, 300, QImage.Format.Format_RGB32)
    image.fill(0)
    assert image.save(str(path))
    return f'<img src="{QUrl.fromLocalFile(str(path)).toString()}">'


def test_labels_and_message_boxes_show_names_as_they_are():
    get_qapp()
    with tempfile.TemporaryDirectory() as tmp:
        name = "x" + _picture_tag(Path(tmp))
        qts = QtWidgets.QLabel(name)          # Qt's own: decides by the text
        ours = qt_text.QLabel(name)
        assert qts.sizeHint().height() >= 300          # it read the picture
        assert ours.sizeHint().height() < 100, ours.sizeHint()
        assert ours.textFormat() == Qt.TextFormat.PlainText
        assert ours.text() == name
        box = qt_text.QMessageBox(qt_text.QMessageBox.Icon.Warning, "Title",
                                  name)
        assert box.textFormat() == Qt.TextFormat.PlainText
        box.deleteLater()


def test_message_boxes_answer_as_qts_do():
    """warning, critical, information and question take Qt's arguments and
    give the button clicked; the text is shown as it is."""
    get_qapp()
    button = qt_text.QMessageBox.StandardButton
    seen = []

    def clicking(which):
        def exec_(box):
            seen.append((box.textFormat(), box.text(), box.windowTitle()))
            box.button(which).click()
        return exec_

    original = qt_text.QMessageBox.exec
    try:
        qt_text.QMessageBox.exec = clicking(button.No)
        assert qt_text.QMessageBox.question(None, "Q", "<b>Sure?</b>") \
            == button.No
        qt_text.QMessageBox.exec = clicking(button.Yes)
        assert qt_text.QMessageBox.question(
            None, "Q", "x", button.Yes | button.No, button.No) == button.Yes
        qt_text.QMessageBox.exec = clicking(button.Ok)
        for kind in ("warning", "critical", "information"):
            assert getattr(qt_text.QMessageBox, kind)(None, kind, "<i>a</i>") \
                == button.Ok
    finally:
        qt_text.QMessageBox.exec = original
    assert len(seen) == 5
    assert all(fmt == Qt.TextFormat.PlainText for fmt, _, _ in seen)
    assert seen[0][1] == "<b>Sure?</b>" and seen[2][2] == "warning"


def test_literal_and_defang_keep_the_text():
    get_qapp()
    text = 'a <img src="x.png"> & <b>b</b>\nsecond line'
    document = QTextDocument()
    document.setHtml(qt_text.literal(text))
    assert document.toPlainText().replace(" ", "\n") == text
    with tempfile.TemporaryDirectory() as tmp:
        name = "0: " + _picture_tag(Path(tmp))
        safe = qt_text.defang(name)
        assert safe.replace("​", "") == name          # looks the same
        label = QtWidgets.QLabel(safe)          # Qt's own, as in napari
        assert label.sizeHint().height() < 100, label.sizeHint()


def test_load_dialog_summary_escapes_the_file_name():
    """The "What to load" summary is formatted (bold): the image's name
    goes in escaped, so it shows as it is ("&" here; "<" and ">" cannot
    be in a Windows file name, but in a Linux or Mac one)."""
    get_qapp()
    from test_volume_cropper import make_test_image
    from volume_cropper import core as vc
    from volume_cropper.load_dialog import image_summary
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a&b.zarr"
        make_test_image(path)
        summary = image_summary(vc.open_image(path))
        assert "<b>a&amp;b.zarr</b>" in summary, summary
        document = QTextDocument()
        document.setHtml(summary)
        assert document.toPlainText().startswith("a&b.zarr (OME-Zarr 0.5")


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["text"]))
