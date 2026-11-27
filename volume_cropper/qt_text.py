"""Text from image and regions files, shown safely in Qt (0.23.0).

Qt shows the text of labels, message boxes, tooltips and some of its own
dialogs as HTML ("rich text") whenever the text looks like HTML. Names read
from files (channels, axes, units, scenes, regions, file names) end up in
such texts, and an HTML image tag in a name, <img src="...">, makes Qt read
the file it names: on Windows also from a network share, which makes
Windows connect to that computer and send it the user's login (found by the
pre-publication review, 2026-11-27). So the tool's modules use, instead of
Qt's own:

- QLabel and QMessageBox from here: they show plain text. A label the tool
  formats itself is set to rich text explicitly, with every name in it
  passed through escape();
- literal(text) for tooltips and for the labels of Qt's own dialogs
  (QInputDialog, QProgressDialog): rich text that shows the text exactly as
  it is, whatever it holds;
- defang(name) for names that napari shows (layer names): napari's own
  labels decide by the text too.
"""
import html

from qtpy import QtWidgets
from qtpy.QtCore import Qt


def escape(text) -> str:
    """Text for a place inside HTML: shown exactly as it is."""
    return html.escape(str(text), quote=False)


def literal(text) -> str:
    """Rich text that shows `text` exactly as it is (line breaks kept), for
    places where Qt decides itself whether a text is HTML (tooltips, the
    labels of Qt's own dialogs)."""
    return "<qt>" + escape(text).replace("\n", "<br>") + "</qt>"


def defang(name) -> str:
    """A name that Qt never takes for HTML, for widgets of others (napari's
    layer names): an invisible zero-width space after every "<", so no tag
    can start there; it looks the same."""
    return str(name).replace("<", "<​")


class QLabel(QtWidgets.QLabel):
    """A label that shows its text as plain text. (A label the tool formats
    itself: setTextFormat(Qt.TextFormat.RichText), and escape() for every
    name in it.)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTextFormat(Qt.TextFormat.PlainText)


class QMessageBox(QtWidgets.QMessageBox):
    """A message box that shows its text as plain text. Its warning,
    critical, information and question take the same arguments as Qt's own
    and give the same answer: the button clicked, or for Esc and closing
    the window the escape button, as Qt's do."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTextFormat(Qt.TextFormat.PlainText)

    @staticmethod
    def _show(icon, parent, title, text, buttons, default):
        button = QtWidgets.QMessageBox.StandardButton
        box = QMessageBox(icon, title, text, buttons, parent)
        if default is not None and default != button.NoButton:
            box.setDefaultButton(default)
        box.exec()
        clicked = box.clickedButton()
        return box.standardButton(clicked) if clicked is not None \
            else button.NoButton

    @staticmethod
    def warning(parent, title, text, buttons=None, defaultButton=None):
        return QMessageBox._show(
            QtWidgets.QMessageBox.Icon.Warning, parent, title, text,
            buttons or QtWidgets.QMessageBox.StandardButton.Ok, defaultButton)

    @staticmethod
    def critical(parent, title, text, buttons=None, defaultButton=None):
        return QMessageBox._show(
            QtWidgets.QMessageBox.Icon.Critical, parent, title, text,
            buttons or QtWidgets.QMessageBox.StandardButton.Ok, defaultButton)

    @staticmethod
    def information(parent, title, text, buttons=None, defaultButton=None):
        return QMessageBox._show(
            QtWidgets.QMessageBox.Icon.Information, parent, title, text,
            buttons or QtWidgets.QMessageBox.StandardButton.Ok, defaultButton)

    @staticmethod
    def question(parent, title, text, buttons=None, defaultButton=None):
        button = QtWidgets.QMessageBox.StandardButton
        return QMessageBox._show(
            QtWidgets.QMessageBox.Icon.Question, parent, title, text,
            buttons or (button.Yes | button.No), defaultButton)
