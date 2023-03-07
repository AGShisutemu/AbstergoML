import sys
from PyQt5 import QtWidgets


class Console(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        self.insertPlainText(text)

    def toggle_console(self):
        # Toggle the visibility of the console widget
        self.console.setVisible(not self.console.isVisible())
