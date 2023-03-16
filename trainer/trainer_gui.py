import sys
import os
import re
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTableView,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QPlainTextEdit,
    QToolBar,
    QToolButton,
    QMenu
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QTextCursor
import shutil

from trainer import ImageClassifierTrainer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a table view
        self.table_view = QTableView(self)
        # self.setFixedSize(800, 600)
        self.setWindowFlags(
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
        )

        # Create a data model for the table
        self.model = QStandardItemModel(0, 2, self)
        self.model.setHorizontalHeaderLabels(["File Name", "Ripeness"])
        self.table_view.setModel(self.model)

        # Create an image preview widget
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.show_image("placeholder.jpg")

        # Create two buttons
        self.button1 = QPushButton("Add Images", self)
        self.button2 = QPushButton("Train Models", self)
        self.button1.clicked.connect(self.add_images)
        self.button2.clicked.connect(self.train_model)

        # Create a horizontal layout to hold the image and buttons
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.button1)
        image_layout.addWidget(self.button2)

        # Create a horizontal layout to hold the table and image layouts
        table_widget = QWidget(self)
        layout = QHBoxLayout(table_widget)
        layout.addWidget(self.table_view)
        layout.addLayout(image_layout)

        # Set the central widget of the main window
        self.setCentralWidget(table_widget)

        # Connect the clicked signal of the table view to a custom slot
        self.table_view.clicked.connect(self.handle_table_click)

        self.text_edit = QPlainTextEdit()
        # Redirect terminal output to the log
        sys.stdout = self
        # Initialize the output buffer
        self.buffer = ""
        image_layout.addWidget(self.text_edit)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def show_context_menu(self, position):
        context_menu = QMenu(self)
        action1 = context_menu.addAction("Delete")

        action1.triggered.connect(self.action1_triggered)

        context_menu.exec(self.table_view.viewport().mapToGlobal(position))

    def action1_triggered(self):
        index = self.table_view.currentIndex()
        if index.isValid():
            self.table_view.model().removeRow(index.row())

    def write(self, message):
        # Append the message to the output buffer
        self.buffer += message

        # If a newline character is encountered, flush the buffer to the log
        if "\n" in message:
            lines = self.buffer.split("\n")
            for line in lines[:-1]:
                # Write each line to the log
                line = re.sub(r"[^\x20-\x7E]+", "", line)
                self.text_edit.moveCursor(QTextCursor.End)
                self.text_edit.insertPlainText(line + "\n")
            # Clear the buffer
            self.buffer = lines[-1]

    def flush(self):
        pass

    def show_image(self, filename):
        pixmap = QPixmap(filename)
        pixmap = pixmap.scaledToHeight(200)
        self.image_label.setPixmap(pixmap)

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open file", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)"
        )
        for i, file in enumerate(files):
            self.show_image(file)
            row = self.model.rowCount()
            self.model.insertRow(row)
            combo_box = QComboBox(self)
            combo_box.addItems(["Ripe", "Unripe", "Semi Ripe", "Overripe"])
            for column in range(1):
                self.model.setItem(row, 0, QStandardItem(file))
                self.table_view.setIndexWidget(self.model.index(row, 1), combo_box)

    def handle_table_click(self, index):
        # Get the selected row and column index
        index = self.table_view.selectedIndexes()[0]
        row = index.row()
        self.show_image(self.model.item(row, 0).text())

    def train_model(self):
        num_rows = self.model.rowCount()

        # Create the folders if they don't exist
        for folder in ["Ripe", "Unripe", "Semi Ripe", "Overripe"]:
            if not os.path.exists(f"../dataset/train/{folder}"):
                print(f"Creating {folder} folder for training")
                os.makedirs(f"../dataset/train/{folder}")
            if not os.path.exists(f"../dataset/validation/{folder}"):
                print(f"Creating {folder} folder for validation")
                os.makedirs(f"../dataset/validation/{folder}")

        for row in range(num_rows):
            # Get the combo box widget for the current row
            combo_box = self.table_view.indexWidget(self.model.index(row, 1))
            # Get the selected item from the combo box
            item_data = combo_box.currentText()
            print(f"{row}: {item_data} {self.model.item(row, 0).text()}")
            # get the file extension
            file_extension = self.model.item(row, 0).text().split(".")[-1]
            # Transfer the image to the appropriate folder
            shutil.copy(
                self.model.item(row, 0).text(),
                f"../dataset/train/{item_data}/image_{row}.{file_extension}",
            )
            print(f"copying to ../dataset/train/{item_data}/image_{row}.{file_extension}")
        # Train the model
        trainer = ImageClassifierTrainer("VGG19")
        trainer.train(10)
        trainer.save_model("model.h5")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
