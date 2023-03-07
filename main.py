import sys
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
    QRadioButton,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QTextCursor
from prediction import VGG19CoffeeClassifier


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
        self.model.setHorizontalHeaderLabels(["File Name", "Ripeness", "Accuracy"])
        self.table_view.setModel(self.model)

        self.pre_trained = True
        self.model_path = None

        # Create an image preview widget
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.show_image("placeholder.jpg")

        # Create two buttons
        self.button1 = QPushButton("Add Images", self)
        self.button2 = QPushButton("Predict Images", self)
        self.button1.clicked.connect(self.add_images)
        self.button2.clicked.connect(self.predict_images)

        # Create radio buttons
        self.radio_button_1 = QRadioButton("Pretrained Models")
        self.radio_button_2 = QRadioButton("Custom Models")
        self.radio_button_1.toggled.connect(self.radio_button_toggled)
        self.radio_button_2.toggled.connect(self.radio_button_toggled)

        # Create a horizontal layout to hold the image and buttons
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.button1)
        image_layout.addWidget(self.button2)
        image_layout.addWidget(self.radio_button_1)
        image_layout.addWidget(self.radio_button_2)

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

    def radio_button_toggled(self):
        if self.radio_button_2.isChecked():
            if self.model_path is None:
                self.add_model()
                self.pre_trained = False

        elif self.radio_button_1.isChecked():
            self.pre_trained = True
            print("Using pretrained")
        else:
            print("No option selected")

    def show_image(self, filename):
        pixmap = QPixmap(filename)
        pixmap = pixmap.scaledToHeight(200)
        self.image_label.setPixmap(pixmap)

    def add_model(self):
        files, _ = QFileDialog.getOpenFileName(
            self, "Open file", "", "Custom Model (*.h5)"
        )

        self.model_path = files

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open file", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)"
        )
        for i, file in enumerate(files):
            self.show_image(file)
            row = self.model.rowCount()
            self.model.insertRow(row)
            for column in range(1):
                self.model.setItem(row, 0, QStandardItem(file))

    def handle_table_click(self, index):
        # Get the selected row and column index
        index = self.table_view.selectedIndexes()[0]
        row = index.row()
        self.show_image(self.model.item(row, 0).text())

    def predict_images(self):
        vgg_coffee = VGG19CoffeeClassifier(
            pretrained=self.pre_trained, model_path=self.model_path
        )
        for i in range(self.model.rowCount()):
            file = self.model.item(i, 0).text()
            print(vgg_coffee.classify(file))
            data = vgg_coffee.classify(file)
            self.model.setItem(i, 1, QStandardItem(data['label']))
            self.model.setItem(i, 2, QStandardItem(str(data['accuracy'])))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
