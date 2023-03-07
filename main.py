import tkinter as tk
from tkinter import filedialog
from tkinter import Tk
from tkinter import ttk
from PIL import Image, ImageTk


class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid(sticky=tk.N + tk.S + tk.E + tk.W)
        self.create_widgets()

        # configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def create_widgets(self):
        self.table = ttk.Treeview(root, columns=("col1", "col2", "col3"))
        self.table.heading("#0", text="Index")
        self.table.heading("col1", text="File Name")
        self.table.heading("col2", text="Ripeness")
        self.table.heading("col3", text="Accuracy")
        self.table.grid(
            row=0,
            column=2,
            padx=10,
            pady=10,
            rowspan=3,
            sticky=tk.N + tk.S + tk.E + tk.W,
        )

        # Add a selection event handler for the table rows
        self.table.bind("<ButtonRelease-1>", self.handle_row_selection)

        self.preview_frame = tk.Frame(self, width=400, height=400)
        self.preview_frame.grid(
            row=0,
            column=1,
            padx=10,
            pady=10,
            rowspan=3,
            sticky=tk.N + tk.S + tk.E + tk.W,
        )
        self.preview_frame.grid_propagate(0)

        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.add_button = tk.Button(
            self, text="Add Images", command=self.add_images
        )
        self.add_button.grid(row=3, column=3, padx=10, pady=10)

        self.upload_button = tk.Button(
            self, text="Predict Images", command=self.upload_images
        )

        # Center the upload button
        self.upload_button.grid(row=3, column=2, padx=11, pady=10)

    def add_images(self):
        files = filedialog.askopenfilenames(
            initialdir="./",
            title="Select Images",
            filetypes=[
                ("JPEG Files", "*.jpg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*"),
            ],
        )
        for i, file in enumerate(files):
            self.table.insert(
                "",
                "end",
                text=str(i),
                values=(str(file), "Riped", "0.9"),
            )
            self.show_image(file)

    def show_image(self, filename):
        image = Image.open(filename)
        w, h = image.size
        aspect_ratio = w / h
        max_size = 400

        if aspect_ratio > 1:
            w = max_size
            h = int(w / aspect_ratio)
        else:
            h = max_size
            w = int(h * aspect_ratio)

        image = image.resize((w, h))
        photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def handle_row_selection(self, event):
        # Get the ID of the selected row
        selection = self.table.selection()
        if len(selection) == 0:
            return
        row_id = selection[0]

        # Get the data for the selected row
        row_data = self.table.item(row_id)["values"]
        self.show_image(row_data[0])

    def upload_images(self):
        for i in range(self.filelist.size()):
            filename = self.filelist.get(i)
            # do something with the image file here
            print(f"Uploading {filename}...")


root = Tk()
root.resizable(False, False)
app = App(root)
app.mainloop()
