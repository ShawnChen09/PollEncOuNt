import os
import sys
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


class TextRedirector:
    """
    Replaces sys.stdout so that print statements get written
    directly into the 'log_text' ScrolledText widget.
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

def train_main():

    def browse_data_path():
        path = filedialog.askopenfilename(
            filetypes=[("Data File", "*.yaml *.YAML")]
        )
        if path:
            print(f"Select Data YAML File to: {path}\n")
            data_path_var.set(path)

    def browse_save_dir():
        path = filedialog.askdirectory()
        if path:
            print(f"Select Project Save Directory to: {path}\n")
            save_dir_var.set(path)

    def browse_pretrained_model():
        path = filedialog.askopenfilename(
            filetypes=[("Model Files", "*.pt *.pth *.onnx")]
        )
        if path:
            print(f"Select Model to: {path}\n")
            pretrained_model_var.set(path)

            yolo_model_var.set("Select a model")

    def on_select_yolo_model(choice):
        if choice:
            print(f"Select Model to: {choice}\n")
            yolo_model_var.set(choice)
            pretrained_model_var.set("")

    def on_start_training():
        answer = messagebox.askyesno(
            "Start Training?",
            "Are you sure you want to start training?"
        )
        if not answer:
            return  # User chose 'No'

        data_path = data_path_var.get()
        save_dir = save_dir_var.get()
        epochs = int(epochs_var.get())
        device = device_var.get()

        all_valid = True

        if not os.path.exists(data_path):
            print(f"Error: Please select a valid Data YAML Path. Your input: '{data_path}'\n")
            all_valid = False

        if not save_dir:
            print(f"Error: Please select a valid Project Save Directory. Your input: '{save_dir}'\n")
            all_valid = False

        if pretrained_model_var.get():
            model_path = pretrained_model_var.get()
            if not os.path.exists(model_path):
                print(f"Error: Please select a valid Pre-trained Model path. Your input: '{model_path}'\n")
                all_valid = False
        else:
            if yolo_model_var.get() == "Select a model":
                print(f"Error: Please select one type of model (pre-trained or YOLO)\n")
                all_valid = False
            model_path = yolo_model_var.get()

        if epochs < 1:
            print(f"Error: Please select a Epochs > 0. Your input: {epochs}\n")
            all_valid = False
        
        if not all_valid:
            return

        start_training_button.config(state=tk.DISABLED)

        original_stdout = sys.stdout
        sys.stdout = TextRedirector(log_text)

        original_stderr = sys.stderr
        sys.stderr = TextRedirector(log_text)

        def training_thread():
            from peon import peon_train
            peon_train(
                data_path=data_path,
                save_dir=save_dir,
                model_path=model_path,
                epochs=epochs,
                device=device,
            )

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            start_training_button.config(state=tk.NORMAL)

        t = threading.Thread(target=training_thread)
        t.start()

    def on_reset():
        start_training_button.config(state=tk.NORMAL)
        # Clear the log
        log_text.delete("1.0", tk.END)
        # Reset variables
        data_path_var.set("")
        save_dir_var.set("")
        pretrained_model_var.set("")
        yolo_model_var.set(yolo_models[0])
        epochs_var.set("100")
        device_var.set("cpu")

    root = tk.Tk()
    root.title("PEON Train GUI")

    style = ttk.Style(root)
    style.theme_use("alt")

    data_path_var = tk.StringVar()
    save_dir_var = tk.StringVar()
    pretrained_model_var = tk.StringVar(value="")
    yolo_models = [
        "Select a model",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    ]
    yolo_model_var = tk.StringVar(value=yolo_models[0])
    epochs_var = tk.StringVar(value="100")
    device_var = tk.StringVar(value="cpu")

    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ========== DATA YAML PATH ==========
    ttk.Label(
        main_frame,
        text="Data YAML File",
        font=("TkDefaultFont", 16)
    ).grid(
        row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 5)
    )
    ttk.Entry(main_frame, textvariable=data_path_var, width=47).grid(
        row=1, column=0, padx=5, pady=5, columnspan=2, sticky="w"
    )
    ttk.Button(main_frame, text="Browse", command=browse_data_path).grid(
        row=1, column=2, padx=5, pady=5, sticky="e"
    )

    # ========== SAVE DIRECTORY ==========
    ttk.Label(
        main_frame,
        text="Project Save Directory",
        font=("TkDefaultFont", 16)
    ).grid(
        row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5)
    )
    ttk.Entry(main_frame, textvariable=save_dir_var, width=47).grid(
        row=3, column=0, padx=5, pady=5, columnspan=2, sticky="w"
    )
    ttk.Button(main_frame, text="Browse", command=browse_save_dir).grid(
        row=3, column=2, padx=5, pady=5, sticky="e"
    )

    # ========== MODEL SELECTION TITLE ==========
    ttk.Label(
        main_frame, 
        text="Model selection (pre-trained or YOLO)", 
        font=("TkDefaultFont", 16)
    ).grid(
        row=4, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5)
    )

    # ========== MODEL FRAME ==========
    model_frame = ttk.Frame(main_frame)
    model_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    # ---- Pre-trained model (left side) ----
    pretrained_frame = ttk.Frame(model_frame, borderwidth=1, relief="groove", padding=5)
    pretrained_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

    ttk.Label(pretrained_frame, text="Pre-trained").grid(
        row=0, column=0, sticky="e", padx=5, pady=2
    )
    ttk.Entry(pretrained_frame, textvariable=pretrained_model_var, width=10).grid(
        row=0, column=1, padx=5, pady=2
    )
    ttk.Button(pretrained_frame, text="Browse", command=browse_pretrained_model).grid(
        row=0, column=2, padx=5, pady=2
    )

    # ---- YOLOv8 model (right side) ----
    yolo_frame = ttk.Frame(model_frame, borderwidth=1, relief="groove", padding=5)
    yolo_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    ttk.Label(yolo_frame, text="YOLO").grid(
        row=0, column=0, sticky="e", padx=5, pady=2
    )
    yolo_model_optionmenu = ttk.OptionMenu(
        yolo_frame,
        yolo_model_var,
        yolo_models[0],
        *yolo_models,
        command=on_select_yolo_model
    )
    yolo_model_optionmenu.config(width=10)
    yolo_model_optionmenu.grid(row=0, column=1, padx=5, pady=2)

    # ========== TRAIN SETTINGS TITLE ==========
    train_title_frame = ttk.Frame(main_frame)
    train_title_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 5))

    for i in range(2):
        train_title_frame.columnconfigure(i, weight=1)

    ttk.Label(
        train_title_frame,
        text="Train Settings",
        font=("TkDefaultFont", 16)
    ).grid(row=0, column=0, sticky="w", padx=5)

    ttk.Label(
        train_title_frame,
        text="Start and Reset",
        font=("TkDefaultFont", 16)
    ).grid(row=0, column=1, sticky="e", padx=130)

    # ========== TRAIN SETTINGS FRAME ==========
    train_settings_frame = ttk.Frame(main_frame, borderwidth=1, relief="groove", padding=5)
    train_settings_frame.grid(row=7, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

    for i in range(5):
        train_settings_frame.columnconfigure(i, weight=1)

    # --- Epochs ---
    ttk.Label(train_settings_frame, text="Epochs").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    try:
        epochs_spinbox = ttk.Spinbox(
            train_settings_frame,
            from_=1,
            to=100000,
            textvariable=epochs_var,
            width=6
        )
        epochs_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    except AttributeError:
        tk.Spinbox(
            train_settings_frame,
            from_=1,
            to=100000,
            textvariable=epochs_var,
            width=6
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # --- Device ---
    ttk.Label(train_settings_frame, text="Device").grid(row=0, column=2, sticky="e", padx=5, pady=5)
    device_combo = ttk.Combobox(
        train_settings_frame,
        textvariable=device_var,
        values=["cpu", "gpu"],
        state="readonly",
        width=5
    )
    device_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")

    # --- Start Training Button ---
    start_training_button = ttk.Button(train_settings_frame, text="Start Training", command=on_start_training)
    start_training_button.grid(row=0, column=4, padx=5, pady=5)

    # --- Reset Button ---
    reset_button = ttk.Button(train_settings_frame, text="Reset", command=on_reset)
    reset_button.grid(row=0, column=5, padx=5, pady=5)

    # ========== LOG FRAME ==========
    log_frame = ttk.LabelFrame(main_frame, text="LOGS", padding=5)
    log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

    log_text = scrolledtext.ScrolledText(log_frame, wrap="word", width=60, height=30)
    log_text.pack(fill="both", expand=True)
    sys.stdout = TextRedirector(log_text)
    sys.stderr = TextRedirector(log_text)

    root.mainloop()

def predict_main():
    def browse_img_files():
        paths = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")]
        )
        if paths:
            # Convert the tuple/list of paths into a newline-separated string for display
            selected_files = "\n".join(paths)
            print(f"Selected Image Files:\n{selected_files}\n")
            img_files_var.set(selected_files)

    def browse_model_path():
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pt *.pth *.onnx")]
        )
        if path:
            print(f"Selected Model File: {path}\n")
            model_path_var.set(path)

    def browse_save_dir():
        path = filedialog.askdirectory(title="Select Save Directory")
        if path:
            print(f"Selected Save Directory: {path}\n")
            save_dir_var.set(path)

    def on_start_prediction():
        answer = messagebox.askyesno(
            "Start Prediction?",
            "Are you sure you want to start the prediction process?"
        )
        if not answer:
            return  # User chose 'No'

        # Retrieve user inputs
        raw_img_files = img_files_var.get().strip()
        model_path = model_path_var.get().strip()
        save_dir = save_dir_var.get().strip()
        save_img = save_img_var.get()
        save_csv = save_csv_var.get()

        # Validation checks
        all_valid = True
        if not raw_img_files:
            print("Error: Please select one or more image files.\n")
            all_valid = False
        # Convert newline-separated list of files into a Python list
        img_files_list = [f for f in raw_img_files.splitlines() if f.strip()]
        # Check existence of at least the first file (basic check)
        if not os.path.exists(img_files_list[0]):
            print(f"Error: Please select valid image file(s). First invalid: '{img_files_list[0]}'\n")
            all_valid = False

        if not model_path or not os.path.exists(model_path):
            print(f"Error: Please select a valid Model path. Your input: '{model_path}'\n")
            all_valid = False

        if not save_dir:
            print(f"Error: Please select a valid Save Directory. Your input: '{save_dir}'\n")
            all_valid = False

        if not all_valid:
            return

        # Disable the button to prevent multiple concurrent predictions
        start_predict_button.config(state=tk.DISABLED)

        original_stdout = sys.stdout
        sys.stdout = TextRedirector(log_text)

        original_stderr = sys.stderr
        sys.stderr = TextRedirector(log_text)

        def prediction_thread():
            from peon import peon_predict  # Import here so it doesn't block GUI creation

            peon_predict(
                img_files=img_files_list,
                model_path=model_path,
                save_dir=save_dir,
                save_img=save_img,
                save_csv=save_csv
            )

            # Restore original stdout/stderr when done
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            start_predict_button.config(state=tk.NORMAL)

        t = threading.Thread(target=prediction_thread)
        t.start()

    def on_reset():
        start_predict_button.config(state=tk.NORMAL)
        # Clear the log
        log_text.delete("1.0", tk.END)
        # Reset variables
        img_files_var.set("")
        model_path_var.set("")
        save_dir_var.set("")
        save_img_var.set(True)
        save_csv_var.set(True)

    # ========== Main Window ==========
    root = tk.Tk()
    root.title("PEON Predict GUI")

    style = ttk.Style(root)
    style.theme_use("alt")  # Or pick any available style on your system

    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ========== Variables ==========
    img_files_var = tk.StringVar(value="")
    model_path_var = tk.StringVar(value="")
    save_dir_var = tk.StringVar(value="")
    save_img_var = tk.BooleanVar(value=True)
    save_csv_var = tk.BooleanVar(value=True)

    # ========== IMAGE FILES ==========
    ttk.Label(
        main_frame,
        text="Image Files",
        font=("TkDefaultFont", 16)
    ).grid(
        row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 5)
    )
    ttk.Entry(main_frame, textvariable=img_files_var, width=47).grid(
        row=1, column=0, padx=5, pady=5, columnspan=2, sticky="w"
    )
    ttk.Button(main_frame, text="Browse", command=browse_img_files).grid(
        row=1, column=2, padx=5, pady=5, sticky="e"
    )

    # ========== MODEL PATH ==========
    ttk.Label(
        main_frame,
        text="Model File",
        font=("TkDefaultFont", 16)
    ).grid(
        row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5)
    )
    ttk.Entry(main_frame, textvariable=model_path_var, width=47).grid(
        row=3, column=0, padx=5, pady=5, columnspan=2, sticky="w"
    )
    ttk.Button(main_frame, text="Browse", command=browse_model_path).grid(
        row=3, column=2, padx=5, pady=5, sticky="e"
    )

    # ========== SAVE DIRECTORY ==========
    ttk.Label(
        main_frame,
        text="Project Save Directory",
        font=("TkDefaultFont", 16)
    ).grid(
        row=4, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5)
    )
    ttk.Entry(main_frame, textvariable=save_dir_var, width=47).grid(
        row=5, column=0, padx=5, pady=5, columnspan=2, sticky="w"
    )
    ttk.Button(main_frame, text="Browse", command=browse_save_dir).grid(
        row=5, column=2, padx=5, pady=5, sticky="e"
    )

    # ========== PREDICT SETTINGS TITLE ==========
    predict_title_frame = ttk.Frame(main_frame)
    predict_title_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 5))
    # for i in range(2):
    #     predict_title_frame.columnconfigure(i, weight=1)

    ttk.Label(
        predict_title_frame,
        text="Predict Settings",
        font=("TkDefaultFont", 16)
    ).grid(row=0, column=0, sticky="w", padx=5)

    ttk.Label(
        predict_title_frame,
        text="Start and Reset",
        font=("TkDefaultFont", 16)
    ).grid(row=0, column=1, sticky="e", padx=130)

    # ========== PREDICT SETTINGS FRAME ==========
    predict_settings_frame = ttk.Frame(main_frame, borderwidth=1, relief="groove", padding=5)
    predict_settings_frame.grid(row=7, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    for i in range(5):
        predict_settings_frame.columnconfigure(i, weight=1)

    # Checkbuttons for saving images & CSV
    ttk.Checkbutton(
        predict_settings_frame,
        text="Save Images",
        variable=save_img_var
    ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

    ttk.Checkbutton(
        predict_settings_frame,
        text="Save CSV",
        variable=save_csv_var
    ).grid(row=0, column=1, padx=5, pady=5, sticky="w")

    # Start and Reset buttons
    start_predict_button = ttk.Button(predict_settings_frame, text="Start Prediction", command=on_start_prediction)
    start_predict_button.grid(row=0, column=2, padx=5, pady=5)

    reset_button = ttk.Button(predict_settings_frame, text="Reset", command=on_reset)
    reset_button.grid(row=0, column=3, padx=5, pady=5)

    # ========== LOG FRAME ==========
    log_frame = ttk.LabelFrame(main_frame, text="LOGS", padding=5)
    log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

    log_text = scrolledtext.ScrolledText(log_frame, wrap="word", width=70, height=20)
    log_text.pack(fill="both", expand=True)

    # Redirect stdout and stderr to the log
    sys.stdout = TextRedirector(log_text)
    sys.stderr = TextRedirector(log_text)

    root.mainloop()

def main():
    root = tk.Tk()
    root.title("PEON Main Menu")

    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # Create a menu "Actions" with Train, Predict, and Exit
    menu_actions = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label="Actions", menu=menu_actions)

    # Commands for each menu item
    def open_train():
        root.destroy()
        train_main()

    def open_predict():
        root.destroy()
        predict_main()

    def on_exit():
        sys.exit(0)

    menu_actions.add_command(label="Train", command=open_train)
    menu_actions.add_command(label="Predict", command=open_predict)
    menu_actions.add_separator()
    menu_actions.add_command(label="Exit", command=on_exit)

    # A simple label in the main window
    info_label = ttk.Label(root, text="Select an option from the 'Actions' menu above.")
    info_label.pack(padx=20, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()