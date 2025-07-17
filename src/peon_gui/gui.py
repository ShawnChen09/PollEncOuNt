import os
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

label_font = ("Arial", 16)


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
        path = filedialog.askopenfilename(filetypes=[("Data File", "*.yaml *.YAML")])
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
            model_var.set(path)
            yolo_model_var.set("Select a model")

    def on_select_yolo_model(choice):
        if choice and choice != "Select a model":
            print(f"Select Model to: {choice}\n")
            yolo_model_var.set(choice)
            model_var.set(choice)  # Show selected model in entry

    def create_add_param_callback(button_to_destroy):
        def add_custom_param():
            param_window = tk.Toplevel()
            param_window.title("Add Parameter")
            param_window.geometry("400x200")
            param_window.resizable(True, True)
            param_window.grab_set()  # Make it modal

            # Center the window
            param_window.update_idletasks()
            width = param_window.winfo_width()
            height = param_window.winfo_height()
            x = (param_window.winfo_screenwidth() // 2) - (width // 2)
            y = (param_window.winfo_screenheight() // 2) - (height // 2)
            param_window.geometry(f"{width}x{height}+{x}+{y}")

            # Configure grid to be resizable
            param_window.columnconfigure(0, weight=1)
            param_window.columnconfigure(1, weight=3)
            param_window.rowconfigure(3, weight=1)

            # Valid YOLO train arguments
            valid_args = [
                "batch",
                "imgsz",
                "save",
                "save_period",
                "cache",
                "workers",
                "name",
                "exist_ok",
                "pretrained",
                "optimizer",
                "seed",
                "deterministic",
                "single_cls",
                "classes",
                "rect",
                "multi_scale",
                "cos_lr",
                "close_mosaic",
                "resume",
                "amp",
                "fraction",
                "profile",
                "freeze",
                "lr0",
                "lrf",
                "momentum",
                "weight_decay",
                "warmup_epochs",
                "warmup_momentum",
                "warmup_bias_lr",
                "box",
                "cls",
                "dfl",
                "pose",
                "kobj",
                "nbs",
                "overlap_mask",
                "mask_ratio",
                "dropout",
                "val",
                "plots",
                "patience",
                "time",
            ]

            # Variables for the new parameter
            name_var = tk.StringVar()
            value_var = tk.StringVar()
            value_type = tk.StringVar(value="string")

            # Create form elements
            ttk.Label(param_window, text="Parameter Name:").grid(
                row=0, column=0, padx=10, pady=5, sticky="w"
            )
            name_entry = ttk.Combobox(
                param_window, textvariable=name_var, values=valid_args, width=20
            )

            name_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

            ttk.Label(param_window, text="Parameter Value:").grid(
                row=1, column=0, padx=10, pady=5, sticky="w"
            )
            ttk.Entry(param_window, textvariable=value_var, width=20).grid(
                row=1, column=1, padx=10, pady=5, sticky="ew"
            )

            ttk.Label(param_window, text="Value Type:").grid(
                row=2, column=0, padx=10, pady=5, sticky="w"
            )
            type_combo = ttk.Combobox(
                param_window, textvariable=value_type, state="readonly", width=20
            )
            type_combo["values"] = [
                "string",
                "integer",
                "float",
                "boolean",
                "list[int]",
            ]
            type_combo.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

            # Parameter description area
            desc_frame = ttk.LabelFrame(param_window, text="Parameter Description")
            desc_frame.grid(
                row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew"
            )
            desc_frame.columnconfigure(0, weight=1)
            desc_frame.rowconfigure(0, weight=1)

            desc_text = tk.Text(
                desc_frame, wrap="word", height=2, width=30, state="disabled"
            )
            desc_text.pack(fill="both", expand=True, padx=5, pady=5)

            # Parameter descriptions dictionary
            param_descriptions = {
                "batch": "Batch size for training (auto-scales for GPU memory)",
                "imgsz": "Image size for training as integer or (h, w)",
                "save": "Save training checkpoints and final model",
                "save_period": "Save checkpoint every x epochs (-1 disables)",
                "cache": "Cache images in memory or on disk for faster training",
                "workers": "Number of worker threads for data loading",
                "name": "Run name within project for saving outputs",
                "exist_ok": "Allow overwriting existing project/name",
                "pretrained": "Start training from pretrained model",
                "optimizer": "Optimizer: SGD, Adam, AdamW, etc.",
                "seed": "Random seed for reproducibility",
                "deterministic": "Force deterministic algorithms",
                "single_cls": "Treat all classes as single class",
                "classes": "Filter by class IDs for training",
                "rect": "Use rectangular training with minimal padding",
                "multi_scale": "Vary img-size +/- 50% during training",
                "cos_lr": "Use cosine learning rate scheduler",
                "close_mosaic": "Disable mosaic in last N epochs",
                "resume": "Resume training from last checkpoint",
                "amp": "Use Automatic Mixed Precision",
                "fraction": "Dataset fraction to train on (0.1 = 10%)",
                "profile": "Profile ONNX and TensorRT speeds during training",
                "freeze": "Freeze first N layers or list of layer indices",
                "lr0": "Initial learning rate",
                "lrf": "Final learning rate fraction of lr0",
                "momentum": "SGD momentum/Adam beta1",
                "weight_decay": "Optimizer weight decay",
                "warmup_epochs": "Epochs to warmup LR from 0 to lr0",
                "warmup_momentum": "Initial momentum",
                "warmup_bias_lr": "Initial bias learning rate",
                "box": "Box loss gain",
                "cls": "Cls loss gain",
                "dfl": "Distribution focal loss gain",
                "pose": "Pose loss gain",
                "kobj": "Keypoint objectness loss gain",
                "nbs": "Nominal batch size",
                "overlap_mask": "Merge masks by overlapping",
                "mask_ratio": "Mask downsample ratio",
                "dropout": "Dropout rate for classification layers",
                "val": "Validate during training",
                "plots": "Generate plots during training",
                "patience": "Early stopping patience (epochs without improvement)",
                "time": "Maximum training time in hours",
            }

            # Update description when parameter name changes
            def update_description(*args):
                param_name = name_var.get()
                desc_text.config(state="normal")
                desc_text.delete(1.0, tk.END)
                if param_name in param_descriptions:
                    desc_text.insert(tk.END, param_descriptions[param_name])

                    # Set appropriate default type based on parameter
                    if param_name in [
                        "lr0",
                        "lrf",
                        "momentum",
                        "weight_decay",
                        "warmup_epochs",
                        "warmup_momentum",
                        "warmup_bias_lr",
                        "box",
                        "cls",
                        "dfl",
                        "pose",
                        "kobj",
                        "dropout",
                        "time",
                        "fraction",
                    ]:
                        value_type.set("float")
                    elif param_name in [
                        "batch",
                        "imgsz",
                        "save_period",
                        "workers",
                        "seed",
                        "close_mosaic",
                        "nbs",
                        "mask_ratio",
                        "patience",
                        "freeze",
                    ]:
                        value_type.set("integer")
                    elif param_name in ["classes"]:
                        value_type.set("list[int]")
                    elif param_name in [
                        "save",
                        "cache",
                        "exist_ok",
                        "pretrained",
                        "deterministic",
                        "single_cls",
                        "rect",
                        "multi_scale",
                        "cos_lr",
                        "resume",
                        "amp",
                        "profile",
                        "val",
                        "plots",
                        "overlap_mask",
                    ]:
                        value_type.set("boolean")
                    else:
                        value_type.set("string")
                desc_text.config(state="disabled")

            name_var.trace_add("write", update_description)

            # Button frame
            button_frame = ttk.Frame(param_window)
            button_frame.grid(row=4, column=0, columnspan=2, pady=10)

            def on_cancel():
                param_window.destroy()

            def on_confirm():
                param_name = name_var.get().strip()
                param_value = value_var.get().strip()
                param_type = value_type.get()

                # Check for empty fields
                if not param_name or not param_value:
                    messagebox.showerror(
                        "Error", "Parameter name and value cannot be empty"
                    )
                    return

                # Validate parameter name is a valid YOLO train argument
                if param_name not in valid_args:
                    messagebox.showerror(
                        "Error",
                        f"'{param_name}' is not a valid YOLO parameter. Please select from the dropdown.",
                    )
                    return

                # Calculate the position for the new parameter
                row_idx = (
                    len(custom_params) + 1
                ) // 4  # Epochs is already in position 0,0
                col_idx = (len(custom_params) + 1) % 4

                # Create a frame to hold the parameter
                param_frame = ttk.Frame(param_settings_frame)
                param_frame.grid(
                    row=row_idx, column=col_idx, padx=5, pady=5, sticky="w"
                )

                # Parameter name as label
                ttk.Label(param_frame, text=f"{param_name}:").pack(
                    side=tk.LEFT, padx=(0, 2)
                )

                # Parameter value as entry
                entry_var = tk.StringVar(value=param_value)
                ttk.Entry(param_frame, textvariable=entry_var, width=8).pack(
                    side=tk.LEFT
                )

                # Store parameter info with name, entry variable, and type
                custom_params.append((param_name, entry_var, param_type))

                param_window.destroy()

                # Remove the old button
                button_to_destroy.destroy()

                # Calculate the next position for the "Add Parameter" button
                next_row = row_idx
                next_col = (col_idx + 1) % 4
                if next_col == 0:  # If we need to move to the next row
                    next_row += 1

                # Create a new button at the next position
                new_button = ttk.Button(
                    param_settings_frame,
                    text="Add Parameter",
                    # Command will be set after button creation
                )
                new_button.grid(
                    row=next_row, column=next_col, padx=5, pady=5, sticky="w"
                )

                # Now set the command with the new button as the parameter
                new_button.config(command=create_add_param_callback(new_button))

            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
                side=tk.LEFT, padx=10
            )
            ttk.Button(button_frame, text="Confirm", command=on_confirm).pack(
                side=tk.LEFT, padx=10
            )

        return add_custom_param

    def on_start_training():
        answer = messagebox.askyesno(
            "Start Training?", "Are you sure you want to start training?"
        )
        if not answer:
            return  # User chose 'No'

        data_path = data_path_var.get()
        save_dir = save_dir_var.get()
        epochs = int(epochs_var.get())
        model_path = model_var.get()

        all_valid = True

        if not os.path.exists(data_path):
            print(
                f"Error: Please select a valid Data YAML Path. Your input: '{data_path}'\n"
            )
            all_valid = False

        if not save_dir:
            print(
                f"Error: Please select a valid Project Save Directory. Your input: '{save_dir}'\n"
            )
            all_valid = False

        if epochs < 1:
            print(f"Error: Please select a Epochs > 0. Your input: {epochs}\n")
            all_valid = False

        if not model_path or (
            not os.path.exists(model_path) and model_path not in yolo_models[1:]
        ):
            print(f"Error: Please select a valid model. Your input: '{model_path}'\n")
            all_valid = False

        # Prepare kwargs with custom parameters
        kwargs = {}
        for param_name, value_var, param_type in custom_params:
            param_value = value_var.get()
            if param_value:
                # Convert based on specified type
                if param_type == "integer":
                    try:
                        param_value = int(param_value)
                    except ValueError:
                        print(
                            f"Warning: Parameter '{param_name}' should be an integer. Using as string."
                        )
                elif param_type == "float":
                    try:
                        param_value = float(param_value)
                    except ValueError:
                        print(
                            f"Warning: Parameter '{param_name}' should be a float. Using as string."
                        )
                elif param_type == "boolean":
                    param_value = param_value.lower() in ("true", "yes", "t", "y", "1")
                elif param_type == "list[int]":
                    try:
                        # Convert comma-separated or space-separated string to list of ints
                        param_value = [
                            int(x.strip())
                            for x in re.split(r"[,\s]+", param_value)
                            if x.strip()
                        ]
                    except ValueError:
                        print(
                            f"Warning: Parameter '{param_name}' should be a comma-separated list of integers. Using as string."
                        )

                kwargs[param_name] = param_value

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
            )

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            start_training_button.config(state=tk.NORMAL)

        t = threading.Thread(target=training_thread)
        t.start()

    def on_reset():
        if not start_training_button:
            return
        start_training_button.config(state=tk.NORMAL)
        # Clear the log
        log_text.delete("1.0", tk.END)
        # Reset variables
        data_path_var.set("")
        save_dir_var.set("")
        model_var.set("")
        yolo_model_var.set(yolo_models[0])
        epochs_var.set("100")

    root = tk.Tk()
    root.title("PEON Train GUI")
    root.geometry("600x820")
    root.minsize(600, 820)

    style = ttk.Style(root)
    style.theme_use("alt")

    data_path_var = tk.StringVar()
    save_dir_var = tk.StringVar()
    model_var = tk.StringVar(value="")
    yolo_models = [
        "Select a model",
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
        "yolov9t.pt",
        "yolov9s.pt",
        "yolov9m.pt",
        "yolov9c.pt",
        "yolov9e.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
    ]
    yolo_model_var = tk.StringVar(value=yolo_models[0])
    epochs_var = tk.StringVar(value="100")
    custom_params = []  # tuples of (name_var, value_var)

    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.columnconfigure(2, weight=1)
    root.rowconfigure(0, weight=1)

    # ========== DATA YAML PATH ==========
    ttk.Label(main_frame, text="Data YAML File", font=label_font).grid(
        row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 5)
    )

    data_frame = ttk.Frame(main_frame)
    data_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    data_frame.columnconfigure(0, weight=1)  # Make the frame expandable

    data_entry = ttk.Entry(data_frame, textvariable=data_path_var)
    data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    ttk.Button(data_frame, text="Browse", command=browse_data_path).pack(side=tk.RIGHT)

    # ========== SAVE DIRECTORY ==========
    ttk.Label(main_frame, text="Project Save Directory", font=label_font).grid(
        row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5)
    )

    save_frame = ttk.Frame(main_frame)
    save_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    save_frame.columnconfigure(0, weight=1)  # Make the frame expandable

    save_entry = ttk.Entry(save_frame, textvariable=save_dir_var)
    save_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    ttk.Button(save_frame, text="Browse", command=browse_save_dir).pack(side=tk.RIGHT)

    # ========== MODEL SELECTION TITLE ==========
    ttk.Label(
        main_frame,
        text="Model selection (Custom or YOLO pre-trained model)",
        font=label_font,
    ).grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5))

    # ========== COMBINED MODEL FRAME ==========
    model_frame = ttk.Frame(main_frame)
    model_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    model_frame.columnconfigure(0, weight=1)  # Make the entry expandable

    # Combined model entry and selection
    model_entry = ttk.Entry(model_frame, textvariable=model_var)
    model_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

    # Browse button
    browse_button = ttk.Button(
        model_frame, text="Browse", command=browse_pretrained_model
    )
    browse_button.grid(row=0, column=1, padx=(0, 5))

    # YOLO model dropdown
    yolo_model_optionmenu = ttk.OptionMenu(
        model_frame,
        yolo_model_var,
        yolo_models[0],
        *yolo_models,
        command=on_select_yolo_model,
    )
    yolo_model_optionmenu.config(width=15)
    yolo_model_optionmenu.grid(row=0, column=2)

    # ========== TRAIN SETTINGS TITLE ==========
    train_title_frame = ttk.Frame(main_frame)
    train_title_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 5))
    train_title_frame.columnconfigure(0, weight=1)

    ttk.Label(train_title_frame, text="Parameter Settings", font=label_font).grid(
        row=0, column=0, sticky="w", padx=5
    )

    # ========== TRAIN SETTINGS FRAME ==========
    param_settings_frame = ttk.Frame(
        main_frame, borderwidth=1, relief="groove", padding=5
    )
    param_settings_frame.grid(
        row=7, column=0, columnspan=3, sticky="ew", padx=5, pady=5
    )
    for i in range(4):  # Change from 2 to 4 columns for more params per row
        param_settings_frame.columnconfigure(i, weight=1)

    # --- Epochs ---
    epochs_frame = ttk.Frame(param_settings_frame)
    epochs_frame.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Label(epochs_frame, text="Epochs").pack(side=tk.LEFT, padx=5, pady=5)
    try:
        epochs_spinbox = ttk.Spinbox(
            epochs_frame, from_=1, to=100000, textvariable=epochs_var, width=6
        )
        epochs_spinbox.pack(side=tk.LEFT, padx=5, pady=5)
    except AttributeError:
        tk.Spinbox(
            epochs_frame, from_=1, to=100000, textvariable=epochs_var, width=6
        ).pack(side=tk.LEFT, padx=5, pady=5)

    # Initialize Add Parameter button
    add_param_button = ttk.Button(param_settings_frame, text="Add Parameter")
    add_param_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    add_param_button.config(command=create_add_param_callback(add_param_button))

    # ========== START AND RESET BUTTONS (SEPARATE ROW) ==========
    control_frame = ttk.Frame(main_frame, padding=5)
    control_frame.grid(row=8, column=0, columnspan=3, sticky="e", padx=5, pady=5)

    start_training_button = ttk.Button(
        control_frame, text="Start Training", command=on_start_training
    )
    start_training_button.pack(side=tk.LEFT, padx=(0, 10))

    reset_button = ttk.Button(control_frame, text="Reset", command=on_reset)
    reset_button.pack(side=tk.LEFT)

    # ========== LOG FRAME ==========
    log_frame = ttk.LabelFrame(main_frame, text="LOGS", padding=5)
    log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

    log_text = scrolledtext.ScrolledText(log_frame, wrap="word", width=70, height=20)
    log_text.pack(fill="both", expand=True)
    sys.stdout = TextRedirector(log_text)
    sys.stderr = TextRedirector(log_text)

    root.mainloop()


def predict_main():
    def browse_img_files():
        paths = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
        )
        if paths:
            # Convert the tuple/list of paths into a newline-separated string for display
            selected_files = "\n".join(paths)
            print(f"Selected Image Files:\n{selected_files}\n")
            img_files_var.set(selected_files)

    def browse_model_path():
        path = filedialog.askopenfilename(
            title="Select Model File", filetypes=[("Model Files", "*.pt *.pth *.onnx")]
        )
        if path:
            print(f"Selected Model File: {path}\n")
            model_path_var.set(path)

    def browse_save_dir():
        path = filedialog.askdirectory(title="Select Save Directory")
        if path:
            print(f"Selected Save Directory: {path}\n")
            save_dir_var.set(path)

    def create_add_param_callback(button_to_destroy):
        def add_custom_param():
            param_window = tk.Toplevel()
            param_window.title("Add Parameter")
            param_window.geometry("400x220")
            param_window.resizable(True, True)  # Changed to True to make resizable
            param_window.grab_set()  # Make it modal

            # Center the window
            param_window.update_idletasks()
            width = param_window.winfo_width()
            height = param_window.winfo_height()
            x = (param_window.winfo_screenwidth() // 2) - (width // 2)
            y = (param_window.winfo_screenheight() // 2) - (height // 2)
            param_window.geometry(f"{width}x{height}+{x}+{y}")

            # Configure grid to be resizable
            param_window.columnconfigure(0, weight=1)
            param_window.columnconfigure(1, weight=3)  # Make value column expand more
            param_window.rowconfigure(3, weight=1)

            # Valid YOLO predict arguments
            valid_args = [
                "rect",
                "half",
                "device",
                "batch",
                "vid_stride",
                "stream_buffer",
                "visualize",
                "augment",
                "agnostic_nms",
                "classes",
                "retina_masks",
                "embed",
                "stream",
                "verbose",
                "save_frames",
                "save_txt",
                "save_conf",
                "save_crop",
                "show_conf",
                "show_boxes",
                "line_width",
            ]

            # Variables for the new parameter
            name_var = tk.StringVar()
            value_var = tk.StringVar()
            value_type = tk.StringVar(value="string")

            # Create form elements
            ttk.Label(param_window, text="Parameter Name:").grid(
                row=0, column=0, padx=10, pady=5, sticky="w"
            )
            name_entry = ttk.Combobox(
                param_window, textvariable=name_var, values=valid_args, width=17
            )
            name_entry.grid(row=0, column=1, padx=10, pady=5)

            ttk.Label(param_window, text="Parameter Value:").grid(
                row=1, column=0, padx=10, pady=5, sticky="w"
            )
            ttk.Entry(param_window, textvariable=value_var, width=20).grid(
                row=1, column=1, padx=10, pady=5
            )

            ttk.Label(param_window, text="Value Type:").grid(
                row=2, column=0, padx=10, pady=5, sticky="w"
            )
            type_combo = ttk.Combobox(
                param_window, textvariable=value_type, state="readonly", width=17
            )
            type_combo["values"] = [
                "string",
                "integer",
                "float",
                "boolean",
                "list[int]",
            ]
            type_combo.grid(row=2, column=1, padx=10, pady=5)

            # Parameter description area
            desc_frame = ttk.LabelFrame(param_window, text="Parameter Description")
            desc_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

            desc_text = tk.Text(
                desc_frame, wrap="word", height=2, width=30, state="disabled"
            )
            desc_text.pack(fill="both", expand=True, padx=5, pady=5)

            # Parameter descriptions dictionary
            param_descriptions = {
                "rect": "Use rectangular inference if True",
                "half": "Use FP16 half-precision inference",
                "device": "Device to run inference on (cpu, cuda:0, etc)",
                "batch": "Batch size for processing",
                "vid_stride": "Video frame-rate stride",
                "stream_buffer": "Buffer all streaming frames if True",
                "visualize": "Visualize model features if True",
                "augment": "Apply augmentation to prediction if True",
                "agnostic_nms": "Class-agnostic NMS if True",
                "classes": "Filter by class index (e.g. 0, 2, 3)",
                "retina_masks": "High-resolution masks if True",
                "embed": "Layers to extract embeddings from",
                "stream": "Return frame-by-frame generator",
                "verbose": "Print verbose output",
                "save_frames": "Save video frames as images",
                "save_txt": "Save results as .txt file",
                "save_conf": "Save results with confidence scores",
                "save_crop": "Save cropped images with results",
                "show_conf": "Show confidence scores",
                "show_boxes": "Show bounding boxes",
                "line_width": "Line width for bounding boxes",
            }

            # Update description when parameter name changes
            def update_description(*args):
                param_name = name_var.get()
                desc_text.config(state="normal")
                desc_text.delete(1.0, tk.END)
                if param_name in param_descriptions:
                    desc_text.insert(tk.END, param_descriptions[param_name])

                    # Set appropriate default type based on parameter
                    if param_name in ["batch", "vid_stride", "line_width"]:
                        value_type.set("integer")
                    elif param_name in ["classes", "embed"]:
                        value_type.set("list[int]")
                    elif param_name in [
                        "rect",
                        "half",
                        "visualize",
                        "augment",
                        "agnostic_nms",
                        "retina_masks",
                        "stream",
                        "verbose",
                        "save_frames",
                        "save_txt",
                        "save_conf",
                        "save_crop",
                        "show_conf",
                        "show_boxes",
                        "stream_buffer",
                    ]:
                        value_type.set("boolean")
                    else:
                        value_type.set("string")
                desc_text.config(state="disabled")

            name_var.trace_add("write", update_description)

            # Button frame
            button_frame = ttk.Frame(param_window)
            button_frame.grid(row=4, column=0, columnspan=2, pady=10)

            def on_cancel():
                param_window.destroy()

            def on_confirm():
                param_name = name_var.get().strip()
                param_value = value_var.get().strip()
                param_type = value_type.get()

                if not param_name or not param_value:
                    messagebox.showerror(
                        "Error", "Parameter name and value cannot be empty"
                    )
                    return

                if param_name not in valid_args:
                    messagebox.showerror(
                        "Error",
                        f"'{param_name}' is not a valid YOLO parameter. Please select from the dropdown.",
                    )
                    return

                row_idx = (
                    1 + (len(custom_params) + 2) // 4
                )  # +2 because already have 2 params in row 1
                col_idx = (len(custom_params) + 2) % 4

                param_frame = ttk.Frame(param_settings_frame)
                param_frame.grid(
                    row=row_idx, column=col_idx, padx=5, pady=5, sticky="w"
                )

                ttk.Label(param_frame, text=f"{param_name}:").pack(
                    side=tk.LEFT, padx=(0, 2)
                )

                entry_var = tk.StringVar(value=param_value)
                ttk.Entry(param_frame, textvariable=entry_var, width=8).pack(
                    side=tk.LEFT
                )

                custom_params.append((param_name, entry_var, param_type))

                param_window.destroy()

                button_to_destroy.destroy()

                next_row = row_idx
                next_col = (col_idx + 1) % 4
                if next_col == 0:
                    next_row += 1

                new_button = ttk.Button(
                    param_settings_frame,
                    text="Add Parameter",
                )
                new_button.grid(
                    row=next_row, column=next_col, padx=5, pady=5, sticky="w"
                )

                new_button.config(command=create_add_param_callback(new_button))

            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
                side=tk.LEFT, padx=10
            )
            ttk.Button(button_frame, text="Confirm", command=on_confirm).pack(
                side=tk.LEFT, padx=10
            )

        return add_custom_param

    def on_start_prediction():
        answer = messagebox.askyesno(
            "Start Prediction?",
            "Are you sure you want to start the prediction process?",
        )
        if not answer:
            return  # User chose 'No'

        # Retrieve user inputs
        raw_img_files = img_files_var.get().strip()
        model_path = model_path_var.get().strip()
        save_dir = save_dir_var.get().strip()
        save_img = save_img_var.get()
        save_csv = save_csv_var.get()
        conf = float(conf_var.get())
        iou = float(iou_var.get())
        imgsz = int(imgsz_var.get())
        max_det = int(max_det_var.get())

        # Validation checks
        all_valid = True
        if not raw_img_files:
            print("Error: Please select one or more image files.\n")
            all_valid = False
        # Convert newline-separated list of files into a Python list
        img_files_list = [f for f in raw_img_files.splitlines() if f.strip()]
        # Check existence of at least the first file (basic check)
        if not os.path.exists(img_files_list[0]):
            print(
                f"Error: Please select valid image file(s). First invalid: '{img_files_list[0]}'\n"
            )
            all_valid = False

        if not model_path or not os.path.exists(model_path):
            print(
                f"Error: Please select a valid Model path. Your input: '{model_path}'\n"
            )
            all_valid = False

        if not save_dir:
            print(
                f"Error: Please select a valid Save Directory. Your input: '{save_dir}'\n"
            )
            all_valid = False

        if not 0 <= conf <= 1:
            print(
                f"Error: Please select a valid Confidence Threshold between 0 and 1. Your input: '{conf}'\n"
            )

        if not 0 < iou <= 1:
            print(
                f"Error: Please select a valid IoU Threshold between 0 and 1. Your input: '{iou}'\n"
            )

        # Collect custom parameters
        kwargs = {}
        for param_name, value_var, value_type in custom_params:
            param_value = value_var.get()
            if param_value:
                # Convert based on specified type
                if value_type == "integer":
                    try:
                        param_value = int(param_value)
                    except ValueError:
                        print(
                            f"Warning: Parameter '{param_name}' should be an integer. Using as string."
                        )
                elif value_type == "float":
                    try:
                        param_value = float(param_value)
                    except ValueError:
                        print(
                            f"Warning: Parameter '{param_name}' should be a float. Using as string."
                        )
                elif value_type == "boolean":
                    param_value = param_value.lower() in ("true", "yes", "t", "y", "1")
                elif value_type == "list[int]":
                    try:
                        param_value = [
                            int(x.strip())
                            for x in re.split(r"[,\s]+", param_value)
                            if x.strip()
                        ]
                    except ValueError:
                        print(
                            f"Warning: Parameter '{param_name}' should be a comma-separated list of integers. Using as string."
                        )

                kwargs[param_name] = param_value

        if not all_valid:
            return

        # Disable the button to prevent multiple concurrent predictions
        start_predict_button.config(state=tk.DISABLED)

        original_stdout = sys.stdout
        sys.stdout = TextRedirector(log_text)

        original_stderr = sys.stderr
        sys.stderr = TextRedirector(log_text)

        def prediction_thread():
            from peon import peon_predict

            peon_predict(
                img_files=img_files_list,
                model_path=model_path,
                save_dir=save_dir,
                save_img=save_img,
                save_csv=save_csv,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                **kwargs,
            )

            # Restore original stdout/stderr when done
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            start_predict_button.config(state=tk.NORMAL)

        t = threading.Thread(target=prediction_thread)
        t.start()

    def on_reset():
        if not start_predict_button:
            return
        start_predict_button.config(state=tk.NORMAL)
        # Clear the log
        log_text.delete("1.0", tk.END)
        # Reset variables
        img_files_var.set("")
        model_path_var.set("")
        save_dir_var.set("")
        save_img_var.set(True)
        save_csv_var.set(True)
        conf_var.set("0.0")
        iou_var.set("0.5")
        imgsz_var.set(1280)
        max_det_var.set(500)

    # ========== Main Window ==========
    root = tk.Tk()
    root.title("PEON Predict GUI")
    root.geometry("600x820")
    root.minsize(600, 820)

    style = ttk.Style(root)
    style.theme_use("alt")  # Or pick any available style on your system

    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.columnconfigure(2, weight=1)
    root.rowconfigure(0, weight=1)

    # ========== Variables ==========
    img_files_var = tk.StringVar(value="")
    model_path_var = tk.StringVar(value="")
    save_dir_var = tk.StringVar(value="")
    save_img_var = tk.BooleanVar(value=True)
    save_csv_var = tk.BooleanVar(value=True)
    conf_var = tk.StringVar(value=0.0)
    iou_var = tk.StringVar(value=0.5)
    imgsz_var = tk.StringVar(value=1280)
    max_det_var = tk.StringVar(value=500)
    custom_params = []  # tuples of (name_var, value_var)

    # ========== IMAGE FILES ==========
    ttk.Label(main_frame, text="Image File(s)", font=label_font).grid(
        row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 5)
    )

    img_frame = ttk.Frame(main_frame)
    img_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    img_frame.columnconfigure(0, weight=1)  # Make entry expandable

    img_entry = ttk.Entry(img_frame, textvariable=img_files_var)
    img_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

    ttk.Button(img_frame, text="Browse", command=browse_img_files).grid(
        row=0, column=1, sticky="e"
    )

    # ========== MODEL PATH ==========
    ttk.Label(main_frame, text="Model File", font=label_font).grid(
        row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5)
    )

    model_frame = ttk.Frame(main_frame)
    model_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    model_frame.columnconfigure(0, weight=1)  # Make entry expandable

    model_entry = ttk.Entry(model_frame, textvariable=model_path_var)
    model_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

    ttk.Button(model_frame, text="Browse", command=browse_model_path).grid(
        row=0, column=1, sticky="e"
    )

    # ========== SAVE DIRECTORY ==========
    ttk.Label(main_frame, text="Project Save Directory", font=label_font).grid(
        row=4, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5)
    )

    save_frame = ttk.Frame(main_frame)
    save_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
    save_frame.columnconfigure(0, weight=1)  # Make entry expandable

    save_entry = ttk.Entry(save_frame, textvariable=save_dir_var)
    save_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

    ttk.Button(save_frame, text="Browse", command=browse_save_dir).grid(
        row=0, column=1, sticky="e"
    )

    # ========== PREDICT SETTINGS TITLE ==========
    predict_title_frame = ttk.Frame(main_frame)
    predict_title_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 5))
    predict_title_frame.columnconfigure(0, weight=1)

    ttk.Label(predict_title_frame, text="Parameter Settings", font=label_font).grid(
        row=0, column=0, sticky="w", padx=5
    )

    # ========== PARAMETER SETTINGS FRAME ==========
    param_settings_frame = ttk.Frame(
        main_frame, borderwidth=1, relief="groove", padding=5
    )
    param_settings_frame.grid(
        row=7, column=0, columnspan=3, sticky="ew", padx=5, pady=5
    )
    for i in range(4):  # 4 items per row
        param_settings_frame.columnconfigure(i, weight=1)

    # Row 0: First set of parameters
    ttk.Checkbutton(
        param_settings_frame, text="Save Images", variable=save_img_var
    ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

    ttk.Checkbutton(param_settings_frame, text="Save CSV", variable=save_csv_var).grid(
        row=0, column=1, padx=5, pady=5, sticky="w"
    )

    # Confidence threshold
    conf_frame = ttk.Frame(param_settings_frame)
    conf_frame.grid(row=0, column=2, padx=5, pady=5, sticky="w")
    ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
    ttk.Entry(conf_frame, textvariable=conf_var, width=6).pack(side=tk.LEFT)

    # IoU threshold
    iou_frame = ttk.Frame(param_settings_frame)
    iou_frame.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    ttk.Label(iou_frame, text="Int./Union (IoU):").pack(side=tk.LEFT)
    ttk.Entry(iou_frame, textvariable=iou_var, width=6).pack(side=tk.LEFT)

    # Image size
    imgsz_frame = ttk.Frame(param_settings_frame)
    imgsz_frame.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Label(imgsz_frame, text="Image Size:").pack(side=tk.LEFT)
    ttk.Entry(imgsz_frame, textvariable=imgsz_var, width=6).pack(side=tk.LEFT)

    # Max detections
    max_det_frame = ttk.Frame(param_settings_frame)
    max_det_frame.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    ttk.Label(max_det_frame, text="Max Detections:").pack(side=tk.LEFT)
    ttk.Entry(max_det_frame, textvariable=max_det_var, width=6).pack(side=tk.LEFT)

    # Initialize Add Parameter button
    add_param_button = ttk.Button(param_settings_frame, text="Add Parameter")
    add_param_button.grid(row=1, column=2, padx=5, pady=5, sticky="w")
    add_param_button.config(command=create_add_param_callback(add_param_button))

    # ========== START AND RESET BUTTONS (NEW INDEPENDENT ROW) ==========
    control_frame = ttk.Frame(main_frame, padding=5)
    control_frame.grid(row=8, column=0, columnspan=3, sticky="e", padx=5, pady=5)

    start_predict_button = ttk.Button(
        control_frame, text="Start Prediction", command=on_start_prediction
    )
    start_predict_button.pack(side=tk.LEFT, padx=(5, 10))

    reset_button = ttk.Button(control_frame, text="Reset", command=on_reset)
    reset_button.pack(side=tk.LEFT)

    # ========== LOG FRAME ==========
    log_frame = ttk.LabelFrame(main_frame, text="LOGS", padding=5)
    log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

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
