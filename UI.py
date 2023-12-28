import tkinter as tk
from tkinter import ttk, filedialog


class SLAMBenchmarkUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SLAM Benchmark Framework")

        # Variables
        self.ground_truth_map_path = tk.StringVar()
        self.slam_map_path = tk.StringVar()
        self.comparison_type_var = tk.StringVar()
        self.comparison_type_var.set("Image Similarity")  # Default selection
        self.comparison_method_var = tk.StringVar()
        self.comparison_method_var.set("SSIM")  # Default selection

        # Styles
        self.style = ttk.Style()

        # Layout
        self.create_widgets()

    def create_widgets(self):
        # Frame
        frame = ttk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        # File Selection
        ttk.Label(frame, text="Ground Truth Map:", font=("Helvetica", 12)).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.ground_truth_map_path, state="readonly", font=("Helvetica", 12)).grid(row=0,
                                                                                                                 column=1,
                                                                                                                 pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_ground_truth).grid(row=0, column=2, padx=10)

        ttk.Label(frame, text="SLAM Map:", font=("Helvetica", 12)).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.slam_map_path, state="readonly", font=("Helvetica", 12)).grid(row=1,
                                                                                                         column=1,
                                                                                                         pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_slam_map).grid(row=1, column=2, padx=10)

        # Pre-processing Stage
        ttk.Label(frame, text="Pre-processing Stage", font=("Helvetica", 14, "bold")).grid(row=2, column=0,
                                                                                           columnspan=3, pady=10,
                                                                                           sticky="w")

        ttk.Label(frame, text="1. Image Registration", font=("Helvetica", 12)).grid(row=3, column=0, sticky="w", pady=5)
        ttk.Button(frame, text="Run Image Registration", command=self.run_image_registration).grid(row=3, column=1,
                                                                                                   pady=5)

        ttk.Label(frame, text="2. Thinning Operation", font=("Helvetica", 12)).grid(row=4, column=0, sticky="w", pady=5)
        ttk.Button(frame, text="Run Thinning Operation", command=self.run_thinning_operation).grid(row=4, column=1,
                                                                                                   pady=5)

        # Evaluation Stage
        ttk.Label(frame, text="Evaluation Stage", font=("Helvetica", 14, "bold")).grid(row=5, column=0, columnspan=3,
                                                                                       pady=10, sticky="w")

        # Comparison Type
        ttk.Label(frame, text="3. Comparison Type:", font=("Helvetica", 12)).grid(row=6, column=0, sticky="w", pady=5)
        ttk.OptionMenu(frame, self.comparison_type_var,
                       *("Image Similarity", "Geometric Distance Measurement", "Correspondence Matching"),
                       command=self.update_comparison_method).grid(row=6, column=1, pady=5)

        # Comparison Method
        ttk.Label(frame, text="4. Comparison Method:", font=("Helvetica", 12)).grid(row=7, column=0, sticky="w", pady=5)
        ttk.OptionMenu(frame, self.comparison_method_var, *(), command=self.update_comparison_method).grid(row=7,
                                                                                                           column=1,
                                                                                                           pady=5)

        # Run Benchmark Button
        ttk.Button(frame, text="Run Benchmark", command=self.run_benchmark).grid(row=8, column=0, columnspan=3, pady=20)

    def browse_ground_truth(self):
        file_path = filedialog.askopenfilename(title="Select Ground Truth Map")
        if file_path:
            self.ground_truth_map_path.set(file_path)

    def browse_slam_map(self):
        file_path = filedialog.askopenfilename(title="Select SLAM Map")
        if file_path:
            self.slam_map_path.set(file_path)

    def run_image_registration(self):
        # Implement image registration logic here
        print("Running Image Registration...")

    def run_thinning_operation(self):
        # Implement thinning operation logic here
        print("Running Thinning Operation...")

    def update_comparison_method(self, *args):
        # Update the available comparison methods based on the selected type
        comparison_type = self.comparison_type_var.get()
        if comparison_type == "Image Similarity":
            methods = ("SSIM", "FSIM")
        elif comparison_type == "Geometric Distance Measurement":
            methods = ("Hausdorff Distance", "Wasserstein Distance")
        elif comparison_type == "Correspondence Matching":
            methods = ("ICPc", "ICPe")
        else:
            methods = ()
        self.comparison_method_var.set(methods[0])  # Set default method
        menu = self.root.nametowidget(self.comparison_method_var.trace_info()[0])
        menu.menu.delete(0, tk.END)
        for method in methods:
            menu.menu.add_command(label=method, command=lambda m=method: self.comparison_method_var.set(m))

    def run_benchmark(self):
        # Implement the benchmarking logic here
        # Access paths: self.ground_truth_map_path.get(), self.slam_map_path.get()
        # Access comparison type: self.comparison_type_var.get()
        # Access comparison method: self.comparison_method_var.get()
        print("Running benchmark...")


if __name__ == "__main__":
    root = tk.Tk()
    app = SLAMBenchmarkUI(root)
    root.mainloop()
