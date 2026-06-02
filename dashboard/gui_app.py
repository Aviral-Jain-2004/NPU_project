"""
Hybrid LLM Live Interface – Tkinter GUI
Sends prompts to the Flask backend (backend_server.py) and displays results.

Run: python dashboard/gui_app.py
Requires: backend_server.py running on http://localhost:5000
"""

import tkinter as tk
from tkinter import messagebox

import requests

BACKEND_URL = "http://localhost:5000/infer"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid LLM Live Interface")
        self.root.geometry("700x520")
        self.root.resizable(False, False)

        # --- Prompt input ---
        tk.Label(root, text="Enter your prompt:").pack(anchor="w", padx=10, pady=(10, 0))
        self.prompt_entry = tk.Entry(root, width=90)
        self.prompt_entry.insert(0, "Explain heterogeneous computing in simple terms.")
        self.prompt_entry.pack(padx=10, pady=5)

        # --- Run button ---
        self.run_btn = tk.Button(root, text="Run Inference", command=self.run_inference)
        self.run_btn.pack(pady=5)

        # --- Output display ---
        tk.Label(root, text="Generated Output:").pack(anchor="w", padx=10, pady=(10, 0))
        self.output_text = tk.Text(root, height=15, width=85, wrap="word", state="disabled")
        self.output_text.pack(padx=10, pady=5)

        # --- Metrics ---
        metrics_frame = tk.Frame(root)
        metrics_frame.pack(padx=10, pady=10, anchor="w")

        tk.Label(metrics_frame, text="Latency:").grid(row=0, column=0, sticky="w")
        self.latency_var = tk.StringVar(value="—")
        tk.Label(metrics_frame, textvariable=self.latency_var).grid(row=0, column=1, padx=10, sticky="w")

        tk.Label(metrics_frame, text="Tokens/sec:").grid(row=1, column=0, sticky="w")
        self.tps_var = tk.StringVar(value="—")
        tk.Label(metrics_frame, textvariable=self.tps_var).grid(row=1, column=1, padx=10, sticky="w")

    def run_inference(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Input Required", "Please enter a prompt.")
            return

        # Disable button while waiting
        self.run_btn.config(state="disabled", text="Running …")
        self.root.update_idletasks()

        try:
            response = requests.post(BACKEND_URL, json={"prompt": prompt}, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Update output
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", data.get("output", ""))
            self.output_text.config(state="disabled")

            # Update metrics
            self.latency_var.set(f"{data.get('latency', 0):.4f} s")
            self.tps_var.set(f"{data.get('tokens_per_sec', 0):.2f}")

        except requests.ConnectionError:
            messagebox.showerror(
                "Connection Error",
                "Cannot connect to backend.\n\n"
                "Make sure backend_server.py is running:\n"
                "  python backend_server.py"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.run_btn.config(state="normal", text="Run Inference")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
