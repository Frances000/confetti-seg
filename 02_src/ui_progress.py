# ui_progress.py
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk

class TrainingProgressDialog(tk.Toplevel):
    """
    Non-blocking training progress window with:
      - Title + overall progressbar
      - Current step label
      - Scrolling log
      - Cancel button
    Usage:
      dlg = TrainingProgressDialog(root, total_steps=<int>)
      reporter = dlg.make_reporter()
      # pass reporter(message, step_delta=0) into your training code
    """
    def __init__(self, master=None, total_steps=100, title="Training progress"):
        super().__init__(master)
        self.title(title)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.resizable(True, True)

        self.total_steps = max(1, int(total_steps))
        self.current_steps = 0
        self.cancelled = False
        self.msg_q = queue.Queue()

        # Widgets
        self.lbl = ttk.Label(self, text="Initialising…")
        self.lbl.pack(fill="x", padx=10, pady=(10, 5))

        self.pb = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=self.total_steps)
        self.pb.pack(fill="x", padx=10, pady=5)

        self.txt = tk.Text(self, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=5)
        self.txt.configure(state="disabled")

        self.btn_frame = ttk.Frame(self)
        self.btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.cancel_btn = ttk.Button(self.btn_frame, text="Cancel", command=self._on_cancel)
        self.cancel_btn.pack(side="right")

        # start polling the queue
        self.after(100, self._drain_queue)

    def make_reporter(self):
        """
    Returns a function: reporter(msg: str, step_delta: int = 0)
    - Puts messages + step increments into a queue for UI update in main thread.
        """
        def reporter(msg: str, step_delta: int = 0):
            self.msg_q.put(("msg", msg, step_delta))
        return reporter

    def _append_text(self, text):
        self.txt.configure(state="normal")
        self.txt.insert("end", text + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _drain_queue(self):
        try:
            while True:
                kind, msg, step_delta = self.msg_q.get_nowait()
                if kind == "msg":
                    if step_delta:
                        self.current_steps = min(self.total_steps, self.current_steps + step_delta)
                        self.pb["value"] = self.current_steps
                    self.lbl.configure(text=msg if len(msg) < 150 else msg[:150] + "…")
                    self._append_text(msg)
        except queue.Empty:
            pass
        if not self.cancelled:
            self.after(100, self._drain_queue)

    def _on_cancel(self):
        self.cancelled = True
        self._append_text("⚠️ Cancel requested by user.")
        self.lbl.configure(text="Cancelling… you may close this window.")
        # Progress bar to indeterminate to show pending shutdown
        self.pb.configure(mode="indeterminate")
        self.pb.start(40)

    def _on_close(self):
        # user pressed the close button
        self._on_cancel()
        self.destroy()
