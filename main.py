import clean_basemap
import consolidate_islands
import main_region_growing
import secondary_region_growing
import merge_islands
import collect_data
import generate_names
import collect_neighbor_names
import generate_histories
import generate_map
import tkinter as tk
from tkinter import ttk
import threading
import time
import inspect

def print1():
    time.sleep(1)
    print("1")
    
def print2(text="2"):
    time.sleep(1)
    print(text)
    
def print3(text="3"):
    time.sleep(1)
    print(text)
    
def print4():
    time.sleep(1)
    print("4")
    
    
TASKS = [clean_basemap.main, consolidate_islands.main, main_region_growing.main, secondary_region_growing.main, merge_islands.main, collect_data.main]
ONLINE_TASKS = [generate_names.main, collect_neighbor_names.main, generate_histories.main, generate_map.main]


class ProcessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stochastory")
        self.root.geometry("1000x300") # Slightly taller for the spinner row
        
        self.running = False
        self.stop_event = threading.Event()
        self.current_task_list = []
        self.spinner_id = None
        
        # Unicode characters for the spinner animation
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0

        # --- 1. Control Frame (Checkbox & Input) ---
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(pady=10, fill=tk.X, padx=20)

        self.online_mode_var = tk.BooleanVar(value=True)
        
        self.online_check = ttk.Checkbutton(
            self.control_frame, 
            text="Enable Online Mode", 
            variable=self.online_mode_var,
            command=self.on_mode_toggle
        )
        self.online_check.pack(side=tk.LEFT)

        self.input_frame = ttk.Frame(self.control_frame)
        self.input_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        self.input_label = ttk.Label(self.input_frame, text="API Key:")
        self.input_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.online_entry_var = tk.StringVar()
        self.online_entry = ttk.Entry(self.input_frame, textvariable=self.online_entry_var)
        self.online_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- 2. Action Frame (Start/Stop & Spinner) ---
        self.action_frame = ttk.Frame(root)
        self.action_frame.pack(pady=10)

        # Start Button
        self.start_btn = ttk.Button(self.action_frame, text="START", command=self.start_process)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        # Stop Button (Initially hidden)
        self.stop_btn = ttk.Button(self.action_frame, text="STOP", command=self.stop_process)
        # We use pack_forget initially so it takes no space
        self.stop_btn.pack_forget() 

        # Spinner Label (Initially hidden/empty)
        self.spinner_label = ttk.Label(self.action_frame, text="", width=3, font=("Consolas", 14))
        self.spinner_label.pack_forget()

        # --- 3. Progress Bar ---
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=1)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)

        # --- 4. Status Label ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to start...")
        self.status_label = ttk.Label(root, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.pack(pady=5)

        # Initialize UI State
        self.on_mode_toggle()

    def calculate_total_steps(self):
        """Calculates total steps based on current checkbox state."""
        base_count = len(TASKS)
        online_count = len(ONLINE_TASKS) if self.online_mode_var.get() else 0
        return base_count + online_count

    def start_spinner(self):
        """Starts the spinner animation."""
        self.spinner_label.pack(side=tk.LEFT, padx=10)
        self._animate_spinner()

    def _animate_spinner(self):
        """Internal recursive function to update spinner frame."""
        if self.running:
            self.spinner_label.config(text=self.spinner_frames[self.spinner_index])
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
            # Schedule next frame in 100ms
            self.spinner_id = self.root.after(100, self._animate_spinner)
        else:
            self.spinner_label.config(text="")

    def stop_spinner(self):
        """Stops the spinner animation."""
        if self.spinner_id:
            self.root.after_cancel(self.spinner_id)
            self.spinner_id = None
        self.spinner_label.pack_forget()
        self.spinner_label.config(text="")

    def on_mode_toggle(self):
        """Updates UI visibility and Progress Bar max when checkbox changes."""
        is_checked = self.online_mode_var.get()
        
        # 1. Toggle Input Field Visibility
        if is_checked:
            self.input_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
            # Ensure entry is enabled when visible (unless running)
            if not self.running:
                self.online_entry.config(state=tk.NORMAL)
        else:
            self.input_frame.pack_forget()
            self.online_entry_var.set("") # Clear input when disabled

        # 2. Update Progress Bar Maximum (only if not running)
        if not self.running:
            total = self.calculate_total_steps()
            max_val = total if total > 0 else 1
            
            if hasattr(self, 'progress_bar'):
                self.progress_bar.config(maximum=max_val)
                self.progress_var.set(0)
                
                mode_text = "Online" if is_checked else "Offline"
                self.status_var.set(f"{mode_text} Mode Selected. {total} tasks queued.")

    def start_process(self):
        if self.running:
            return
        
        # --- NEW VALIDATION LOGIC ---
        if self.online_mode_var.get():
            input_data = self.online_entry_var.get().strip()
            if not input_data:
                self.status_var.set("Error: Please enter API Key to use Online Mode.")
                # Shake animation or visual cue could be added here
                return
        
        self.running = True
        
         # UI Updates
        self.start_btn.pack_forget() # Hide Start
        self.stop_btn.pack(side=tk.LEFT, padx=10) # Show Stop
        self.start_spinner() # Start Animation
        
        # Disable controls
        self.start_btn.config(state=tk.DISABLED)
        self.online_check.config(state=tk.DISABLED)
        self.online_entry.config(state=tk.DISABLED)
        
        # Build the execution list dynamically
        self.current_task_list = list(TASKS)
        
        if self.online_mode_var.get():
            self.current_task_list.extend(ONLINE_TASKS)
        
        total_steps = len(self.current_task_list)
        
        if total_steps == 0:
            self.update_status("No tasks to run.")
            self.reset_controls()
            return

        # Update progress bar
        self.progress_bar.config(maximum=total_steps)
        self.progress_var.set(0)
        
        self.update_status("Starting processes...")
        
        thread = threading.Thread(target=self.run_tasks)
        thread.start()

    def stop_process(self):
        """Sets the flag to stop the thread gracefully."""
        self.update_status("Stopping... Please wait for current task to finish.")
        self.stop_event.set()
        
    def run_tasks(self):
        is_online = self.online_mode_var.get()
        # We already validated this isn't empty if is_online is True
        online_data = self.online_entry_var.get() if is_online else None
        #     clean_basemap.main()
        #     consolidate_islands.main()
        #     main_region_growing.main()
        #     secondary_region_growing.main()
        #     merge_islands.main()
        #     collect_data.main()
        #     if online_mode:
        #         generate_names.main(api_key)
        #         collect_neighbor_names.main()
        #         generate_histories.main(api_key)
        for i, func in enumerate(self.current_task_list):
            working_text = ''
            match i:
                case 0:
                    working_text = "Cleaning basemap"
                case 1:
                    working_text = "Consolidating islands"
                case 2:
                    working_text = "Growing regions (First pass)"
                case 3:
                    working_text = "Growing regions (Second pass)"
                case 4:
                    working_text = "Merging islands back to map"
                case 5:
                    working_text = "Collecting data and coloring regions"
                case 6:
                    working_text = "Generating names with ELM"
                case 7:
                    working_text = "Consolidating neighbor data"
                case 8:
                    working_text = "Generating histories with ELM"
                case 9:
                    working_text = "Creating webmap"
                    
            self.update_status(working_text+"...")
            try:
                sig = inspect.signature(func)
                if len(sig.parameters) > 0 and is_online and online_data:
                    func(online_data)
                else:
                    func()
            except Exception as e:
                if self.stop_event.is_set():
                    break # Exit silently if stopped
                self.update_status(f"Error in step {i} - {working_text}: {str(e)}")
                break
            
            # Check stop flag again after function completes
            if self.stop_event.is_set():
                self.update_status("Process stopped by user.")
                break

            self.update_progress(i + 1)
        
        if not self.stop_event.is_set() and self.running:
            self.update_status("All processes completed successfully.")
        
        self.reset_controls()

    def reset_controls(self):
        self.running = False
        
        # UI Reset
        self.stop_spinner()
        self.stop_btn.pack_forget() # Hide Stop
        self.start_btn.pack(side=tk.LEFT, padx=10) # Show Start
        

    
        self.root.after(0, lambda: self.online_check.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

        if self.online_mode_var.get():
            self.root.after(0, lambda: self.online_entry.config(state=tk.NORMAL))


    def update_status(self, message):
        self.root.after(0, lambda: self.status_var.set(message))

    def update_progress(self, value):
        self.root.after(0, lambda: self.progress_var.set(value))

if __name__ == "__main__":
    root = tk.Tk()
    app = ProcessGUI(root)
    root.mainloop()
