import os
import shutil
import random
from pathlib import Path
import cv2
import json
import pandas as pd
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageEnhance
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import defaultdict

class AdvancedTerminalSealLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("🔧 Terminal Seal Labeler - Advanced YOLOv8 Edition")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2b2b2b')
        
        # Modern style configuration
        self.setup_style()
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.original_image = None
        self.image_list = []
        self.current_index = 0
        self.bboxes = []  # Current image bounding boxes
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.last_pan_x = 0
        self.last_pan_y = 0
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'annotated_images': 0,
            'total_annotations': 0,
            'current_session_annotations': 0
        }
        
        # Undo/Redo functionality
        self.history = []
        self.history_index = -1
        
        self.setup_ui()
        self.setup_shortcuts()
        self.update_statistics()
        
    def setup_style(self):
        """Modern dark theme setup"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme colors
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('Dark.TButton', background='#404040', foreground='#ffffff')
        style.configure('Accent.TButton', background='#0d7377', foreground='#ffffff')
        style.configure('Danger.TButton', background='#dc3545', foreground='#ffffff')
        style.configure('Success.TButton', background='#28a745', foreground='#ffffff')
        
    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create paned window for layout
        paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_panel = ttk.Frame(paned_window, style='Dark.TFrame', width=400)
        paned_window.add(left_panel, weight=1)
        
        # Right panel for image
        right_panel = ttk.Frame(paned_window, style='Dark.TFrame')
        paned_window.add(right_panel, weight=3)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        # Title
        title_label = ttk.Label(parent, text="🔧 Terminal Seal Labeler", 
                               font=('Arial', 16, 'bold'), style='Dark.TLabel')
        title_label.pack(pady=(0, 20))
        
        # File operations frame
        file_frame = ttk.LabelFrame(parent, text="📁 File Operations", style='Dark.TFrame')
        file_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        ttk.Button(file_frame, text="📂 Load Images Folder", 
                  command=self.load_images_folder, style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        ttk.Button(file_frame, text="💾 Save All Progress", 
                  command=self.save_all_progress, style='Success.TButton').pack(fill=tk.X, pady=2)
        
        ttk.Button(file_frame, text="📊 Export Statistics", 
                  command=self.export_statistics, style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        # Navigation frame
        nav_frame = ttk.LabelFrame(parent, text="🧭 Navigation", style='Dark.TFrame')
        nav_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        nav_buttons_frame = ttk.Frame(nav_frame, style='Dark.TFrame')
        nav_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_buttons_frame, text="⏮️ First", 
                  command=self.first_image, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons_frame, text="◀️ Previous", 
                  command=self.previous_image, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons_frame, text="▶️ Next", 
                  command=self.next_image, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons_frame, text="⏭️ Last", 
                  command=self.last_image, width=8).pack(side=tk.LEFT, padx=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(nav_frame, variable=self.progress_var, 
                                           maximum=100, length=300)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(nav_frame, text="No images loaded", style='Dark.TLabel')
        self.progress_label.pack(pady=2)
        
        # Jump to image
        jump_frame = ttk.Frame(nav_frame, style='Dark.TFrame')
        jump_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(jump_frame, text="Jump to:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(jump_frame, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_frame, text="Go", command=self.jump_to_image).pack(side=tk.LEFT)
        
        # Annotation frame
        annotation_frame = ttk.LabelFrame(parent, text="✏️ Annotation Tools", style='Dark.TFrame')
        annotation_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        annotation_buttons_frame = ttk.Frame(annotation_frame, style='Dark.TFrame')
        annotation_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(annotation_buttons_frame, text="💾 Save", 
                  command=self.save_annotation, style='Success.TButton').pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(annotation_buttons_frame, text="🗑️ Clear", 
                  command=self.clear_boxes, style='Danger.TButton').pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        undo_redo_frame = ttk.Frame(annotation_frame, style='Dark.TFrame')
        undo_redo_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(undo_redo_frame, text="↶ Undo", 
                  command=self.undo_action).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(undo_redo_frame, text="↷ Redo", 
                  command=self.redo_action).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Current annotations list
        ttk.Label(annotation_frame, text="Current Annotations:", style='Dark.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(annotation_frame, style='Dark.TFrame')
        list_frame.pack(fill=tk.X, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.annotations_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                                            height=4, bg='#404040', fg='white',
                                            selectbackground='#0d7377')
        self.annotations_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.annotations_listbox.yview)
        
        # Delete selected annotation
        ttk.Button(annotation_frame, text="🗑️ Delete Selected", 
                  command=self.delete_selected_annotation, style='Danger.TButton').pack(fill=tk.X, pady=2)
        
        # View controls frame
        view_frame = ttk.LabelFrame(parent, text="🔍 View Controls", style='Dark.TFrame')
        view_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        zoom_frame = ttk.Frame(view_frame, style='Dark.TFrame')
        zoom_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(zoom_frame, text="🔍+", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔍-", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔄 Reset", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="📐 Fit", command=self.fit_to_window).pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = ttk.Label(view_frame, text="Zoom: 100%", style='Dark.TLabel')
        self.zoom_label.pack(pady=2)
        
        # Image enhancement frame
        enhance_frame = ttk.LabelFrame(parent, text="🎨 Image Enhancement", style='Dark.TFrame')
        enhance_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Brightness control
        ttk.Label(enhance_frame, text="🔆 Brightness:", style='Dark.TLabel').pack(anchor=tk.W)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(enhance_frame, from_=0.5, to=2.0, 
                                   variable=self.brightness_var, orient=tk.HORIZONTAL,
                                   command=self.update_image_enhancement)
        brightness_scale.pack(fill=tk.X, pady=2)
        
        # Contrast control
        ttk.Label(enhance_frame, text="🌓 Contrast:", style='Dark.TLabel').pack(anchor=tk.W)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(enhance_frame, from_=0.5, to=2.0, 
                                 variable=self.contrast_var, orient=tk.HORIZONTAL,
                                 command=self.update_image_enhancement)
        contrast_scale.pack(fill=tk.X, pady=2)
        
        ttk.Button(enhance_frame, text="🔄 Reset Enhancement", 
                  command=self.reset_enhancement).pack(fill=tk.X, pady=2)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(parent, text="📊 Statistics", style='Dark.TFrame')
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10), padx=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=35, 
                                 bg='#404040', fg='white', font=('Consolas', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add matplotlib graph for statistics
        self.setup_stats_graph(stats_frame)
        
    def setup_right_panel(self, parent):
        # Image display frame
        image_frame = ttk.LabelFrame(parent, text="🖼️ Image Display", style='Dark.TFrame')
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(image_frame, style='Dark.TFrame')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main canvas
        self.canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', 
                               yscrollcommand=v_scrollbar.set,
                               xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.start_pan)  # Right click for panning
        self.canvas.bind("<B3-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-3>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        
        # Status bar
        status_frame = ttk.Frame(image_frame, style='Dark.TFrame')
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready | Use mouse to draw bounding boxes", 
                                     style='Dark.TLabel')
        self.status_label.pack(side=tk.LEFT)
        
        self.coordinates_label = ttk.Label(status_frame, text="", style='Dark.TLabel')
        self.coordinates_label.pack(side=tk.RIGHT)
        
        # Mouse position tracking
        self.canvas.bind("<Motion>", self.update_mouse_coordinates)
        
    def setup_stats_graph(self, parent):
        """Statistics graph setup"""
        try:
            # Create matplotlib figure
            self.fig, self.ax = plt.subplots(figsize=(4, 3), facecolor='#2b2b2b')
            self.ax.set_facecolor('#2b2b2b')
            
            # Embed in tkinter
            self.graph_canvas = FigureCanvasTkAgg(self.fig, parent)
            self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
            
            # Initial empty graph
            self.update_stats_graph()
        except Exception as e:
            print(f"Graph setup failed: {e}")
            
    def setup_shortcuts(self):
        """Keyboard shortcuts setup"""
        shortcuts = {
            '<Control-o>': self.load_images_folder,
            '<Control-s>': self.save_annotation,
            '<Control-z>': self.undo_action,
            '<Control-y>': self.redo_action,
            '<Control-Shift-Z>': self.redo_action,
            '<Delete>': self.clear_boxes,
            '<Right>': self.next_image,
            '<Left>': self.previous_image,
            '<Home>': self.first_image,
            '<End>': self.last_image,
            '<Control-plus>': self.zoom_in,
            '<Control-minus>': self.zoom_out,
            '<Control-0>': self.reset_view,
            '<F1>': self.show_help,
            '<Escape>': self.cancel_current_action,
            '<space>': self.next_image,
            '<BackSpace>': self.previous_image,
        }
        
        for key, command in shortcuts.items():
            self.root.bind(key, lambda event, cmd=command: cmd())
            
    def load_images_folder(self):
        """Images folder load karo"""
        folder_path = filedialog.askdirectory(title="Select Images Folder")
        if folder_path:
            self.image_list = []
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            for ext in extensions:
                self.image_list.extend(Path(folder_path).glob(f'*{ext}'))
                self.image_list.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            
            if self.image_list:
                self.image_list.sort()  # Sort for consistent order
                self.current_index = 0
                self.stats['total_images'] = len(self.image_list)
                self.load_current_image()
                self.update_statistics()
                self.update_progress()
                messagebox.showinfo("Success", f"Loaded {len(self.image_list)} images")
                self.status_label.config(text=f"Loaded {len(self.image_list)} images")
            else:
                messagebox.showwarning("Warning", "No images found in selected folder")
                
    def load_current_image(self):
        """Current image load karo"""
        if not self.image_list:
            return
            
        self.current_image_path = self.image_list[self.current_index]
        
        try:
            # Original image load karo
            img = cv2.imread(str(self.current_image_path))
            if img is None:
                messagebox.showerror("Error", f"Cannot load image: {self.current_image_path.name}")
                return
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(img_rgb)
            
            # Image enhancement apply karo
            self.current_image = self.apply_enhancement(self.original_image)
            
            # Display image
            self.display_image()
            
            # Load existing annotations
            self.load_existing_annotation()
            
            # Update UI
            self.update_progress()
            self.update_annotations_list()
            self.status_label.config(text=f"Loaded: {self.current_image_path.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def apply_enhancement(self, image):
        """Image enhancement apply karo"""
        if not image:
            return None
            
        enhanced = image.copy()
        
        # Brightness
        if self.brightness_var.get() != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(self.brightness_var.get())
            
        # Contrast
        if self.contrast_var.get() != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(self.contrast_var.get())
            
        return enhanced
        
    def display_image(self):
        """Image canvas par display karo"""
        if not self.current_image:
            return
            
        # Zoom apply karo
        img_width = int(self.current_image.width * self.zoom_factor)
        img_height = int(self.current_image.height * self.zoom_factor)
        
        display_image = self.current_image.resize((img_width, img_height), Image.Resampling.LANCZOS)
        
        # Canvas par display karo
        self.photo = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo)
        
        # Scroll region set karo
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Bounding boxes draw karo
        self.draw_all_boxes()
        
        # Zoom label update karo
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        
    def draw_all_boxes(self):
        """Saare bounding boxes draw karo"""
        self.canvas.delete("bbox")
        
        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2 = bbox
            
            # Zoom aur pan apply karo
            x1_display = int(x1 * self.zoom_factor) + self.pan_x
            y1_display = int(y1 * self.zoom_factor) + self.pan_y
            x2_display = int(x2 * self.zoom_factor) + self.pan_x
            y2_display = int(y2 * self.zoom_factor) + self.pan_y
            
            # Rectangle draw karo
            self.canvas.create_rectangle(
                x1_display, y1_display, x2_display, y2_display,
                outline="#00ff00", width=2, tags="bbox"
            )
            
            # Label add karo
            self.canvas.create_text(
                x1_display, y1_display - 10,
                text=f"Terminal Seal {i+1}",
                fill="#00ff00", anchor=tk.W, tags="bbox",
                font=('Arial', 8, 'bold')
            )
            
    def on_canvas_click(self, event):
        """Canvas click event"""
        if not self.current_image:
            return
            
        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        self.start_x = int((canvas_x - self.pan_x) / self.zoom_factor)
        self.start_y = int((canvas_y - self.pan_y) / self.zoom_factor)
        
        self.drawing = True
        self.status_label.config(text="Drawing bounding box...")
        
    def on_canvas_drag(self, event):
        """Canvas drag event"""
        if self.drawing and self.current_image:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            current_x = int((canvas_x - self.pan_x) / self.zoom_factor)
            current_y = int((canvas_y - self.pan_y) / self.zoom_factor)
            
            # Preview rectangle draw karo
            self.canvas.delete("preview")
            
            x1_display = int(self.start_x * self.zoom_factor) + self.pan_x
            y1_display = int(self.start_y * self.zoom_factor) + self.pan_y
            x2_display = int(current_x * self.zoom_factor) + self.pan_x
            y2_display = int(current_y * self.zoom_factor) + self.pan_y
            
            self.canvas.create_rectangle(
                x1_display, y1_display, x2_display, y2_display,
                outline="#ff0000", width=2, tags="preview"
            )
            
    def on_canvas_release(self, event):
        """Canvas release event"""
        if self.drawing and self.current_image:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            end_x = int((canvas_x - self.pan_x) / self.zoom_factor)
            end_y = int((canvas_y - self.pan_y) / self.zoom_factor)
            
            # Valid box check karo
            if abs(end_x - self.start_x) > 10 and abs(end_y - self.start_y) > 10:
                # Coordinates normalize karo
                x1 = min(self.start_x, end_x)
                y1 = min(self.start_y, end_y)
                x2 = max(self.start_x, end_x)
                y2 = max(self.start_y, end_y)
                
                # Image boundaries check karo
                x1 = max(0, min(x1, self.current_image.width))
                y1 = max(0, min(y1, self.current_image.height))
                x2 = max(0, min(x2, self.current_image.width))
                y2 = max(0, min(y2, self.current_image.height))
                
                # History save karo
                self.save_to_history()
                
                # Bounding box add karo
                self.bboxes.append([x1, y1, x2, y2])
                self.update_annotations_list()
                self.display_image()
                
                self.status_label.config(text=f"Added bounding box. Total: {len(self.bboxes)}")
                self.stats['current_session_annotations'] += 1
            
            self.drawing = False
            self.canvas.delete("preview")
            
    def start_pan(self, event):
        """Pan mode start karo"""
        self.panning = True
        self.last_pan_x = event.x
        self.last_pan_y = event.y
        self.status_label.config(text="Panning image...")
        
    def pan_image(self, event):
        """Image pan karo"""
        if self.panning:
            dx = event.x - self.last_pan_x
            dy = event.y - self.last_pan_y
            
            self.pan_x += dx
            self.pan_y += dy
            
            self.last_pan_x = event.x
            self.last_pan_y = event.y
            
            self.display_image()
            
    def end_pan(self, event):
        """Pan mode end karo"""
        self.panning = False
        self.status_label.config(text="Ready")
        
    def on_mouse_wheel(self, event):
        """Mouse wheel zoom"""
        if self.current_image:
            # Get mouse position relative to canvas
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Zoom in/out
            if event.delta > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1
                
            # Limit zoom
            self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
            
            self.display_image()
            
    def on_double_click(self, event):
        """Double click to fit image"""
        self.fit_to_window()
        
    def update_mouse_coordinates(self, event):
        """Mouse coordinates update karo"""
        if self.current_image:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            img_x = int((canvas_x - self.pan_x) / self.zoom_factor)
            img_y = int((canvas_y - self.pan_y) / self.zoom_factor)
            
            if 0 <= img_x < self.current_image.width and 0 <= img_y < self.current_image.height:
                self.coordinates_label.config(text=f"({img_x}, {img_y})")
            else:
                self.coordinates_label.config(text="")
                
    def zoom_in(self):
        """Zoom in on the image"""
        if self.current_image:
            self.zoom_factor *= 1.1
            self.zoom_factor = min(self.zoom_factor, 10.0)  # Max zoom limit
            self.display_image()

    def zoom_out(self):
        """Zoom out of the image"""
        if self.current_image:
            self.zoom_factor /= 1.1
            self.zoom_factor = max(0.1, self.zoom_factor)  # Min zoom limit
            self.display_image()

    def reset_view(self):
        """Reset zoom and pan to default"""
        if self.current_image:
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.display_image()

    def fit_to_window(self):
        """Fit image to canvas window"""
        if self.current_image and self.canvas.winfo_width() > 1:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            img_width = self.current_image.width
            img_height = self.current_image.height
            
            # Calculate zoom to fit
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            self.zoom_factor = min(width_ratio, height_ratio) * 0.95  # 95% to leave some margin
            
            # Center the image
            self.pan_x = (canvas_width - img_width * self.zoom_factor) / 2
            self.pan_y = (canvas_height - img_height * self.zoom_factor) / 2
            
            self.display_image()

    def update_image_enhancement(self, *args):
        """Update image with current enhancement settings"""
        if self.original_image:
            self.current_image = self.apply_enhancement(self.original_image)
            self.display_image()

    def reset_enhancement(self):
        """Reset image enhancement settings"""
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.update_image_enhancement()

    def save_to_history(self):
        """Save current state to history for undo/redo"""
        # Remove future states if any
        self.history = self.history[:self.history_index + 1]
        # Save current state
        self.history.append(list(self.bboxes))  # Deep copy
        self.history_index += 1
        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1

    def undo_action(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.bboxes = list(self.history[self.history_index])  # Deep copy
            self.update_annotations_list()
            self.display_image()
            self.status_label.config(text="Undo performed")

    def redo_action(self):
        """Redo last undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.bboxes = list(self.history[self.history_index])  # Deep copy
            self.update_annotations_list()
            self.display_image()
            self.status_label.config(text="Redo performed")

    def update_annotations_list(self):
        """Update annotations listbox"""
        self.annotations_listbox.delete(0, tk.END)
        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2 = bbox
            self.annotations_listbox.insert(tk.END, 
                f"Seal {i+1}: ({x1}, {y1}) to ({x2}, {y2})")

    def delete_selected_annotation(self):
        """Delete selected annotation from listbox"""
        selection = self.annotations_listbox.curselection()
        if selection:
            self.save_to_history()
            index = selection[0]
            self.bboxes.pop(index)
            self.update_annotations_list()
            self.display_image()
            self.stats['current_session_annotations'] -= 1
            self.status_label.config(text=f"Deleted annotation {index + 1}")

    def clear_boxes(self):
        """Clear all bounding boxes for current image"""
        if self.bboxes and messagebox.askyesno("Confirm", "Clear all annotations for this image?"):
            self.save_to_history()
            self.bboxes = []
            self.update_annotations_list()
            self.display_image()
            self.stats['current_session_annotations'] -= len(self.bboxes)
            self.status_label.config(text="Cleared all annotations")

    def load_existing_annotation(self):
        """Load existing annotations for current image"""
        self.bboxes = []
        annotation_path = self.current_image_path.with_suffix('.txt')
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    for line in f:
                        # Assuming YOLO format: class x_center y_center width height
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, x_center, y_center, width, height = map(float, parts)
                            img_width = self.current_image.width
                            img_height = self.current_image.height
                            
                            # Convert YOLO format to absolute coordinates
                            x1 = int((x_center - width/2) * img_width)
                            y1 = int((y_center - height/2) * img_height)
                            x2 = int((x_center + width/2) * img_width)
                            y2 = int((y_center + height/2) * img_height)
                            
                            self.bboxes.append([x1, y1, x2, y2])
                            
                self.update_annotations_list()
                self.stats['total_annotations'] += len(self.bboxes)
                if len(self.bboxes) > 0:
                    self.stats['annotated_images'] += 1
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to load annotations: {str(e)}")

    def save_annotation(self):
        """Save annotations for current image in YOLO format"""
        if not self.current_image_path or not self.bboxes:
            return
            
        annotation_path = self.current_image_path.with_suffix('.txt')
        img_width = self.current_image.width
        img_height = self.current_image.height
        
        try:
            with open(annotation_path, 'w') as f:
                for x1, y1, x2, y2 in self.bboxes:
                    # Convert to YOLO format: class x_center y_center width height
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
            self.status_label.config(text=f"Saved annotations for {self.current_image_path.name}")
            self.stats['annotated_images'] = sum(1 for img in self.image_list 
                                               if img.with_suffix('.txt').exists())
            self.update_statistics()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

    def save_all_progress(self):
        """Save annotations for all images"""
        if not self.image_list:
            return
            
        # Save current image annotations
        self.save_annotation()
        
        # Create a labels directory if it doesn't exist
        labels_dir = self.image_list[0].parent / 'labels'
        labels_dir.mkdir(exist_ok=True)
        
        self.status_label.config(text="Saved all progress")
        messagebox.showinfo("Success", "All annotations saved successfully")

    def update_statistics(self):
        """Update statistics display"""
        self.stats_text.delete(1.0, tk.END)
        stats_str = (
            f"Total Images: {self.stats['total_images']}\n"
            f"Annotated Images: {self.stats['annotated_images']}\n"
            f"Total Annotations: {self.stats['total_annotations']}\n"
            f"Session Annotations: {self.stats['current_session_annotations']}\n"
            f"Current Image: {self.current_index + 1}/{self.stats['total_images']}"
        )
        self.stats_text.insert(tk.END, stats_str)
        self.update_stats_graph()

    def update_stats_graph(self):
        """Update statistics graph"""
        try:
            self.ax.clear()
            self.ax.set_facecolor('#2b2b2b')
            self.fig.patch.set_facecolor('#2b2b2b')
            
            # Data for graph
            labels = ['Annotated', 'Unannotated']
            values = [self.stats['annotated_images'], 
                     self.stats['total_images'] - self.stats['annotated_images']]
            colors = ['#0d7377', '#dc3545']
            
            self.ax.bar(labels, values, color=colors)
            self.ax.set_ylabel('Number of Images', color='white')
            self.ax.tick_params(colors='white')
            
            # Set text color for better visibility
            for spine in self.ax.spines.values():
                spine.set_edgecolor('white')
                
            self.graph_canvas.draw()
        except Exception as e:
            print(f"Graph update failed: {e}")

    def export_statistics(self):
        """Export statistics to CSV"""
        if not self.image_list:
            return
            
        output_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Statistics"
        )
        
        if output_file:
            stats_data = {
                'Metric': ['Total Images', 'Annotated Images', 
                          'Total Annotations', 'Session Annotations'],
                'Value': [
                    self.stats['total_images'],
                    self.stats['annotated_images'],
                    self.stats['total_annotations'],
                    self.stats['current_session_annotations']
                ]
            }
            
            df = pd.DataFrame(stats_data)
            df.to_csv(output_file, index=False)
            messagebox.showinfo("Success", f"Statistics exported to {output_file}")

    def update_progress(self):
        """Update progress bar and label"""
        if self.image_list:
            progress = (self.current_index + 1) / len(self.image_list) * 100
            self.progress_var.set(progress)
            self.progress_label.config(
                text=f"Image {self.current_index + 1}/{len(self.image_list)}"
            )
        else:
            self.progress_var.set(0)
            self.progress_label.config(text="No images loaded")

    def show_help(self):
        """Show help dialog"""
        help_text = (
            "Terminal Seal Labeler - Help\n\n"
            "Shortcuts:\n"
            "Ctrl+O: Load images folder\n"
            "Ctrl+S: Save current annotation\n"
            "Ctrl+Z: Undo\n"
            "Ctrl+Y/Ctrl+Shift+Z: Redo\n"
            "Delete: Clear all boxes\n"
            "Left/Right: Previous/Next image\n"
            "Home/End: First/Last image\n"
            "Ctrl++/Ctrl+-: Zoom in/out\n"
            "Ctrl+0: Reset view\n"
            "F1: Show this help\n"
            "Esc: Cancel current action\n"
            "Space: Next image\n"
            "Backspace: Previous image\n\n"
            "Mouse Controls:\n"
            "Left-click + drag: Draw bounding box\n"
            "Right-click + drag: Pan image\n"
            "Mouse wheel: Zoom in/out\n"
            "Double-click: Fit to window"
        )
        messagebox.showinfo("Help", help_text)

    def cancel_current_action(self):
        """Cancel current drawing or panning action"""
        if self.drawing:
            self.drawing = False
            self.canvas.delete("preview")
            self.status_label.config(text="Drawing cancelled")
        elif self.panning:
            self.panning = False
            self.status_label.config(text="Panning cancelled")

    # Navigation methods
    def first_image(self):
        if self.image_list:
            self.current_index = 0
            self.load_current_image()
            
    def previous_image(self):
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
            
    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()
            
    def last_image(self):
        if self.image_list:
            self.current_index = len(self.image_list) - 1
            self.load_current_image()
            
    def jump_to_image(self):
        try:
            index = int(self.jump_entry.get()) - 1  # 1-based to 0-based
            if 0 <= index < len(self.image_list):
                self.current_index = index
                self.load_current_image()
            else:
                messagebox.showwarning("Invalid Index", f"Please enter number between 1 and {len(self.image_list)}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number")

    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedTerminalSealLabeler(root)
    app.run()