import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser
from PIL import Image, ImageTk
import os

# Global variables
current_image = None
original_image = None
image_history = []
history_index = -1
drawing = False
drawing_mode = "pen"
start_x, start_y = -1, -1
last_x, last_y = None, None
current_color = (255, 0, 0)  # BGR
text_to_draw = ""
camera_active = False
cap = None
rt_detect = False
mouse_x, mouse_y = 0, 0
preview_photo = None

line_thickness = 3
font_scale = 1.2

# Load cascades
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(os.path.join(SCRIPT_DIR, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(SCRIPT_DIR, 'haarcascade_eye.xml'))

if face_cascade.empty() or eye_cascade.empty():
    messagebox.showerror("Error", "Missing Haar cascade files!\nPlace them in the same folder.")

# ========== IMAGE LOADING & SAVING ==========
def load_image():
    global current_image, original_image, image_history, history_index
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    if file_path:
        img = cv2.imread(file_path)
        if img is not None:
            current_image = img.copy()
            original_image = img.copy()
            image_history = [img.copy()]
            history_index = 0
            update_display()
            update_info()

def save_image():
    global current_image
    if current_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            cv2.imwrite(file_path, current_image)
            messagebox.showinfo("Saved", f"Image saved: {file_path}")

# ========== HISTORY ==========
def save_to_history():
    global image_history, history_index, current_image
    if current_image is not None:
        image_history = image_history[:history_index + 1]
        image_history.append(current_image.copy())
        history_index += 1

def undo():
    global current_image, history_index
    if history_index > 0:
        history_index -= 1
        current_image = image_history[history_index].copy()
        update_display()
        update_info()

def redo():
    global current_image, history_index
    if history_index < len(image_history) - 1:
        history_index += 1
        current_image = image_history[history_index].copy()
        update_display()
        update_info()

# ========== DISPLAY ==========
def update_display():
    global preview_photo, current_image
    if current_image is not None:
        img_rgb = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (720, 520))
        img_pil = Image.fromarray(img_rgb)
        preview_photo = ImageTk.PhotoImage(img_pil)
        panel.config(image=preview_photo)
        panel.image = preview_photo

def update_info():
    global current_image
    if current_image is not None:
        h, w = current_image.shape[:2]
        ch = current_image.shape[2] if len(current_image.shape) > 2 else 1
        info_label.config(text=f"Size: {w} × {h} px • {ch} channels")

# ========== CURSOR ==========
def update_cursor_position(event):
    global mouse_x, mouse_y, current_image
    if current_image is not None:
        img_h, img_w = current_image.shape[:2]
        mouse_x = int(event.x * img_w / 720)
        mouse_y = int(event.y * img_h / 520)
        mouse_x = max(0, min(mouse_x, img_w - 1))
        mouse_y = max(0, min(mouse_y, img_h - 1))
        coord_label.config(text=f"X: {mouse_x}  Y: {mouse_y}")
    else:
        coord_label.config(text="X: -  Y: -")

# ========== LIVE PREVIEW & FREEHAND DRAWING (FIXED) ==========
def show_live_preview(temp_img):
    global preview_photo
    rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (720, 520))
    pil = Image.fromarray(rgb)
    preview_photo = ImageTk.PhotoImage(pil)
    panel.config(image=preview_photo)
    panel.image = preview_photo

def on_mouse_drag(event):
    global last_x, last_y, current_image
    if not drawing or current_image is None:
        return

    h, w = current_image.shape[:2]
    sx = w / 720.0
    sy = h / 520.0

    curr_x = int(event.x * sx)
    curr_y = int(event.y * sy)

    if drawing_mode == "pen":
        if last_x is not None:
            prev_x = int(last_x * sx)
            prev_y = int(last_y * sy)
            # Draw directly on current_image for immediate feedback
            cv2.line(current_image, (prev_x, prev_y), (curr_x, curr_y), current_color, line_thickness)
            update_display()  # Refresh display after each segment
        last_x = event.x
        last_y = event.y
    else:
        # Live preview for shapes/crop
        temp = current_image.copy()
        x1 = int(start_x * sx)
        y1 = int(start_y * sy)
        x2 = curr_x
        y2 = curr_y

        color = (0, 255, 0) if drawing_mode == "crop" else current_color
        thickness = line_thickness

        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        rw = right - left
        rh = bottom - top

        if drawing_mode in ["rectangle", "crop"]:
            cv2.rectangle(temp, (left, top), (right, bottom), color, thickness)
        elif drawing_mode == "circle":
            center = (left + rw//2, top + rh//2)
            radius = min(rw, rh) // 2
            if radius > 0:
                cv2.circle(temp, center, radius, color, thickness)
        elif drawing_mode == "ellipse":
            center = (left + rw//2, top + rh//2)
            axes = (rw//2, rh//2)
            if axes[0] > 0 and axes[1] > 0:
                cv2.ellipse(temp, center, axes, 0, 0, 360, color, thickness)

        show_live_preview(temp)

# ========== DRAWING START/STOP ==========
def start_draw(event):
    global drawing, start_x, start_y, last_x, last_y, current_image
    if current_image is None or drawing_mode == "none":
        return
    save_to_history()
    drawing = True
    start_x = event.x
    start_y = event.y
    last_x = event.x
    last_y = event.y

def stop_draw(event):
    global drawing, last_x, last_y, current_image
    if not drawing or current_image is None:
        return
    drawing = False

    if drawing_mode != "pen":
        h, w = current_image.shape[:2]
        sx = w / 720.0
        sy = h / 520.0

        x1 = int(start_x * sx)
        y1 = int(start_y * sy)
        x2 = int(event.x * sx)
        y2 = int(event.y * sy)

        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        rw = right - left
        rh = bottom - top

        thickness = line_thickness

        if drawing_mode == "rectangle":
            cv2.rectangle(current_image, (left, top), (right, bottom), current_color, thickness)
        elif drawing_mode == "circle":
            center = (left + rw//2, top + rh//2)
            radius = min(rw, rh) // 2
            if radius > 0:
                cv2.circle(current_image, center, radius, current_color, thickness)
        elif drawing_mode == "ellipse":
            center = (left + rw//2, top + rh//2)
            axes = (rw//2, rh//2)
            if axes[0] > 0 and axes[1] > 0:
                cv2.ellipse(current_image, center, axes, 0, 0, 360, current_color, thickness)
        elif drawing_mode == "crop":
            if rw > 20 and rh > 20:
                current_image = current_image[top:bottom, left:right]
                update_info()

    last_x = None
    last_y = None
    update_display()
    update_info()

# ========== COLOR & SIZE ==========
def choose_color():
    global current_color
    color = colorchooser.askcolor(title="Choose Drawing Color")
    if color[0]:
        r, g, b = map(int, color[0])
        current_color = (b, g, r)
        color_button.config(bg=color[1])

def update_line_thickness(val):
    global line_thickness
    line_thickness = int(val)

def update_font_scale(val):
    global font_scale
    font_scale = float(val)

# ========== MODES ==========
def set_mode_pen():
    global drawing_mode
    drawing_mode = "pen"
    status_label.config(text="Mode: Freehand Pen")

def set_mode_rect():
    global drawing_mode
    drawing_mode = "rectangle"
    status_label.config(text="Mode: Rectangle")

def set_mode_circle():
    global drawing_mode
    drawing_mode = "circle"
    status_label.config(text="Mode: Circle")

def set_mode_ellipse():
    global drawing_mode
    drawing_mode = "ellipse"
    status_label.config(text="Mode: Ellipse")

def set_mode_crop():
    global drawing_mode
    drawing_mode = "crop"
    status_label.config(text="Mode: Crop")

def set_mode_normal():
    global drawing_mode
    drawing_mode = "none"
    status_label.config(text="Mode: Normal")

# ========== TEXT ==========
def add_text():
    global text_to_draw
    text = simpledialog.askstring("Add Text", "Enter text:")
    if text:
        text_to_draw = text
        messagebox.showinfo("Text", "Right-click on image to place text")

def insert_text(event):
    global current_image, text_to_draw
    if current_image is not None and text_to_draw:
        save_to_history()
        h, w = current_image.shape[:2]
        x = int(event.x * w / 720)
        y = int(event.y * h / 520)
        cv2.putText(current_image, text_to_draw, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, current_color, line_thickness)
        text_to_draw = ""
        update_display()

# ========== BASIC TOOLS ==========
def convert_grayscale():
    global current_image
    if current_image is not None:
        save_to_history()
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        update_display()

def resize_image():
    global current_image
    if current_image is not None:
        save_to_history()
        h, w = current_image.shape[:2]
        new_w = simpledialog.askinteger("Resize", "New width:", initialvalue=w)
        new_h = simpledialog.askinteger("Resize", "New height:", initialvalue=h)
        if new_w and new_h and new_w > 0 and new_h > 0:
            current_image = cv2.resize(current_image, (new_w, new_h))
            update_display()
            update_info()

def rotate_image():
    global current_image
    if current_image is not None:
        save_to_history()
        angle = simpledialog.askfloat("Rotate", "Angle (degrees):", initialvalue=90)
        if angle is not None:
            h, w = current_image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            current_image = cv2.warpAffine(current_image, M, (w, h))
            update_display()

# ========== DETECTION ==========
def detect_corners():
    global current_image
    if current_image is not None:
        save_to_history()
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            for corner in corners:
                x, y = np.int0(corner).ravel()
                cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
        update_display()

def detect_face_eyes():
    global current_image
    if current_image is not None:
        save_to_history()
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(current_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(current_image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                cx = x + ex + ew // 2
                cy = y + ey + eh // 2
                cv2.circle(current_image, (cx, cy), min(ew, eh)//4, (255, 0, 0), 2)
        update_display()

# ========== CHANNELS ==========
def _modify_channel(channel, value):
    global current_image
    if current_image is not None and len(current_image.shape) == 3:
        save_to_history()
        if channel == 'R': current_image[:,:,2] = value
        elif channel == 'G': current_image[:,:,1] = value
        elif channel == 'B': current_image[:,:,0] = value
        update_display()

def remove_red(): _modify_channel('R', 0)
def remove_green(): _modify_channel('G', 0)
def remove_blue(): _modify_channel('B', 0)
def add_red(): _modify_channel('R', 255)
def add_green(): _modify_channel('G', 255)
def add_blue(): _modify_channel('B', 255)

# ========== FILTERS ==========
def gaussian_blur():
    global current_image
    if current_image is not None:
        save_to_history()
        current_image = cv2.GaussianBlur(current_image, (15, 15), 0)
        update_display()

def median_blur():
    global current_image
    if current_image is not None:
        save_to_history()
        current_image = cv2.medianBlur(current_image, 15)
        update_display()

def canny_edges():
    global current_image
    if current_image is not None:
        save_to_history()
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        update_display()

# ========== LOGICAL OPERATIONS ==========
def logical_op(op_type):
    global current_image, original_image
    if current_image is not None and original_image is not None:
        save_to_history()
        orig_resized = cv2.resize(original_image, (current_image.shape[1], current_image.shape[0]))
        if op_type == "AND":
            result = cv2.bitwise_and(current_image, orig_resized)
        elif op_type == "OR":
            result = cv2.bitwise_or(current_image, orig_resized)
        elif op_type == "XOR":
            result = cv2.bitwise_xor(current_image, orig_resized)
        elif op_type == "ADD":
            result = cv2.add(current_image, orig_resized)
        elif op_type == "SUBTRACT":
            result = cv2.subtract(current_image, orig_resized)
        current_image = result
        update_display()

def op_and(): logical_op("AND")
def op_or(): logical_op("OR")
def op_xor(): logical_op("XOR")
def op_add(): logical_op("ADD")
def op_subtract(): logical_op("SUBTRACT")

def blend():
    global current_image, original_image
    if current_image is not None and original_image is not None:
        save_to_history()
        alpha = simpledialog.askfloat("Blend", "Alpha (0.0 - 1.0):", initialvalue=0.5)
        if alpha is not None:
            orig_resized = cv2.resize(original_image, (current_image.shape[1], current_image.shape[0]))
            current_image = cv2.addWeighted(current_image, alpha, orig_resized, 1 - alpha, 0)
            update_display()

# ========== CAMERA ==========
def toggle_camera():
    global camera_active, cap, camera_button
    if not camera_active:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            camera_active = True
            camera_button.config(text="Stop Camera")
            capture_frame()
        else:
            messagebox.showerror("Error", "Cannot open camera")
    else:
        camera_active = False
        if cap: cap.release()
        camera_button.config(text="Start Camera")

def toggle_rt_detect():
    global rt_detect, rt_button
    if not camera_active:
        messagebox.showwarning("Warning", "Start camera first!")
        return
    rt_detect = not rt_detect
    rt_button.config(text="Stop RT Detect" if rt_detect else "Start RT Detect")

def capture_frame():
    global current_image
    if camera_active and cap:
        ret, frame = cap.read()
        if ret:
            if rt_detect:
                frame = detect_face_eyes_rt(frame)
            current_image = frame.copy()
            update_display()
            update_info()
        root.after(10, capture_frame)

def detect_face_eyes_rt(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            cx = x + ex + ew // 2
            cy = y + ey + eh // 2
            cv2.circle(frame, (cx, cy), min(ew, eh)//4, (255, 0, 0), 2)
    return frame

# ========== DIMENSIONS ==========
def show_dimensions():
    global current_image
    if current_image is not None:
        h, w = current_image.shape[:2]
        messagebox.showinfo("Dimensions", f"Width: {w}px\nHeight: {h}px")

# ========== GUI ==========
def create_gui():
    global root, panel, info_label, coord_label, status_label, camera_button, rt_button, color_button

    root = tk.Tk()
    root.title("Functional Photoshop")
    root.geometry("1100x780")
    root.configure(bg="#1e1e1e")

    left = tk.Frame(root, bg="#252526", width=300)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
    left.pack_propagate(False)

    canvas = tk.Canvas(left, bg="#252526", highlightthickness=0)
    scrollbar = tk.Scrollbar(left, orient="vertical", command=canvas.yview, bg="#424242")
    scroll_frame = tk.Frame(canvas, bg="#252526")

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def header(text):
        tk.Label(scroll_frame, text=text, font=("Helvetica", 13, "bold"), fg="#cccccc", bg="#252526", anchor="w").pack(pady=(20,8), padx=20, fill=tk.X)

    def btn(text, cmd):
        b = tk.Button(scroll_frame, text=text, command=cmd, font=("Helvetica", 10), bg="#3c3c3c", fg="white", relief="flat", height=1)
        b.pack(pady=4, padx=20, fill=tk.X, anchor="w")
        b.bind("<Enter>", lambda e: b.config(bg="#505050"))
        b.bind("<Leave>", lambda e: b.config(bg="#3c3c3c"))
        return b

    header("File")
    btn("Load Image", load_image)
    btn("Save Image", save_image)

    header("Drawing Settings")
    global color_button
    color_button = tk.Button(scroll_frame, text="Choose Color", command=choose_color, font=("Helvetica", 10, "bold"), bg="#0000ff", fg="white", relief="flat")
    color_button.pack(pady=8, padx=20, fill=tk.X)

    tk.Label(scroll_frame, text="Line Thickness", fg="#cccccc", bg="#252526", anchor="w").pack(padx=20, anchor="w")
    tk.Scale(scroll_frame, from_=1, to=20, orient=tk.HORIZONTAL, command=update_line_thickness, bg="#252526", fg="white", highlightthickness=0, troughcolor="#3c3c3c").pack(pady=5, padx=20, fill=tk.X)

    tk.Label(scroll_frame, text="Text Size", fg="#cccccc", bg="#252526", anchor="w").pack(padx=20, anchor="w")
    tk.Scale(scroll_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL, command=update_font_scale, bg="#252526", fg="white", highlightthickness=0, troughcolor="#3c3c3c").pack(pady=5, padx=20, fill=tk.X)

    header("Drawing Tools")
    btn("Freehand Pen", set_mode_pen)
    btn("Rectangle", set_mode_rect)
    btn("Circle", set_mode_circle)
    btn("Ellipse", set_mode_ellipse)
    btn("Crop Mode", set_mode_crop)
    btn("Normal Mode", set_mode_normal)
    btn("Add Text", add_text)

    header("Basic Tools")
    btn("B&W", convert_grayscale)
    btn("Resize", resize_image)
    btn("Rotate", rotate_image)

    header("Detection")
    btn("Detect Corners", detect_corners)
    btn("Detect Face & Eyes", detect_face_eyes)

    header("Channels")
    btn("Remove Red", remove_red)
    btn("Remove Green", remove_green)
    btn("Remove Blue", remove_blue)
    btn("Add Red", add_red)
    btn("Add Green", add_green)
    btn("Add Blue", add_blue)

    header("Filters")
    btn("Gaussian Blur", gaussian_blur)
    btn("Median Blur", median_blur)
    btn("Edge Detection", canny_edges)

    header("Logical Operations")
    btn("AND", op_and)
    btn("OR", op_or)
    btn("XOR", op_xor)
    btn("Add", op_add)
    btn("Subtract", op_subtract)
    btn("Blend", blend)

    header("Camera")
    camera_button = btn("Start Camera", toggle_camera)
    rt_button = btn("Start RT Detect", toggle_rt_detect)

    header("History")
    btn("Undo", undo)
    btn("Redo", redo)
    btn("Show Dimensions", show_dimensions)

    # Right panel
    right = tk.Frame(root, bg="#1e1e1e")
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    panel = tk.Label(right, bg="#2d2d2d", relief="sunken", bd=2)
    panel.pack(fill=tk.BOTH, expand=True)

    # Bottom bar
    bottom = tk.Frame(right, bg="#252526", height=50)
    bottom.pack(fill=tk.X, pady=(10,0))
    bottom.pack_propagate(False)

    status_label = tk.Label(bottom, text="Mode: Freehand Pen", fg="#cccccc", bg="#252526", font=("Helvetica", 11))
    status_label.pack(side=tk.LEFT, padx=20)

    coord_label = tk.Label(bottom, text="X: -  Y: -", fg="#00ff00", bg="#252526", font=("Helvetica", 11, "bold"))
    coord_label.pack(side=tk.RIGHT, padx=20)

    global info_label
    info_label = tk.Label(bottom, text="No image loaded", fg="#cccccc", bg="#252526", font=("Helvetica", 10))
    info_label.pack(side=tk.LEFT, padx=100)

    # Mouse events
    panel.bind("<Motion>", update_cursor_position)
    panel.bind("<B1-Motion>", on_mouse_drag)
    panel.bind("<Button-1>", start_draw)
    panel.bind("<ButtonRelease-1>", stop_draw)
    panel.bind("<Button-3>", insert_text)

    root.mainloop()

if __name__ == "__main__":
    create_gui()