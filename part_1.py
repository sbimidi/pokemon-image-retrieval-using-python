#Name- sandeep reddy bimidi
#ASU ID - 1222081185
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image, ImageTk
from tkinter import ttk

#  average pixel color feature
def average_pixel_color(image):
    return np.mean(image, axis=(0, 1))

# extract color histograms feature
def color_histograms(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

def edge_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges)

# extract Histograms of HoG feature
def hog_feature(image):
    hog = cv2.HOGDescriptor()
    hist = hog.compute(image)
    return hist.flatten()

def extract_features(image, method):
    if method == "Average Pixel Color":
        return average_pixel_color(image)
    elif method == "Color Histograms":
        return color_histograms(image)
    elif method == "Edge Extraction":
        return edge_extraction(image)
    elif method == "HoG":
        return hog_feature(image)
    else:
        return None

# Function to get similar images
def retrieve_similar_images(query_features, feature_library, distance_measure):
    similarities = []
    for image_path, features in feature_library.items():
        print(f"Query features shape: {query_features.shape if isinstance(query_features, np.ndarray) else None}")
        print(f"Features shape: {features.shape if features is not None else None}")
        if isinstance(query_features, np.ndarray):  
            if distance_measure == "Cosine Similarity":
                # Normalize features
                query_features_norm = query_features / np.linalg.norm(query_features)
                features_norm = features / np.linalg.norm(features)
                similarity_score = cosine_similarity([query_features_norm], [features_norm])[0][0]
            else:  
                # Resize 
                min_length = min(query_features.shape[0], features.shape[0])
                query_features_resized = query_features[:min_length]
                features_resized = features[:min_length]
                similarity_score = np.sum(np.square(query_features_resized - features_resized))
            similarities.append((image_path, similarity_score))
        else:
            print("Query features is not a NumPy array.")
    similarities.sort(key=lambda x: x[1])
    return similarities[:10]


def retrieve_images():
    query_image_path = filedialog.askopenfilename()
    if query_image_path:
        query_image = cv2.imread(query_image_path)
        method = selected_method.get()
        query_features = extract_features(query_image, method)
        if query_features is not None:
            similar_images = retrieve_similar_images(query_features, feature_library, selected_distance.get())
            display_results(similar_images)
        else:
            result_label.config(text="Feature extraction method not implemented!")


def display_results(similar_images):
    result_label.config(text="Top 10 Similar Images:")
    
 
    for widget in result_frame.winfo_children():
        widget.destroy()
    
    canvas = tk.Canvas(result_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # scrollbar for the canvas to showl all 10 images "scroll on the scroll bar poin only entire page doesn't scroll"
    scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    result_items_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=result_items_frame, anchor=tk.NW)
    
    for i, (image_path, similarity_score) in enumerate(similar_images):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (100, 100))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Create a frame for each result item
        item_frame = tk.Frame(result_items_frame)
        item_frame.grid(row=i, column=0, padx=10, pady=5, sticky="nw")

        # Display similarity score on the left
        score_label = tk.Label(item_frame, text=f"#{i+1} distance: {similarity_score:.4f}", anchor="w")
        score_label.pack(side=tk.LEFT, padx=10)

        # Display image on the right
        image_label = tk.Label(item_frame, image=tk_image)
        image_label.pack(side=tk.RIGHT, padx=10)
        image_label.image = tk_image  # Keep a reference to avoid garbage collection

    # Update canvas to show all result items
    result_items_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox(tk.ALL))


def load_feature_library():
    feature_library = {}
    database_folder = "/Users/sandeepreddy/Downloads/HW8-FinalProject/Data/Database"
    for filename in os.listdir(database_folder):
        image_path = os.path.join(database_folder, filename)
        image = cv2.imread(image_path)
        features = extract_features(image, selected_method.get())
        feature_library[image_path] = features
    return feature_library

root = tk.Tk()
root.title("Image Retrieval System")

method_frame = tk.Frame(root)
method_frame.pack(pady=10)

selected_method = tk.StringVar()
selected_method.set("Average Pixel Color")  # Default selection
methods = ["Average Pixel Color", "Color Histograms", "Edge Extraction", "HoG"]
for method in methods:
    tk.Radiobutton(method_frame, text=method, variable=selected_method, value=method).pack(side=tk.LEFT)

distance_frame = tk.Frame(root)
distance_frame.pack(pady=10)

# Create radio buttons for selecting distance measure
selected_distance = tk.StringVar()
selected_distance.set("SSD")  
distances = ["SSD", "Angle btw vectors"]
for distance in distances:
    tk.Radiobutton(distance_frame, text=distance, variable=selected_distance, value=distance).pack(side=tk.LEFT)

# Button to initiate image retrieval process
retrieve_button = tk.Button(root, text="SEARCH IMAGE", command=retrieve_images)
retrieve_button.pack(pady=10)

result_frame = tk.Frame(root)
result_frame.pack(pady=10)


result_label = tk.Label(root, text="")
result_label.pack(pady=10)

feature_library = load_feature_library()


root.mainloop()
