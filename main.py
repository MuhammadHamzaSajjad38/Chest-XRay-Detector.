from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import os
import matplotlib.cm as cm
from PIL import Image

app = Flask(__name__)
app.secret_key = 'a_very_secure_secret_key_for_your_fyp'  # Use a better secret key

# Load the trained model
# Using a relative path is generally better
try:
    model = load_model(r'models/best_model.keras', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle case where model fails to load

# Class labels for your Chest X-ray project
class_labels = ['Covid-19', 'Normal', 'Pneumonia', 'Tuberculosis']

# Define the uploads folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# HELPER FUNCTIONS (Your existing functions are good, I've made minor improvements)
# ==============================================================================

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_gradcam(img_array, model, last_conv_layer_name='block14_sepconv2_act'):
    """
    Generates a Grad-CAM heatmap.
    Compatible with TF 2.x and Keras 3.
    Uses a fallback approach if standard gradients fail.
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(
        model.inputs,
        [last_conv_layer.output, model.output]
    )

    img_np = np.array(img_array, dtype=np.float32)

    # ── Attempt 1: Standard Grad-CAM (official Keras approach) ──
    img_tensor = tf.cast(img_np, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        model_output = grad_model(img_tensor, training=False)

        conv_output = model_output[0]
        preds = model_output[1]
        if isinstance(conv_output, (list, tuple)):
            conv_output = conv_output[0]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = tf.reshape(tf.cast(preds, tf.float32), (1, -1))
        pred_index = int(np.argmax(preds.numpy().flatten()))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)

    # ── Attempt 2: Fallback if gradients are None ──
    # Split the model: run conv part first, then chain remaining layers
    # inside the tape so the gradient path is explicitly recorded.
    if grads is None:
        print("[INFO] Standard Grad-CAM gradient was None, using fallback approach")
        conv_model = tf.keras.Model(model.input, last_conv_layer.output)
        conv_output_val = conv_model(img_np, training=False)
        conv_output = tf.identity(conv_output_val)

        with tf.GradientTape() as tape2:
            tape2.watch(conv_output)
            x = conv_output
            found = False
            for layer in model.layers:
                if found:
                    x = layer(x)
                if layer.name == last_conv_layer_name:
                    found = True
            preds2 = tf.reshape(tf.cast(x, tf.float32), (1, -1))
            pred_index = int(np.argmax(preds2.numpy().flatten()))
            class_channel2 = preds2[0, pred_index]

        grads = tape2.gradient(class_channel2, conv_output)

    if grads is None:
        print("[WARNING] Grad-CAM gradients are None even with fallback")
        viz = (img_np[0] * 127.5 + 127.5) if np.min(img_np) < 0 else img_np[0]
        return array_to_img(viz)

    # ── Build the heatmap ──
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_output_np = conv_output[0].numpy()

    heatmap = conv_output_np @ pooled_grads[..., np.newaxis]
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    img_h, img_w = img_np.shape[1], img_np.shape[2]

    import matplotlib
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(255 * heatmap)]
    jet_heatmap = array_to_img(jet_heatmap).resize((img_w, img_h))
    jet_heatmap = img_to_array(jet_heatmap)

    original_img = (img_np[0] * 127.5) + 127.5 if np.min(img_np) < 0 else img_np[0] * 255
    superimposed = jet_heatmap * 0.7 + original_img
    return array_to_img(superimposed)

def predict_condition(image_path):
    """Loads an image, preprocesses it, gets a prediction and Grad-CAM."""
    IMAGE_SIZE = (299, 299)
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Preprocess the image for Xception model
    preprocessed_img = tf.keras.applications.xception.preprocess_input(img_array_expanded.copy())

    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])
    
    gradcam_img = generate_gradcam(preprocessed_img, model)
    
    # Save original and heatmap images for display
    original_filename = os.path.basename(image_path)
    heatmap_filename = f"heatmap_{original_filename}"
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
    gradcam_img.save(heatmap_path)
    
    return (
        class_labels[predicted_class_index], 
        f"{confidence_score*100:.2f}", 
        f'uploads/{original_filename}',
        f'uploads/{heatmap_filename}'
    )

# WEBSITE ROUTES
# ==============================================================================

# Route for the beautiful new Homepage
@app.route('/')
def index():
    return render_template('index.html', title='Home')

# Route for the Detector Tool page
# In your app.py file

# NEW: Route for the Detector Tool page (handles both GET and POST)
@app.route('/detector', methods=['GET', 'POST'])
def detector():
    # This block handles the form submission
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)
            
            try:
                result, confidence, original_img, heatmap_img = predict_condition(file_location)
                
                # =================================================================
                #                *** THIS IS THE CRITICAL LINE ***
                # It MUST render 'detector.html' to keep the two-column layout.
                # =================================================================
                return render_template('detector.html', 
                                    title='Results',
                                    result=result, 
                                    confidence=confidence,
                                    original_img=original_img,
                                    heatmap_img=heatmap_img)
            except Exception as e:
                import traceback
                # Print full traceback to console for debugging
                print(f"\n[ERROR] Analysis failed: {e}")
                traceback.print_exc()
                flash(f'An error occurred during analysis: {e}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'warning')
            return redirect(request.url)
    
    # This block handles the initial visit to the page (GET request)
    # It correctly renders 'detector.html' without any result data.
    return render_template('detector.html', title='Detector Tool')

# Route for the "About Project" page
@app.route('/about')
def about():
    return render_template('about.html', title='About the Project')

# Route for the "How It Works" page
@app.route('/technology')
def technology():
    return render_template('technology.html', title='How It Works')

# Route for the "Dataset" page
@app.route('/dataset')
def dataset():
    return render_template('dataset.html', title='Dataset Information')

# Route for the "Model Results" page
@app.route('/results')
def results():
    return render_template('results.html', title='Model Performance')

# Route for the "Our Team" page
@app.route('/team')
def team():
    return render_template('team.html', title='Meet the Team')

# Route for the "Disclaimer" page
@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html', title='Disclaimer')

# This route is useful for serving the uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Main execution point
# ==============================================================================
if __name__ == '__main__':
    if model is None:
        print("Model could not be loaded. The application will not work correctly.")
    # Port 7860 is required for Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860, debug=False)