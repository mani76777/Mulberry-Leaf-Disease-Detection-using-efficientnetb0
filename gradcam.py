import tensorflow as tf
import numpy as np

def get_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, tf.argmax(preds[0])]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = np.maximum(heatmap.numpy(), 0)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap.astype(np.float32)
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None

def calculate_severity_percentage(heatmap):
    if heatmap is None:
        return 0, "Unknown", "⚪"
    
    # Calculate what % of pixels are above a 'diseased' threshold (0.2)
    diseased_area = np.count_nonzero(heatmap > 0.2)
    total_area = heatmap.size
    severity_pct = (diseased_area / total_area) * 100
    
    if severity_pct < 15: 
        return severity_pct, "Low", "🟢"
    elif severity_pct < 35: 
        return severity_pct, "Medium", "🟡"
    else: 
        return severity_pct, "High", "🔴"