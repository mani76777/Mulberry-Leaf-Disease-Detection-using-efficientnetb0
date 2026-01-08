import numpy as np

def get_gradcam_heatmap(img_array, model=None, last_conv_layer_name=None):
    """
    TensorFlow-free DEMO Grad-CAM heatmap generator.
    This simulates attention regions for cloud deployment.
    """
    try:
        # img_array shape: (1, H, W, C)
        height = img_array.shape[1]
        width = img_array.shape[2]

        # Generate a fake but smooth heatmap
        heatmap = np.random.rand(height, width)

        # Normalize
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap.astype(np.float32)

    except Exception as e:
        print(f"Grad-CAM Demo Error: {e}")
        return None


def calculate_severity_percentage(heatmap):
    """
    Calculate infection severity based on heatmap intensity.
    """
    if heatmap is None:
        return 0.0, "Unknown", "⚪"

    # Threshold for infected region
    diseased_area = np.count_nonzero(heatmap > 0.4)
    total_area = heatmap.size
    severity_pct = (diseased_area / total_area) * 100

    if severity_pct < 15:
        return severity_pct, "Low", "🟢"
    elif severity_pct < 35:
        return severity_pct, "Medium", "🟡"
    else:
        return severity_pct, "High", "🔴"
