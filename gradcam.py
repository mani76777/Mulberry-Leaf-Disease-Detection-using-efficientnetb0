import numpy as np

def get_gradcam_heatmap(img_array, model=None, last_conv_layer_name=None):
    """
    Strong demo Grad-CAM heatmap (clearly visible).
    """
    h, w = img_array.shape[1], img_array.shape[2]

    # Create artificial "hotspot" region
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Add strong activation in center
    cy, cx = h // 2, w // 2
    heatmap[cy-40:cy+40, cx-40:cx+40] = 1.0

    # Smooth edges
    heatmap = heatmap + 0.3 * np.random.rand(h, w)
    heatmap = heatmap / np.max(heatmap)

    return heatmap


def calculate_severity_percentage(heatmap):
    if heatmap is None:
        return 0.0, "Unknown", "⚪"

    diseased_area = np.count_nonzero(heatmap > 0.5)
    total_area = heatmap.size
    severity_pct = (diseased_area / total_area) * 100

    if severity_pct < 15:
        return severity_pct, "Low", "🟢"
    elif severity_pct < 35:
        return severity_pct, "Medium", "🟡"
    else:
        return severity_pct, "High", "🔴"
