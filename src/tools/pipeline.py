from src.tools.hog_window_search import combined_window_search, generate_heatmap


def pipeline(image, svc, X_scaler):
    _, _, _, rects = combined_window_search(image, svc, X_scaler)
    heatmap = generate_heatmap(rects)
    heatmap[heatmap <= 3] = 0
