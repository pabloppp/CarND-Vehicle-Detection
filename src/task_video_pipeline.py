import pickle
from moviepy.editor import VideoFileClip, CompositeVideoClip

from src.tools.prepare_data import load_model
from src.tools.video_processor import VideoProcessor

model = load_model()
svc = model["svc"]
X_scaler = model["scaler"]

clip = VideoFileClip("../project_video.mp4")  # .subclip(16, 18)
videoProcessor = VideoProcessor(svc, X_scaler)

heatmap = clip.fl_image(videoProcessor.heatmap)
detected = clip.fl_image(videoProcessor.labeled)

combo = CompositeVideoClip([detected, heatmap.resize(0.3).set_pos((886, 10))])
combo.write_videofile("../output_video.mp4", audio=False)
