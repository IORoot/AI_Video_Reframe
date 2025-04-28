import argparse
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from video_processor import VideoProcessor
from object_detector import ObjectDetector
from object_tracker import ObjectTracker
from crop_calculator import CropCalculator
from smoothing import CropWindowSmoother
from nicegui import ui





def start_gui():
    """Start the NiceGUI interface."""

    """Override default styles for NiceGUI."""
    ui.query('.nicegui-content').classes('items-stretch')
    # ui.query('.q-field__label').classes('l-auto w-full text-right')
    # ui.query('.q-field--float .q-field__label').style('transform:none')

    ui.label('Content-Aware Video Cropping')

    input_path = ui.input('Input Video Path')
    output_path = ui.input('Output Video Path')
    target_ratio = ui.number('Target Aspect Ratio (width/height)', value=9/16)
    detector = ui.select(['yolo'], label='Object Detection Model', value='yolo')
    smoothing_window = ui.number('Smoothing Window (frames)', value=30)
    skip_frames = ui.number('Skip Frames', value=10)
    model_size = ui.select(['n', 's', 'm', 'l', 'x'], label='YOLOv8 Model Size', value='n')
    conf_threshold = ui.number('Confidence Threshold', value=0.5)
    use_saliency = ui.checkbox('Use Saliency Detection')
    max_workers = ui.number('Max Worker Threads', value=4)

    def run_script():
        ui.notify('Processing started...')
        try:
            args = argparse.Namespace(
                input=input_path.value,
                output=output_path.value,
                target_ratio=target_ratio.value,
                detector=detector.value,
                smoothing_window=int(smoothing_window.value),
                skip_frames=int(skip_frames.value),
                model_size=model_size.value,
                conf_threshold=float(conf_threshold.value),
                use_saliency=use_saliency.value,
                max_workers=int(max_workers.value),
            )
            main(args)
            ui.notify('Processing complete!')
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            ui.notify(f"Error: {str(e)}")

    ui.button('Run', on_click=run_script)

    ui.run()






def parse_args():
    parser = argparse.ArgumentParser(description='Content-aware video cropping')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--target_ratio', type=float, default=9/16, help='Target aspect ratio (width/height)')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'ssd', 'faster_rcnn'], 
                        help='Object detection model to use')
    parser.add_argument('--smoothing_window', type=int, default=30, 
                        help='Number of frames for temporal smoothing')
    parser.add_argument('--skip_frames', type=int, default=10, 
                        help='Process every nth frame for detection (1 = process all frames)')
    parser.add_argument('--model_size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--use_saliency', action='store_true',
                        help='Use saliency detection for regions of interest')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads for parallel processing')
    return parser.parse_args()





def process_keyframe(frame_idx, frame, detector, tracker, tracked_objects_by_frame):
    """Process a keyframe with detection and tracking."""
    print(f"🐤 main : process_keyframe: {frame_idx}")

    # Detect objects in frame
    detected_objects = detector.detect(
        frame,      # 🕵️‍♂️ Detect objects in the frame
        top_n=1,    # 🏷️ Get the top N detection (highest confidence) 
    )
    
    # Update tracker with new detections
    tracked_objects = tracker.update(frame, detected_objects)
    tracked_objects_by_frame[frame_idx] = tracked_objects
    
    return frame_idx





def main(args=None):

    if args is None:
        args = parse_args()
    
    # Initialize components with YOLOv8
    video_processor = VideoProcessor()


    detector = ObjectDetector(
        confidence_threshold=args.conf_threshold,   # 🕵️‍♂️ Confidence threshold for object detection (0-1).
        model_size=args.model_size,                # 📏 Size of the YOLOv8 model (n=small, s=medium, m=large, l=xlarge).
        classes=[0],                               # 🏷️ Classes to detect (0=person, 1=bicycle, 4=car, 7=truck, etc...).
        debug=False,                                # 🐛 If True, saves debug images and logs to help you visualize decisions.
    )

    tracker = ObjectTracker(
        max_disappeared=30,     # 🕵️‍♂️ Number of frames an object can be missing before being considered lost.
        max_distance=50         # 🔍 Maximum distance for re-identifying lost objects (in pixels).
    )


    crop_calculator = CropCalculator(
        target_ratio=args.target_ratio,  # 📐 Desired aspect ratio of the crop (e.g., 16/9 for widescreen). Example: 1.77
        padding_ratio=0.1,               # ➡️ Add 10% padding around the detected area to avoid tight crops.
        class_weights=None,              # 🏷️ Optional: prioritize certain object classes (e.g., faces over background). Example: {0: 1.0, 1: 0.5}
        size_weight=0.4,                 # 📏 How much object size matters (larger objects are more important).
        center_weight=0.3,               # 🎯 How much being close to the frame center matters (centered objects preferred).
        motion_weight=0.3,               # 🎥 How much moving objects are prioritized (good for tracking action).
        history_weight=0.1,              # 🕰️ How much previous frames affect the crop (smoothness over time). Set 0 to ignore history.
        saliency_weight=0.4,             # 👀 How much visual "importance" (saliency maps) matters (e.g., bright or attention-grabbing regions).
        debug=False,                     # 🐛 If True, saves debug images and logs to help you visualize decisions.
        face_detection=False,            # 👤 If True, uses face to enhance detection in the crop. Uses weighted averages.
        weighted_center=False,           # ⚖️ If True, uses weighted average of detected objects' centers for crop center.
        blend_saliency=False,            # 🌈 If True, blends saliency map with detected objects to enhance crop.
    )


    smoother = CropWindowSmoother(
        window_size=args.smoothing_window,  # 📅 Number of frames for smoothing (e.g., 30 for 1 second at 30 FPS).
        position_inertia=0.8,               # 🔄 How much the position of the crop should "stick" to the previous frame (0-1).
        size_inertia=0.9                    # 📏 How much the size of the crop should "stick" to the previous frame (0-1).
    )
    
    # Load video and get properties
    video_info = video_processor.load_video(args.input)
    total_frames = video_info['total_frames']
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    
    print(f"Processing video: {args.input}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    
    # Process frames
    tracked_objects_by_frame = {}
    
    start_time = time.time()
    






    # First pass: detect and track objects on keyframes only
    print("Phase 1: Detecting and tracking objects...")
    
    # Determine keyframes
    keyframes = list(range(0, total_frames, args.skip_frames))
    
    # Use ThreadPoolExecutor for parallel processing of keyframes
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a list to store futures
        futures = []
        
        # Process keyframes
        for frame_idx in keyframes:
            # Set position to keyframe
            video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_processor.cap.read()
            
            if not ret:
                continue
            
            # Submit task to executor
            future = executor.submit(
                process_keyframe, 
                frame_idx, 
                frame, 
                detector, 
                tracker, 
                tracked_objects_by_frame
            )
            futures.append(future)
            
            # Print progress every 10 keyframes
            if len(futures) % 10 == 0:
                print(f"Submitted {len(futures)}/{len(keyframes)} keyframes for processing")
        
        # Wait for all futures to complete
        for i, future in enumerate(futures):
            future.result()  # This will raise any exceptions that occurred
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(keyframes)} keyframes")
    









    # Second pass: calculate crop windows for keyframes
    print("Phase 2: Calculating crop windows for keyframes...")
    
    # Pre-allocate crop windows array
    crop_windows = [None] * total_frames
    
    # Process keyframes
    for frame_idx in keyframes:
        if frame_idx not in tracked_objects_by_frame:
            continue
            
        objects = tracked_objects_by_frame[frame_idx]
        
        # Get the actual frame for additional analysis
        video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_processor.cap.read()
        
        if not ret:
            continue
        
        # Calculate optimal crop window
        crop_window = crop_calculator.calculate(objects, width, height, frame)
        crop_windows[frame_idx] = crop_window
        
        if frame_idx % 100 == 0:
            print(f"Calculated crop window for keyframe {frame_idx}/{total_frames}")
    










    # Phase 3: Interpolate crop windows for non-keyframes
    print("Phase 3: Interpolating crop windows for non-keyframes...")
    
    # Fast interpolation using numpy
    keyframe_indices = np.array(keyframes)
    keyframe_crop_windows = np.array([crop_windows[i] for i in keyframes if crop_windows[i] is not None])
    
    if len(keyframe_crop_windows) > 1:
        # For each frame, find the nearest keyframes and interpolate
        for i in range(total_frames):
            if crop_windows[i] is not None:
                continue
                
            # Find nearest keyframes
            next_idx = keyframe_indices[keyframe_indices > i]
            prev_idx = keyframe_indices[keyframe_indices < i]
            
            if len(next_idx) == 0 and len(prev_idx) > 0:
                # After last keyframe, use last keyframe
                crop_windows[i] = crop_windows[prev_idx[-1]]
            elif len(prev_idx) == 0 and len(next_idx) > 0:
                # Before first keyframe, use first keyframe
                crop_windows[i] = crop_windows[next_idx[0]]
            elif len(prev_idx) > 0 and len(next_idx) > 0:
                # Interpolate between keyframes
                prev_frame = prev_idx[-1]
                next_frame = next_idx[0]
                
                if crop_windows[prev_frame] is not None and crop_windows[next_frame] is not None:
                    # Calculate interpolation factor
                    alpha = (i - prev_frame) / (next_frame - prev_frame)
                    
                    # Linear interpolation
                    prev_crop = np.array(crop_windows[prev_frame])
                    next_crop = np.array(crop_windows[next_frame])
                    interp_crop = prev_crop * (1 - alpha) + next_crop * alpha
                    crop_windows[i] = [int(x) for x in interp_crop]
    
    # Fill any remaining None values with center crop
    for i in range(total_frames):
        if crop_windows[i] is None:
            # Use center crop as fallback
            crop_height = height
            crop_width = int(crop_height * args.target_ratio)
            if crop_width > width:
                crop_width = width
                crop_height = int(crop_width / args.target_ratio)
            x = int((width - crop_width) / 2)
            y = int((height - crop_height) / 2)
            crop_windows[i] = [x, y, crop_width, crop_height]
    










    # Apply temporal smoothing to crop windows
    print("Phase 4: Applying temporal smoothing...")
    #smoothed_windows = smoother.smooth(crop_windows)
    smoothed_windows = crop_windows
    










    # Generate output video with cropped frames
    print("Phase 5: Generating output video...")
    video_processor.generate_output_video(
        output_path=args.output,
        crop_windows=smoothed_windows,
        fps=fps
    )
    
    elapsed_time = time.time() - start_time
    print(f"Video processing completed in {elapsed_time:.2f} seconds")
    print(f"Output saved to: {args.output}")





if __name__ in {"__main__", "__mp_main__"}:
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        start_gui()