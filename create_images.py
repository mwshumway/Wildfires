import cv2
import os


def extract_frames(video_path, output_folder, frame_numbers=None, interval=None):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        save_frame = False

        if frame_numbers is not None:
            if current_frame in frame_numbers:
                save_frame = True
        elif interval is not None:
            if current_frame % interval == 0:
                save_frame = True

        if save_frame:
            filename = os.path.join(output_folder, f"save_those_homies_frame_{current_frame:05d}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

        current_frame += 1

    cap.release()
    print("Frame extraction complete.")


video_file = "animations/so_many_people_to_save.mp4"
output_dir = "images"


frame_indices = [40,80,120,199]
extract_frames(video_file, output_dir, frame_numbers=frame_indices)

# you could also get every nth frame

