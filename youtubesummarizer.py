import os
import cv2
import numpy as np
from PIL import Image
import imagehash
import easyocr
from pytube import Search, YouTube
from pytube.exceptions import AgeRestrictedError
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# This adjusts SSL settings to handle sites with unverified certificates
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def detect_major_frames(video_path, subject):
    """ Detects and saves major frames from a video based on scene changes and visual content. """
    threshold = 50.0
    min_scene_len = 30  # Minimum scene length
    black_threshold = 20
    phash_threshold = 10
    colorfulness_threshold = 25

    cap = cv2.VideoCapture(video_path)
    detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    major_scenes = {}
    attractive_frames = {}
    phash_dict = {}

    if cap.isOpened():
        frame_num = 0
        current_scene = 0
        scene_start_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % 10 == 0:  # Process every 10 frames
                scene_list = detector.process_frame(frame_num, frame)
                if scene_list:
                    handle_scene_change(frame_num, cap, current_scene, scene_start_frame, major_scenes, phash_dict, black_threshold, phash_threshold)
                    current_scene += 1
                    scene_start_frame = frame_num
                if is_attractive_frame(frame, colorfulness_threshold):
                    handle_attractive_frame(frame, current_scene, attractive_frames, phash_dict, phash_threshold)
        cap.release()
        return save_frames(subject, major_scenes, attractive_frames)
    else:
        print(f"Error: Unable to open video file {video_path}")

def handle_scene_change(frame_num, cap, current_scene, scene_start_frame, major_scenes, phash_dict, black_threshold, phash_threshold):
    """ Handles the detection of scene changes and saves the major frames. """
    scene_end_frame = frame_num - 1
    scene_middle_frame = (scene_start_frame + scene_end_frame) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, scene_middle_frame)
    ret, middle_frame = cap.read()
    if ret and not is_black_frame(middle_frame, black_threshold):
        phash = imagehash.phash(Image.fromarray(middle_frame))
        if current_scene not in phash_dict or not is_similar_phash(phash, phash_dict[current_scene], phash_threshold):
            major_scenes[current_scene] = middle_frame
            phash_dict[current_scene] = phash

def handle_attractive_frame(frame, current_scene, attractive_frames, phash_dict, phash_threshold):
    """ Evaluates and saves attractive frames based on visual content. """
    phash = imagehash.phash(Image.fromarray(frame))
    if current_scene not in attractive_frames or not is_similar_phash(phash, phash_dict[current_scene], phash_threshold):
        attractive_frames[current_scene] = frame
        phash_dict[current_scene] = phash

def is_attractive_frame(frame, colorfulness_threshold):
    """ Determines if a frame is visually attractive based on its colorfulness. """
    colorfulness = calculate_colorfulness(frame)
    return colorfulness > colorfulness_threshold

def calculate_colorfulness(frame):
    """ Calculates the colorfulness metric of an image frame. """
    (B, G, R) = cv2.split(frame.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    stdRoot = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    meanRoot = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return stdRoot + (0.3 * meanRoot)

def is_similar_phash(phash1, phash2, threshold):
    """ Checks if two perceptual hashes are similar within a given threshold. """
    return phash1 - phash2 <= threshold

def is_black_frame(frame, threshold):
    """ Determines if a frame is predominantly black. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold

def save_frames(subject, major_scenes, attractive_frames):
    """ Saves key frames identified from major scenes and attractive frames to disk. """
    reader = easyocr.Reader(['en'])  # Initialize the OCR reader
    all_texts = []
    image_paths = []
    for scene, frame in {**major_scenes, **attractive_frames}.items():
        key_frame_path = f'{subject}_scene_{scene}.jpg'
        cv2.imwrite(key_frame_path, frame)
        text = perform_ocr_and_watermark(key_frame_path, reader, "Noya Arav")
        print(f"Text in scene {scene}: {text}")
        all_texts.append(text)
        image_paths.append(key_frame_path)
    display_gif(image_paths, all_texts, subject)  # Call display_gif after frames are saved and processed


        
def perform_ocr_and_watermark(image_path, reader, watermark_text):
    """ Performs OCR on an image and adds a watermark. """
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    text = ' '.join([result[1] for result in results])

    # Adding watermark
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
    position = (image.shape[1] - text_size[0] - 10, image.shape[0] - 10)
    cv2.putText(image, watermark_text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(image_path, image)  # Save the watermarked image

    return text


def display_gif(image_paths, extracted_texts, subject):
    """ Create and display an animated GIF from saved images, and print concatenated texts. """
    print("Creating an animated GIF with all the frames:")
    if image_paths:
        num_frames = len(image_paths)
        duration = min(10000 // num_frames, 1000)  # Maximum 1 second per frame
        gif_path = f"{subject}_summary.gif"
        create_gif(image_paths, gif_path, duration)

        # Open the GIF file
        gif_image = Image.open(gif_path)

        # Display the GIF
        gif_image.show()

        # Print the concatenated text from all frames
        all_text = "\n".join(extracted_texts)
        print("Concatenated text from all frames:")
        print(all_text)
    else:
        print("No images found to create the summary.")

def create_gif(image_paths, gif_path, duration):
    frames = [Image.open(x).convert("P", palette=Image.ADAPTIVE, colors=256) for x in image_paths]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)


    

def search_and_download(subject):
    """ Searches for and downloads a video from YouTube based on a subject query. """
    search_results = Search(subject).results

    downloaded_count = 0

    for video in search_results:
        if video.length < 600:  # Check if video duration is less than 10 minutes
            print(f"Downloading video: {video.title}")
            try:
                # Download the video
                youtube = YouTube(video.watch_url)
                video_stream = youtube.streams.get_highest_resolution()
                video_path = f"{subject}_1.mp4"  # Naming convention for the first video
                video_stream.download(filename=video_path)
                print("Video downloaded successfully!")
                
                # Detect major frames in the downloaded video
                print("start detect ‘scenes’")
                detect_major_frames(video_path, subject)
                print("detect ‘scenes’ finish successfully!")
                downloaded_count += 1
                break  # Stop processing after downloading the first suitable video
            except AgeRestrictedError:
                print("Video is age-restricted and cannot be downloaded. Skipping...")
            except Exception as e:
                print(f"An error occurred while downloading video: {str(e)}")

    if downloaded_count == 0:
        print("No videos found less than 10 minutes.")


def main():
    """ Main function to handle user input and process videos. """
    subject = input("Enter a subject to search on YouTube: ").strip()
    search_and_download(subject)

if __name__ == '__main__':
    main()
