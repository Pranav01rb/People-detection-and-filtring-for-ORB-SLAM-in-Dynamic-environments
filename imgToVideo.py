import cv2
import os


def display_images_like_video(folder_path, frame_delay=30):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder not found")
        return

    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(
        os.path.join(folder_path, f))]

    # Filter out image files (assuming JPEG and PNG formats)
    image_files = [f for f in files if f.lower().endswith(
        ('.png', '.jpg', '.jpeg'))]

    # Sort the files to maintain an order, if necessary
    image_files.sort()

    # Check if there are any image files in the folder
    if not image_files:
        print("No images found in the folder")
        return

    cv2.namedWindow('Image Sequence', cv2.WINDOW_AUTOSIZE)

    # Iterate over the image files
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(folder_path, image_file)

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image {image_file}")
            continue

        # Display the image
        cv2.imshow('Image Sequence', image)

        # Wait for a specified time or key press to move to the next image
        if cv2.waitKey(frame_delay) >= 0:  # Press any key to exit the loop
            break

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


# Example usage
folder_path = r'C:\Users\yasha\OneDrive\Desktop\Desktop stuff\Interesting stuff\ASU 2nd sem\Perception\Project\Training\Output\Originalimages'
display_images_like_video(folder_path)
