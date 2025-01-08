import cv2

def create_video_from_images(datastore, video_name="output_video.mp4", frame_rate=30, image_type='timex', camera=None, site=None):
    """
    Create a video from images in the datastore with optional filtering by image type, camera, and site.
    
    Parameters:
    - datastore: The ImageDatastore object containing images.
    - video_name (str): The output name for the video.
    - frame_rate (int): The frame rate for the video.
    - image_type (str, optional): The image type to filter (e.g., 'bright', 'snap').
    - camera (str, optional): The camera identifier to filter by (e.g., 'CACO03'). If None, process all cameras.
    - site (str, optional): The site identifier to filter by. If None, process all sites.
    """
    
    # Get all images by type from the datastore, optionally filter by site and camera
    images_metadata = datastore.get_image_metadata_by_type([image_type], site=site, camera=camera)
    
    if not images_metadata:
        print(f"No images found for image_type: {image_type} with the specified filters.")
        return
    
    # Sort images by timestamp (ensure they are in chronological order)
    images_metadata.sort(key=lambda x: x['timestamp'])
    
    # Read the first image to obtain the video dimensions
    first_image = cv2.imread(images_metadata[0]['path'])
    height, width, layers = first_image.shape
    
    # Define video parameters and initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 format
    out = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))  # Using specified frame rate
    
    # Loop through the images and add them to the video
    for img_metadata in images_metadata:
        img_path = img_metadata['path']
        image = cv2.imread(img_path)
        
        # Add text overlay with additional metadata (site, camera, date, and time)
        site = img_metadata['site']
        camera = img_metadata['camera']
        month = img_metadata['month']
        day = img_metadata['day']
        time = img_metadata['time']
        
        text = f"Site: {site} | Camera: {camera} | {month}-{day} {time}"
        
        # Add text overlay to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, height - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the frame to the video
        out.write(image)
    
    # Release the video writer
    out.release()
    print(f"Video created successfully: {video_name}")



