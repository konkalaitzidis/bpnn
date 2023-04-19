import numpy as np


# normalize the pixel values of the calcium video
def normalize_video(video_data):
    
    max_pixel_value = video_data.max()
    min_pixel_value = video_data.min()
    range_pixel_value = max_pixel_value - min_pixel_value
    normalized_video_data = (video_data - min_pixel_value) / range_pixel_value
    
    # Verify the normalization by checking the minimum and maximum values
    print('Minimum pixel value: {:.3f}' .format(np.min(normalized_video_data)))
    print('Maximum pixel value:', np.max(normalized_video_data))
    
    return normalized_video_data