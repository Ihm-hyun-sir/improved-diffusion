import os

def get_png_file_list(directory_path):
    """
    Returns a list of all PNG file names in the specified directory.

    Args:
        directory_path (str): Path to the directory containing PNG files.

    Returns:
        list: A list of PNG file names in the directory.
    """
    try:
        # List all files in the directory
        file_list = os.listdir(directory_path)
        
        # Filter files with .png extension
        png_files = [file for file in file_list if file.endswith('.png')]
        
        return png_files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_or_create_value(dic, key):
    """
    Returns the value for the given key in the dictionary. If the key does not exist, creates it with a default value of 1.

    Args:
        dic (dict): The dictionary to search.
        key: The key to look for.

    Returns:
        The value associated with the key, or 1 if the key does not exist.
    """
    if key not in dic:
        dic[key] = 1
    return dic[key]

# Directory containing the PNG files
directory = "/home2/ihmhyunsir/WorkingSpace/lab_proj/diffusion_imbalance/idv2/improved-diffusion/datasets/cifar_lt_train"

# Get the list of PNG files
png_file_list = get_png_file_list(directory)

# Print the list of PNG files


# Example usage of get_or_create_value
dist = {}
for name in png_file_list:
    cls_name = name.split("_")[0]

    if cls_name not in dist.keys():
        dist[cls_name] = 0
    dist[cls_name] += 1

print(dist)
