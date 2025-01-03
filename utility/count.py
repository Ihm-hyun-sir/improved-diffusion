import os
from collections import Counter

def count_images_by_class(folder_path):
    """
    Counts the number of images for each class in the given folder.

    Args:
        folder_path (str): Path to the folder containing the image files.

    Returns:
        dict: A dictionary where keys are class names and values are the counts.
    """
    class_counts = Counter()

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Ensure it is a file and has an underscore to extract the class name
        if os.path.isfile(os.path.join(folder_path, filename)) and "_" in filename:
            # Extract the class name (text before the first underscore)
            class_name = filename.split("_")[0]
            class_counts[class_name] += 1

    return dict(class_counts)

# Paths to the train and test folders
train_folder = "datasets/cifar_lt_train"
test_folder = "datasets/cifar_lt_test"

# Count images by class for train and test folders
train_class_counts = count_images_by_class(train_folder)
test_class_counts = count_images_by_class(test_folder)

# Print the results
print("Train Class Counts:")
print(train_class_counts)

print("\nTest Class Counts:")
print(test_class_counts)
