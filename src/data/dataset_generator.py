"""
Dataset Generator for Viola-Jones Face Detector
Extracts 16×16 patches from Faces94 dataset

Assignment: ELL715 Assignment 5 - Part 1 (20 marks)
AI Usage: Structure and docstrings assisted by Claude Code
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
from tqdm import tqdm
from skimage import color


class DatasetGenerator:
    """Generate training and testing patches from Faces94 dataset"""

    def __init__(self, faces94_path, patch_size=16, num_nonface_per_image=5, min_distance=24):
        """
        Initialize dataset generator

        Args:
            faces94_path: Path to Faces94 directory
            patch_size: Size of square patches (default 16×16)
            num_nonface_per_image: Number of random non-face patches per image
            min_distance: Minimum distance from center for non-face patches
        """
        self.faces94_path = Path(faces94_path)
        self.patch_size = patch_size
        self.num_nonface_per_image = num_nonface_per_image
        self.min_distance = min_distance

        # Training folders
        self.train_folders = [
            self.faces94_path / 'female',
            self.faces94_path / 'malestaff'
        ]

        # Testing folder
        self.test_folder = self.faces94_path / 'male'

    def load_image(self, image_path):
        """
        Load image and convert to grayscale

        Args:
            image_path: Path to image file

        Returns:
            Grayscale image as numpy array (0-255)
        """
        img = Image.open(image_path)
        img_array = np.array(img)

        # Convert to grayscale if RGB
        if len(img_array.shape) == 3:
            img_gray = color.rgb2gray(img_array)
            img_gray = (img_gray * 255).astype(np.uint8)
        else:
            img_gray = img_array

        return img_gray

    def get_all_image_paths(self, folders):
        """
        Get all .jpg image paths from given folders

        Args:
            folders: List of folder paths

        Returns:
            List of image paths
        """
        image_paths = []
        for folder in folders:
            # Iterate through subject directories
            for subject_dir in sorted(folder.iterdir()):
                if subject_dir.is_dir():
                    # Get all .jpg files
                    jpg_files = list(subject_dir.glob('*.jpg'))
                    image_paths.extend(jpg_files)

        return image_paths

    def extract_center_patch(self, image):
        """
        Extract center 16×16 patch (face class)

        Args:
            image: Grayscale image array

        Returns:
            16×16 center patch
        """
        H, W = image.shape
        center_y = H // 2
        center_x = W // 2

        # Extract patch centered at image center
        half_size = self.patch_size // 2
        y_start = center_y - half_size
        y_end = center_y + half_size
        x_start = center_x - half_size
        x_end = center_x + half_size

        patch = image[y_start:y_end, x_start:x_end]

        return patch

    def is_valid_nonface_location(self, x, y, center_x, center_y):
        """
        Check if location is valid for non-face patch (not too close to center)

        Args:
            x, y: Top-left corner of candidate patch
            center_x, center_y: Center of image

        Returns:
            True if valid, False otherwise
        """
        # Calculate patch center
        patch_center_x = x + self.patch_size // 2
        patch_center_y = y + self.patch_size // 2

        # Calculate distance from image center
        distance = np.sqrt((patch_center_x - center_x)**2 +
                          (patch_center_y - center_y)**2)

        return distance > self.min_distance

    def extract_random_nonface_patches(self, image):
        """
        Extract random 16×16 patches avoiding center (non-face class)

        Args:
            image: Grayscale image array

        Returns:
            List of random patches
        """
        H, W = image.shape
        center_y = H // 2
        center_x = W // 2

        patches = []
        max_attempts = 100  # Avoid infinite loop

        for _ in range(self.num_nonface_per_image):
            attempts = 0
            while attempts < max_attempts:
                # Random top-left corner
                x = np.random.randint(0, W - self.patch_size)
                y = np.random.randint(0, H - self.patch_size)

                # Check if valid (not too close to center)
                if self.is_valid_nonface_location(x, y, center_x, center_y):
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    patches.append(patch)
                    break

                attempts += 1

            # If couldn't find valid location, use a corner patch
            if attempts == max_attempts:
                x = 0 if len(patches) % 2 == 0 else W - self.patch_size
                y = 0 if len(patches) < 3 else H - self.patch_size
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)

        return patches

    def generate_patches_from_images(self, image_paths):
        """
        Generate face and non-face patches from list of images

        Args:
            image_paths: List of image paths

        Returns:
            face_patches: Array of face patches (N, 16, 16)
            nonface_patches: Array of non-face patches (N*5, 16, 16)
        """
        face_patches = []
        nonface_patches = []

        print(f"Processing {len(image_paths)} images...")
        for img_path in tqdm(image_paths):
            # Load image
            image = self.load_image(img_path)

            # Extract center patch (face)
            face_patch = self.extract_center_patch(image)
            face_patches.append(face_patch)

            # Extract random patches (non-face)
            nonface_patch_list = self.extract_random_nonface_patches(image)
            nonface_patches.extend(nonface_patch_list)

        # Convert to numpy arrays
        face_patches = np.array(face_patches, dtype=np.uint8)
        nonface_patches = np.array(nonface_patches, dtype=np.uint8)

        return face_patches, nonface_patches

    def generate_dataset(self, output_dir='data/processed'):
        """
        Generate complete training and testing datasets

        Saves:
            - train_faces.pkl: Training face patches
            - train_nonfaces.pkl: Training non-face patches
            - test_faces.pkl: Testing face patches
            - test_nonfaces.pkl: Testing non-face patches
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate training data
        print("\n=== Generating Training Data ===")
        train_image_paths = self.get_all_image_paths(self.train_folders)
        print(f"Found {len(train_image_paths)} training images")

        train_faces, train_nonfaces = self.generate_patches_from_images(train_image_paths)

        print(f"Generated {len(train_faces)} face patches")
        print(f"Generated {len(train_nonfaces)} non-face patches")

        # Generate testing data
        print("\n=== Generating Testing Data ===")
        test_image_paths = self.get_all_image_paths([self.test_folder])
        print(f"Found {len(test_image_paths)} testing images")

        test_faces, test_nonfaces = self.generate_patches_from_images(test_image_paths)

        print(f"Generated {len(test_faces)} face patches")
        print(f"Generated {len(test_nonfaces)} non-face patches")

        # Save to pickle files
        print("\n=== Saving Datasets ===")
        with open(output_path / 'train_faces.pkl', 'wb') as f:
            pickle.dump(train_faces, f)
        print(f"Saved: {output_path / 'train_faces.pkl'}")

        with open(output_path / 'train_nonfaces.pkl', 'wb') as f:
            pickle.dump(train_nonfaces, f)
        print(f"Saved: {output_path / 'train_nonfaces.pkl'}")

        with open(output_path / 'test_faces.pkl', 'wb') as f:
            pickle.dump(test_faces, f)
        print(f"Saved: {output_path / 'test_faces.pkl'}")

        with open(output_path / 'test_nonfaces.pkl', 'wb') as f:
            pickle.dump(test_nonfaces, f)
        print(f"Saved: {output_path / 'test_nonfaces.pkl'}")

        print("\n=== Dataset Generation Complete! ===")
        print(f"Training: {len(train_faces)} faces, {len(train_nonfaces)} non-faces")
        print(f"Testing: {len(test_faces)} faces, {len(test_nonfaces)} non-faces")

        return {
            'train_faces': train_faces,
            'train_nonfaces': train_nonfaces,
            'test_faces': test_faces,
            'test_nonfaces': test_nonfaces
        }


def main():
    """Main function to generate dataset"""
    # Path to Faces94 dataset
    faces94_path = 'Faces94'

    # Create generator
    generator = DatasetGenerator(
        faces94_path=faces94_path,
        patch_size=16,
        num_nonface_per_image=5,
        min_distance=24
    )

    # Generate and save dataset
    dataset = generator.generate_dataset(output_dir='data/processed')


if __name__ == '__main__':
    main()
