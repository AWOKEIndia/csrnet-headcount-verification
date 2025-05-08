import os
import argparse
from pathlib import Path

def check_dataset_structure(base_path, part='A'):
    """
    Check if the ShanghaiTech dataset structure exists at the given path

    Args:
        base_path (str): Path to check
        part (str): Dataset part, 'A' or 'B'

    Returns:
        bool: True if structure is valid, False otherwise
    """
    # Convert to absolute path
    base_path = os.path.abspath(base_path)
    print(f"Checking dataset at: {base_path}")

    # Expected directory structure
    part_path = os.path.join(base_path, f'ShanghaiTech_Part{part}')
    train_images_path = os.path.join(part_path, 'train_data', 'images')
    train_gt_path = os.path.join(part_path, 'train_data', 'ground-truth')
    test_images_path = os.path.join(part_path, 'test_data', 'images')
    test_gt_path = os.path.join(part_path, 'test_data', 'ground-truth')

    # Check if directories exist
    directories = [
        part_path,
        train_images_path,
        train_gt_path,
        test_images_path,
        test_gt_path
    ]

    all_exist = True
    for dir_path in directories:
        exists = os.path.isdir(dir_path)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        all_exist = all_exist and exists

    # Check for specific files
    if all_exist:
        print("\nChecking for specific files...")
        sample_train_img = os.path.join(train_images_path, 'IMG_1.jpg')
        sample_train_gt = os.path.join(train_gt_path, 'GT_IMG_1.mat')

        img_exists = os.path.isfile(sample_train_img)
        gt_exists = os.path.isfile(sample_train_gt)

        img_status = "✓" if img_exists else "✗"
        gt_status = "✓" if gt_exists else "✗"

        print(f"{img_status} Sample train image: {sample_train_img}")
        print(f"{gt_status} Sample ground truth: {sample_train_gt}")

        all_exist = all_exist and img_exists and gt_exists

    return all_exist

def suggest_fixes(base_path, part='A'):
    """
    Suggest fixes for common dataset structure issues

    Args:
        base_path (str): Path to check
        part (str): Dataset part, 'A' or 'B'
    """
    print("\nSuggested fixes:")

    # Convert to Path object for easier manipulation
    base_path = Path(os.path.abspath(base_path))

    # Check if ShanghaiTech_PartX directory exists directly
    direct_part_path = base_path / f"ShanghaiTech_Part{part}"

    if not direct_part_path.exists():
        # Check if any ShanghaiTech directory exists
        shanghai_dirs = list(base_path.glob("*Shanghai*"))

        if shanghai_dirs:
            print(f"Found potential ShanghaiTech directories:")
            for i, dir_path in enumerate(shanghai_dirs):
                print(f"  {i+1}. {dir_path}")
            print(f"\nTry using one of these paths instead of: {base_path}")
        else:
            # Check parent directories
            parent = base_path.parent
            parent_shanghai_dirs = list(parent.glob("*Shanghai*"))

            if parent_shanghai_dirs:
                print(f"Found potential ShanghaiTech directories in parent folder:")
                for i, dir_path in enumerate(parent_shanghai_dirs):
                    print(f"  {i+1}. {dir_path}")
                print(f"\nTry using one of these paths instead of: {base_path}")
            else:
                print("Could not find any ShanghaiTech directory near the specified path.")
                print("Please ensure you have downloaded and extracted the dataset correctly.")

    # Common structure issues
    print("\nCommon fixes:")
    print("1. Ensure you've downloaded the ShanghaiTech dataset from the official source")
    print("2. Extract the dataset with its original directory structure")
    print("3. When running the script, use absolute paths or correct relative paths")
    print("4. Make sure your working directory is set correctly when running the script")

    # Example command with absolute paths
    print("\nExample command with absolute paths:")
    print(f"python prepare_dataset.py --dataset-path=\"{os.path.abspath(base_path)}\" --part={part}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check ShanghaiTech dataset structure')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to ShanghaiTech dataset directory')
    parser.add_argument('--part', type=str, default='A', choices=['A', 'B'],
                        help='Dataset part, A or B')
    args = parser.parse_args()

    print("ShanghaiTech Dataset Structure Checker")
    print("======================================")

    valid_structure = check_dataset_structure(args.dataset_path, args.part)

    if valid_structure:
        print("\n✓ Dataset structure is valid!")
    else:
        print("\n✗ Dataset structure is invalid!")
        suggest_fixes(args.dataset_path, args.part)
