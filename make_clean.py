"""CAUTION

This script removes the following folders

    runs
    weights

"""
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Folders to be deleted.')
    parser.add_argument("--folders", action="store", type=str, nargs="*")
    args = parser.parse_args()

    if args.folders is None:
        print("No folders specified to be deleted.")
    elif args.folders:
        print(f"Folders to be deleted: {' '.join(args.folders)}")
        print("Sure?")
        input_ = input("Continue [y/n]: ")
        if input_ == "y":
            command = f"rm -rf {' '.join(args.folders)}"
            os.system(command)
        else:
            print("Exit.")

