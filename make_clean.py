"""CAUTION

This script removes the following folders

    runs
    weights

"""
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Folders to be deleted.')
    parser.add_argument("--folders", "-f", dest="folders", help="Folders to be deleted.", action="store", type=str, nargs="*")
    args = parser.parse_args()

    if args.folders is None:
        print("No folders specified to be deleted.")
    elif args.folders:
        print(f"Folders to be deleted: {' '.join(args.folders)}")
        print("Sure?")
        command = input("Continue [yes / no]: ")
        if command == "yes":
            command = f"rm -rf {' '.join(args.folders)}"
            os.system(command)
        else:
            print("Exit.")

