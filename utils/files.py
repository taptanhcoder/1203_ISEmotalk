
import os, shutil


def mkdirs(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)