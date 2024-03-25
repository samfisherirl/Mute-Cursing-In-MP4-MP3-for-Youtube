import winreg


def set_registry_key(path, name, value):
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0, winreg.KEY_ALL_ACCESS) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
            return True
    except PermissionError:
        print("Permission denied: You need to run this script as an administrator.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
    name = "LongPathsEnabled"
    value = 1  # Set to 1 to enable long paths

    if set_registry_key(path, name, value):
        print("Successfully set LongPathsEnabled to 1. You may need to restart your computer for changes to take effect.")
    else:
        print("Failed to set LongPathsEnabled.")
