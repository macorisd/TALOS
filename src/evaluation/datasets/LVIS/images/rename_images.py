import os

def remove_leading_zeros_from_filenames():
    script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '\\')
    
    for filename in os.listdir(script_directory):
        if filename.endswith(".jpg"):
            name_part, ext = os.path.splitext(filename)
            new_name = name_part.lstrip('0') + ext
            
            if new_name and new_name != filename:
                old_path = os.path.join(script_directory, filename)
                new_path = os.path.join(script_directory, new_name)
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_name}')

if __name__ == "__main__":
    remove_leading_zeros_from_filenames()
