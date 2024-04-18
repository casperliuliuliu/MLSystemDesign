import os

def count_files_in_directory(directory):

    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        file_count += len(files)  # Add count of files in the current directory
        
    return file_count

folder_path = 'D:\\Casper\\OTHER\\Data\\dpaml_hw3\\dataset\\empty_photo'  # Replace with your folder path
print("Number of files in directory:", count_files_in_directory(folder_path))