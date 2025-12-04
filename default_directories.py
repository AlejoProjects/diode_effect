import numpy as np
import shutil
import tempfile

import os

def check_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def create_default_directories(source):

    check_create_dir(source)
    section1_dir = source + "1_df_fixed_current_plots"
    section2_dir = source +"2_zero_currents_field_plots"
    section3_dir = source +"3_magentization_plots"
    section4_dir = source +"4_varying_currents_plots"
    check_create_dir(section1_dir)
    check_create_dir(section2_dir)
    check_create_dir(section3_dir)
    check_create_dir(section4_dir)
    ####################################################
    #Subdirectories created here                        #
    ####################################################
    subsection21_dir = section2_dir + "/21_constant_field_fixed_current"
    subsection22_dir = section2_dir + "/22_varying_field_zero_current"
    subsection23_dir = section2_dir + "/23_fixed_field_zero_current_different_heights"
    subsection41_dir = section4_dir + "/41_constant_field"
    subsection42_dir = section4_dir + "/42_constant_field_different_heights"
    subsection43_dir = section4_dir + "/43_different_fields"
    check_create_dir(subsection21_dir)
    check_create_dir(subsection22_dir)
    check_create_dir(subsection23_dir)
    check_create_dir(subsection41_dir)
    check_create_dir(subsection42_dir)
    check_create_dir(subsection43_dir)
    return section1_dir,section2_dir,section3_dir,section4_dir, subsection21_dir,subsection22_dir,subsection23_dir,subsection41_dir,subsection42_dir,subsection43_dir
    
def clean_source():
    source = "./figures/"
    if os.path.exists(source):
        import shutil
        shutil.rmtree(source)
    create_default_directories()
def clean_directory(directory):
    if os.path.exists(directory):
        import shutil
        shutil.rmtree(directory)
        os.makedirs(directory)
def clean_static():
    source = "./figures/"
    section1_dir = source + "1_df_fixed_current_plots"
    section2_dir = source +"2_zero_currents_field_plots"
    subsection21_dir = section2_dir + "/21_constant_field_fixed_current"
    subsection22_dir = section2_dir + "/22_varying_field_zero_current"
    subsection23_dir = section2_dir + "/23_fixed_field_zero_current_different_heights"
    clean_directory(section1_dir)
    clean_directory(subsection21_dir)
    clean_directory(subsection22_dir)
    clean_directory(subsection23_dir)

def save_data(data,file_path, column_titles):
    np.savetxt(file_path, np.column_stack((data[0],data[1])),header=column_titles)

def clean_temp_files():
    """
    Scans the system %temp% directory for folders starting with 'electro2'
    and safely deletes them to free up space.
    """
    # 1. Get the system temp directory path (works on Windows, Mac, Linux)
    temp_dir = tempfile.gettempdir()
    print(f"Scanning directory: {temp_dir} ...")

    deleted_count = 0
    errors = 0

    # 2. List all items in the temp directory
    try:
        items = os.listdir(temp_dir)
    except Exception as e:
        print(f"Error accessing temp directory: {e}")
        return

    # 3. Iterate and filter
    for item in items:
        # Check if name starts with 'electro2'
        if item.startswith("electro2"):
            full_path = os.path.join(temp_dir, item)
            
            # Ensure we are deleting a directory, not a random file
            if os.path.isdir(full_path):
                try:
                    # shutil.rmtree removes the directory and all its contents
                    shutil.rmtree(full_path)
                    print(f"✅ Deleted: {item}")
                    deleted_count += 1
                except PermissionError:
                    print(f"⚠️  Skipped (Locked/Permission Denied): {item}")
                    errors += 1
                except Exception as e:
                    print(f"❌ Error deleting {item}: {e}")
                    errors += 1

    # 4. Final Report
    print("-" * 30)
    if deleted_count == 0 and errors == 0:
        print("No 'electro2' directories found.")
    else:
        print(f"Cleanup complete.")
        print(f"Deleted: {deleted_count}")
        print(f"Skipped: {errors}")

