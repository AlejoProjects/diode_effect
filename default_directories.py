import numpy as np
import os

def check_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def create_default_directories():
    source = "./figures/"
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
def save_data(data,file_path, column_titles):
    np.savetxt(file_path, np.column_stack((data[0],data[1])),header=column_titles)