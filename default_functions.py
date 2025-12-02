from tdgl.visualization.animate import create_animation
from IPython.display import HTML, display
from IPython.display import clear_output
from tdgl.sources import ConstantField
from tdgl.geometry import box, circle
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import h5py
import tdgl
import time
import os
###################################################################
#This script contains the default parameters and functions used in the notebook.#
###################################################################
# ====================================================
## 1. ⚙️ Global Parameters(Optimized)
# ====================================================

### 1.1. Material parameters
LENGTH_UNITS = "um"
xi = 0.5           
london_lambda = 2  
d = 0.1               
gamma = 10             
  # Ancho del puente (dimensión x)
 # Alto del puente (dimensión y)

STRIPE_LENGTH = 0.01   # side of the square contact
OVERLAP = 0.5         # aditional space for overlapping


### 1.3. Parámetros de Malla
MAX_EDGE_LENGTH_IV = xi / 1.5
MAX_EDGE_LENGTH_VORTEX = xi / 1.5
SMOOTHING_STEPS = 100  

# ====================================================
## 2. ⚙️ videos and default functions
# ====================================================  
os.environ["OPENBLAS_NUM_THREADS"] = "1"
MAKE_ANIMATIONS = True
tempdir = tempfile.TemporaryDirectory()
def make_video_from_solution(
    solution,
    quantities=("order_parameter", "phase"),
    fps=20,
    figsize=(5, 4),
):
    """Generates an HTML5 video from a tdgl.Solution."""
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(
                h5file,
                quantities=quantities,
                fps=fps,
                figure_kwargs=dict(figsize=figsize),
            )
            video = anim.to_html5_video()
        return HTML(video)
def default_options(d_filename,skip_t=200,solve_t=200,saves=200):
    options =  tdgl.SolverOptions(
    skip_time=skip_t,  # initial relaxation time
    solve_time=solve_t,  # Real simulation time
    output_file=os.path.join(tempdir.name,d_filename),  # file route
    field_units="mT",  #Units of the applied field (miliTesla)
    current_units="uA",  # Units of the applied current (microamperios)
    save_every=saves,
    )
    return options
    
def default_solution(device,file_name,terminal_currents_applied,vector_potential=0):
    '''
    This function allows the user to apply different solution cases based on the applied current/field 
    :param device: tdgl.device object
    :param type_of_solution: String
    :param vector_potential: Double
    :param terminal_currents: unidimensional array [source_current,drain_current]
    Depending on the type_of_solution a solution is implemented
    '''
    options = default_options(file_name)
    external_field = ConstantField(vector_potential, field_units=options.field_units, length_units=device.length_units)
    solution = tdgl.solve(
         device,
         options,
         terminal_currents= terminal_currents_applied,
         applied_vector_potential=external_field
        )
    return solution
# ====================================================
# 3.)Default configuration
# ====================================================
H5_DIR = "./project_field_h5_files"
os.makedirs(H5_DIR, exist_ok=True)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Configuración de gráficas
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['lines.linewidth'] = 2.0
# ====================================================
# 4.)Device Creation Function
# ====================================================
def plot_parameters(p1,p2,plot_labels,plot_type="plot",color_applied="teal",dir_path = None):
    plt.figure(figsize=(6, 4))
    if plot_type == "plot":
        plt.plot(p1,p2, "o-", color=color_applied)
    elif plot_type == "scatter":
        plt.scatter(p1,p2,color=color_applied,s=3)
    else:
        print("insert a valid plot type")
        return None
    plt.xlabel(plot_labels["x"])
    plt.ylabel(plot_labels["y"])
    plt.title(plot_labels["title"])
    plt.grid(True)
    plt.show()
    if dir_path != None:
        plt.savefig(dir_path)
# ====================================================
## 4. ⚙️ Global Parameters(Optimized)
# ====================================================

def create_device(geometry_added,layer,max_edge_length,dimensions,translationx=0,incrementx=0.0,incrementy=0.0,translationy=0):
    '''
    Since we're using the same geometry, this function is implemented so we can change the dimensions of the right rectangle hence the position of the drain too : tdgl.polygon object,
    :param geometry_used: tdgl.polygon object,
    :param geometry_added: tdgl.polygon object,
    :param max_edge_length:int,
    :param incrementx:float
    :param incrementy:float
    :param source_dimension: array [width,height]
    :param drain_dimension: array [width,height]
    :param translationx:float
    :param translationy:float
    The translations move the source and drain along the polygon
    '''
    width_x = dimensions['width_x']
    #height_y = dimensions['height_y']
    width_x2 = dimensions['width_x2']
    height_y2 = dimensions['height_y2']
    #for points that are the same as the  film width
    real_size_x = width_x2 + incrementx
    real_size_y =3 + incrementy
    film_poly_up = tdgl.Polygon("film_pequeño", points=box(width=real_size_x, height= real_size_y)).translate(dx=+width_x/2)
    combined_geometry = geometry_added.union(film_poly_up)
    #Source
    source_poly = tdgl.Polygon(
        "source", 
        points=box(width=STRIPE_LENGTH,height=height_y2)
    ).translate(dx=-translationx).translate(dy=translationy)
    #Drain 
    drain_poly = tdgl.Polygon(
        "drain", 
        points=box(width=STRIPE_LENGTH, height=real_size_y)
    ).translate(dx=+translationx+incrementx/2).translate(dy=+translationy)
    combined_film = tdgl.Polygon.from_union([combined_geometry, source_poly, drain_poly], name="film")
    probe_points = [(width_x / 3,0), (-width_x / 3,0)]
    device = tdgl.Device(
        "vertical_bridge",
        layer=layer,
        film=combined_film,
        holes=[],
        terminals=[source_poly, drain_poly],
        probe_points=probe_points,
        length_units=LENGTH_UNITS,
    )
    device.make_mesh(max_edge_length=max_edge_length, smooth=SMOOTHING_STEPS)
    #Remove to se more details about the mesh 
    #There are 4 malformed cells as of now , 4/5030 
    clear_output(wait=True)
    print(f"  Malla creada: {len(device.mesh.sites)} puntos")
    fig, ax = device.draw(figsize=(10, 4))
    return device


def plot_solution(solution,order_title = None,current_title = None,currentBool = True,order_path= None,current_path= None):
    '''
    Graphs the applied current on the device and the pahse for a fixed current/constant field 
    :param solution: tdgl.solution object
    :param order_title: String, title for the order parameter plot
    :param current_title: String, title for the current plot
    :param currentBool: Boolean, if True plots the currents
    :param order_path: String, path to save the order parameter plot
    :param current_path: String, path to save the current plot
    returns: None
    '''
    #The plot_solution is only used on the 1st simulation section
    # Create figure with adjusted spacing and plot currents
    
    if currentBool == True:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Wider figure (10 inches width)
        if current_title == None:
            _ = solution.plot_currents(ax=axes[0], streamplot=False)
            _ = solution.plot_currents(ax=axes[1])  
        else:
            _ = solution.plot_currents(ax=axes[0], streamplot=False,title=current_title)
            _ = solution.plot_currents(ax=axes[1],title = current_title)  
        plt.subplots_adjust(wspace=0.4)  # Increase horizontal space between subplots
        plt.tight_layout()  # Automatically adjusts subplots to fit in figure
        plt.show()
        if current_path != None:
            fig.savefig(current_path)
    #Second plot
    # Plot a snapshot of the order parameter in the middle of a phase slip
    t0 = 155
    solution.solve_step = solution.closest_solve_step(t0)
    if order_title == None:
        fig, axes = solution.plot_order_parameter(figsize=(10, 4))
    else:
        fig, axes = solution.plot_order_parameter(figsize=(10, 4),subtitle = order_title)
    plt.show()
    if order_path != None:
            
            fig.savefig(order_path)

def plot_group(solution,figure_size,used_titles,currentBool= True,titleBool=True,order_path= None,current_path= None):
    '''
    Graphs a group of plots including the current, order parameter, vorticity and scalar potential
    :param solution: tdgl.solution object   
    :param figure_size: Tuple, size of the figures
    :param used_titles: Dictionary with titles for each plot
    :param currentBool: Boolean, if True plots the currents
    :param titleBool: Boolean, if True uses titles for the plots
    :param order_path: String, path to save the order parameter plot
    :param current_path: String, path to save the current plot
    returns: None
    '''
    if titleBool == True:
        plot_solution(solution,currentBool=currentBool,order_title=used_titles["order_parameter"],current_title=used_titles["sheet_current"],order_path=order_path,current_path=current_path)
        solution.plot_vorticity(figsize=figure_size,title=used_titles["vorticity"])
        solution.plot_scalar_potential(figsize=figure_size,title=used_titles["scalar_potential"])
    else:
        plot_solution(solution,currentBool=currentBool,order_path=order_path,current_path=current_path)
        solution.plot_vorticity(figsize=figure_size)
        solution.plot_scalar_potential(figsize=figure_size)

# =========================
# 5) Magnetization function
# =========================
def solve_field(device,field,d=0.1):
    '''
    A function that applies a magnetic field sweep to a device and returns the corresponding magnetizations and magnetic moments.
    
    :param device: tdgl.device object
    :param field: List or array of magnetic field values to be applied.
    :param d: Double, depth of the superconductor in micrometers (default is 0.1 µm).
    :return: Two lists containing the total magnetic moments and volumetric magnetizations for each applied
    '''
    d = 0.1  # Superconductor depth in micrometers (µm)
    area = np.sum(device.areas)  # effective area of the device ( µm²)
    # =========================
    # 3) External Field sweep
    # =========================
    # Defines a list of 10 values for the external magnetic field form 0 to 1 mT
    moments = [] #total magnetic moment
    magnetizations = []  # volumetric magnetization
    # Loop for each value of B
    currents = {
        "source": 0.0,
        "drain": 0.0
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        for B in field:
            # Creates a uniform magnetic field of magnitude B
            # Solves Ginzburg–Landau equations with an applied field
            solution_field= default_solution(device,"Bscan.h5",vector_potential=B,terminal_currents_applied=currents)
            #Calculates total magnetic moment (uA · µm²) 
            m = solution_field.magnetic_moment(units="uA * um**2", with_units=False)
            moments.append(m)  # Almacena el valor
            # Calculates volumetric magnetization: M = m / (Area ×depth)  in µA / µm³
            M = m / (area * d)
            magnetizations.append(M)
            
    # =========================
    # 4) Susceptibility dM/dB
    # =========================
    magnetizations = np.array(magnetizations)
    # Numeric derivation of the magnetization with respect to the field: dM/dB
    suceptibility = np.gradient(magnetizations, field)
    # =========================
    # 5)Save data on files
    # =========================
    np.savetxt("magnetization_vs_B.txt", np.column_stack((field, magnetizations)),header="B[mT] M[uA/um^3]")
    np.savetxt("suceptibilidad_rr_vs_B.txt", np.column_stack((field, suceptibility)),header="B[mT] dM/dB [uA/(um^3·mT)]")
    return moments,magnetizations, suceptibility
def find_resistance(currents,voltages):
    '''
    This function calculates the resistance at each point in the IV curve by computing the gradient of voltage with respect to current.
    Parameters:
    :param currents (np.array): Array of current values.
    :param voltages (np.array): Array of voltage values corresponding to the currents.
    Returns:
    np.array: Array of resistance values calculated as dV/dI.
    '''
    dV_dI = np.gradient(voltages, currents)
    return dV_dI
    
def current_application(device,currents,B_field = 0):
    '''
    A function that applies a current sweep to a device and returns the corresponding voltages.
    
    :param device: tdgl.device object
    :param currents: List or array of current values to be applied.
    :param B_field: Double, optional magnetic field to be applied (default is 0).
    '''

    voltages = []
    # =========================================================
    # Simulation
    # =========================================================
    start_time = time.time()
    total_simulations = len(currents)
    j=0
    with tempfile.TemporaryDirectory() as temp_dir:
        for I in currents:
            applied_currents = {
                "source": I,
                "drain": -I
            }
            solution_c = default_solution(
            device,
            f"solution_I_{I:.1f}.h5",
            vector_potential=B_field,
            terminal_currents_applied=applied_currents,
           )
            dynamics = solution_c.dynamics
            indices = dynamics.time_slice(tmin=120)
            voltage = np.abs(np.mean(dynamics.voltage()[indices]))
            voltages.append(voltage)
            j+=1
            print(f"I = {I:.1f} µA, <V> = {voltage:.4f} V₀,progress: {np.round(j/np.size(currents)*100,2)}%", end='\r')
        
    resistances = find_resistance(currents,voltages)
    clear_output(wait=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time/60
    print(" " * 60, end='\r') 
    print("-" * 50)
    print(f"✅ The simulation was completed with {total_simulations} steps.")
    print(f"⏱️ The elapsed time was: {elapsed_time:.2f} seconds.")
    print(f"⏱️ The elapsed time was: {elapsed_minutes:.2f} minutes.")
    print(f"📊 Tiempo mean time per step was: {(elapsed_time / total_simulations):.2f} seconds.")
    print("-" * 50)
    return voltages,resistances
#Possible critic currents step optimizer function

def critic_guess(currents,voltages,delta):
    '''
    This function estimates the critical regions in the IV curve where the voltage changes rapidly with respect to the current.
    It uses the gradient of the voltage with respect to the current to identify these regions based on a threshold defined by the delta parameter.
    Parameters:
    :param currents (np.array): Array of current values.
    :param voltages (np.array): Array of voltage values corresponding to the currents.
    :param delta (float): A scaling factor to determine the threshold for identifying critical regions.
    Returns:
    np.array: Array of critical current values where significant voltage changes occur.
    '''
    dV_dI = find_resistance(currents,voltages)
    threshold = delta * np.max(dV_dI)
    critic_regions = currents[dV_dI > threshold]
    #print(f'for {delta} the size is {np.size(critic_regions)}')
    return critic_regions

def find_critic_regions(currents,voltages,quantity=4,jo=0.5):
    '''
    This function finds critical regions in the IV curve by iteratively adjusting a delta parameter until the desired number of critical regions is found.
    It calls the critic_guess function to identify critical regions based on the gradient of voltage with respect to current.
    Parameters:
    :param currents (np.array): Array of current values.
    :param voltages (np.array): Array of voltage values corresponding to the currents.
    :param quantity (int): Desired number of critical regions to find (default is 4).
    :param jo (float): Initial delta value to start the search (default is 0.5).
    Returns:
    np.array: Array of critical current values where significant voltage changes occur.
    '''
    j = jo
    initial_size = quantity + 1
    critic_regions = np.empty(initial_size)
    while np.size(critic_regions) > quantity:
        if (np.size(critic_regions) - quantity) < 0.1 and (np.size(critic_regions) - quantity) > 0:
            break
        if j == 1:
            break
        critic_regions = critic_guess(currents,voltages,j)
        j += 0.01
    return critic_regions

def critic_currents_augmentation(device, critic_regions, current_bounds, B=1.0, steps=3, critic_steps=10, epsilon=0.5):
    '''
    A function that applies a current sweep with more defined calculations around the critic currents.
    :param current_bounds: Dictionary with "initial" and "final".
    '''
    
    # Store the arrays chunks here, then concatenate at the end
    all_currents_chunks = []
    all_voltages_chunks = []
    
    # Start point for the sweep
    current_cursor = current_bounds["initial"]
    final_limit = current_bounds["final"]
    j = 0
    # 1. Loop through critical regions
    for i in critic_regions:
        # Define the critical window
        co = i - epsilon
        cf = i + epsilon
        
        # Ensure we don't go backwards if regions overlap
        if current_cursor < co:
            # We subtract a small offset to ensure we don't duplicate the start of the critical region
            previous_currents = np.linspace(current_cursor, co, steps, endpoint=False)
            
            if len(previous_currents) > 0:
                previous_voltages = current_application(device, previous_currents, B_field=B)
                all_currents_chunks.append(previous_currents)
                all_voltages_chunks.append(previous_voltages)
        
        # --- CRITICAL (FINE) INTERVAL ---
        critic_currents = np.linspace(co, cf, critic_steps)
        critic_voltages = current_application(device, critic_currents, B_field=B)
        
        all_currents_chunks.append(critic_currents)
        all_voltages_chunks.append(critic_voltages)
        j+=1
        print(f'Critical region {j}/{np.size(critic_regions)} at I = {i:.2f} µA processed.')
        
        # Update the cursor to the end of this critical region
        # Using a small offset to avoid overlapping points in the next linspace
        current_cursor = cf
 
        
    # 2. Final Interval (after the last critical region)
    if current_cursor < final_limit:
        final_currents = np.linspace(current_cursor, final_limit, steps)
        final_voltages = current_application(device, final_currents, B_field=B)
        all_currents_chunks.append(final_currents)
        all_voltages_chunks.append(final_voltages)

    # 3. Concatenate all data
    total_currents = np.concatenate(all_currents_chunks)
    total_voltages = np.concatenate(all_voltages_chunks)
    
    # Sort the data just in case intervals overlapped or were out of order
    sort_indices = np.argsort(total_currents)
    total_currents = total_currents[sort_indices]
    total_voltages = total_voltages[sort_indices]
    

    # 4. Plotting (Done once at the end)
    plot_info = {
        "fig_name": "currents.jpg",
        "title": f'Curva Voltaje vs Corriente ({current_bounds["initial"]}–{current_bounds["final"]} µA)',
        "x": "Corriente $I$ [$\mu$A]",
        "y": "Voltaje promedio $\\langle \Delta \\mu \\rangle$ [$V_0$]"
    }
    plot_info2 = {
        "fig_name": "resistances.jpg",
        "title": f'Resistencia vs Corriente ({current_bounds["initial"]}–{current_bounds["final"]} µA)',
        "x": "Corriente $I$ [$\mu$A]",
        "y": "Resistencia $dV/dI$ [$R_0$]"
    }
    total_resistance = find_resistance(total_currents, total_voltages)
    plot_parameters(total_currents, total_voltages, plot_info)
    plot_parameters(total_currents, total_resistance, plot_info2)
    return total_currents, total_voltages,total_resistance


def varying_increments(geometry_used,layer,MAX_EDGE_LENGTH_IV,dimensions,displacement,currents,deltay = 1,field = 1.0):

    '''
    This function applies a current sweep to devices with varying heights and returns the corresponding voltages.
    :param geometry_used: tdgl.polygon
    :param geometry_added: tdgl.polygon
    :param layer: tdgl.layer object
    :param MAX_EDGE_LENGTH_IV: int
    :param dimensions: dictionary with the dimensions of the device
    :param displacement: float, translation value for the source and drain
    :param currents: List or array of current values to be applied.
    :param deltay: float, optional vertical translation for the source and drain (default is 0).
    :param field: Double, optional magnetic field to be applied (default is 0).
    returns: voltages_arr
    '''
    if deltay == 4:
        deltay = 5
    device_l  =create_device(geometry_used,layer,MAX_EDGE_LENGTH_IV,dimensions,translationx=displacement,incrementy=deltay)#
    fig, ax = device_l.plot(mesh=True)
    voltages =  current_application(device_l, currents,B_field = field)
    return voltages

        

