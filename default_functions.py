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
    
def default_solution(device,file_name,vector_potential=0,terminal_currents_applied=[0,0]):
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
         terminal_currents= dict(source=terminal_currents_applied[0],drain=terminal_currents_applied[1]),
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
def plot_parameters(p1,p2,plot_labels,plot_type="plot",color_applied="teal"):
    plt.figure(figsize=(6, 4))
    if plot_type == "plot":
        plt.plot(p1,p2, "o-", color=color_applied)
    elif plot_type == "scatter":
        plt.scatter(p1,p2, "o-", color=color_applied)
    else:
        print("insert a valid plot type")
        return None
    plt.xlabel(plot_labels["x"])
    plt.ylabel(plot_labels["y"])
    plt.title(plot_labels["title"])
    plt.grid(True)
    plt.show()
    plt.savefig(plot_labels["fig_name"])
# ====================================================
## 4. ⚙️ Global Parameters(Optimized)
# ====================================================

def create_device(geometry_used,geometry_added,layer,max_edge_length,dimensions,translationx=0,incrementx=0.0,incrementy=0.0,translationy=0):
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
    height_y = dimensions['height_y']
    width_x2 = dimensions['width_x2']
    height_y2 = dimensions['height_y2']
    #for points that are the same as the  film width
    contact_size = width_x #1 µm squares
    real_size_x = width_x2 + incrementx
    real_size_y =3 + incrementy
    film_poly_up = tdgl.Polygon("film_pequeño", points=box(width=real_size_x, height= real_size_y)).translate(dx=+width_x/2)
    combined_geometry = geometry_used.union(geometry_added,film_poly_up)
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
    fig, ax = device.draw()
    return device
    

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
    m_solutions = []
    # Loop for each value of B

    with tempfile.TemporaryDirectory() as temp_dir:
        for B in field:
            # Creates a uniform magnetic field of magnitude B
            # Solves Ginzburg–Landau equations with an applied field
            solution_field= default_solution(device,"Bscan.h5",vector_potential=B,terminal_currents_applied=[0.0,0.0])
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
    #suceptibilidad_fast = np.gradient(magnetizaciones_fast, campos_fast)
    # =========================
    # 5)Save data on files
    # =========================
    np.savetxt("magnetization_vs_B.txt", np.column_stack((field, magnetizations)),
               header="B[mT] M[uA/um^3]")
    #np.savetxt("suceptibilidad_rr_vs_B.txt", np.column_stack((campos_rr, suceptibilidad_rr)),
               #header="B[mT] dM/dB [uA/(um^3·mT)]")
    return moments,magnetizations

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
            solution_c = default_solution(
            device,
            f"solution_I_{I:.1f}.h5",
            vector_potential=B_field,
            terminal_currents_applied=[I, -I],
           )
            dynamics = solution_c.dynamics
            indices = dynamics.time_slice(tmin=120)
            voltage = np.abs(np.mean(dynamics.voltage()[indices]))
            voltages.append(voltage)
            j+=1
            print(f"I = {I:.1f} µA, <V> = {voltage:.4f} V₀,progress: {np.round(j/np.size(currents)*100,2)}%", end='\r')
        
  
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
    return voltages
#Possible critic currents step optimizer function
def critic_currents_augmentation(device,critic_regions,currents,voltages,B= 0):
    '''
    A function that applies a current sweep to a device and returns the corresponding voltages with more resolution in critic regions.
    
    :param device: tdgl.device object
    :param critic_regions: List or array of current values where more resolution is needed.
    :param currents: List or array of current values to be applied.
    :param voltages: List or array of voltage values corresponding to the currents.
    :param B: Double, optional magnetic field to be applied (default is 0).
    '''
    size = np.size(currents)
    steps = 30
    epsilon = 1
    for i in critic_regions:
        co = i - epsilon
        cf = i + epsilon
        critic_currents = np.linspace(co, cf, steps)
        critic_voltages = current_application(device, critic_currents,B = 0)
        #Masks for the currents
        mask_left = currents <= co 
        mask_right = currents >= cf
        #Masks for the voltages
        maskv1 = voltages <= critic_voltages
        maskv2 = voltages >= critic_voltages
        currents = np.concatenate((currents[mask_left],critic_currents,currents[mask_right]))
        voltages = np.concatenate((voltages[mask_left],critic_voltages[j],voltages[mask_right]))
    plot_info1 = {"fig_name":"currents.jpg","title":f'Curva Voltaje vs Corriente ({currents[0]}–{currents[size]}µA)',"x":"Corriente $I$ [$\mu$A]","y":"Voltaje promedio $\\langle \Delta \\mu \\rangle$ [$V_0$]"}
    plot_parameters(currents,voltages,plot_info1)
def varying_increments(device,currents,io,ifi,field = 0):
    size = ifi - io +1
    voltages_arr = []
    devices_arr = []
    J = 0
    for h in range(io,ifi):      
        deltay = h
        if h == 3:
            deltay = 4
        device_l  =create_device(film_poly,half_geometry,layer,MAX_EDGE_LENGTH_IV,dimensions,translationx=displacement,incrementy=deltay)
        voltages =  current_application(device_l, currents,B_field = field)
        devices_arr.append(device_l)
        voltages_arr.append(voltages)
        J+= 1
        print(f'progress: {np.round((J/size)*100,3)}')

