from tdgl.visualization.animate import create_animation
from shapely.geometry import LineString, Point
from IPython.display import HTML, display
from IPython.display import clear_output
from tdgl.sources import ConstantField
from tdgl.geometry import box, circle
import default_directories as dd
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

tempdir = tempfile.TemporaryDirectory(prefix='electro2_')
H5_DIR = "./project_field_h5_files"
os.makedirs(H5_DIR, exist_ok=True)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Configuración de gráficas
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['lines.linewidth'] = 2.0
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
    fig = plt.gcf()
    if dir_path != None:
        fig.savefig(dir_path,facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.show()


# #################################################################################
# =================================================================================
## Device functions 
# =================================================================================
# #################################################################################
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

    width_x2 = dimensions['width_x2']
    height_y2 = dimensions['height_y2']
    #for points that are the same as the  film width
    real_size_x = width_x2 + incrementx
    real_size_y =3 + incrementy
    film_poly_up = tdgl.Polygon("film_pequeño", points=box(width=real_size_x, height= real_size_y)).translate(dx=+width_x/2)
    combined_geometry = geometry_added.union(film_poly_up)
    combined_film = combined_geometry
    device = tdgl.Device(
        "vertical_bridge",
        layer=layer,
        film=combined_film,
        holes=[],
        length_units=LENGTH_UNITS,
    )
    #device.make_mesh(max_edge_length=max_edge_length, smooth=SMOOTHING_STEPS)
    #Remove to se more details about the mesh
    #There are 4 malformed cells as of now , 4/5030
    clear_output(wait=True)
    #print(f"  Malla creada: {len(device.mesh.sites)} puntos")
    fig, ax = device.draw(figsize=(10, 4))
    return device

def visualize_segments(device,view=True):
    """
    Plots the device boundary with numbered segments and returns the segment list.
    """
    # 1. Extract shell
    points = device.film.points
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # 2. Identify Sides
    segments = segment_boundary(points)
    if view == True:
        # 3. Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        device.plot(ax=ax, mesh=False, legend=False)
        
        for i, seg in enumerate(segments):
            # Calculate Label Position
            mid_idx = len(seg['points']) // 2
            mid_pt = seg['points'][mid_idx]
            
            if seg['type'] == 'circle':
                vec = mid_pt - seg['center']
                norm_vec = vec / np.linalg.norm(vec)
            else:
                p1 = seg['points'][0]
                p2 = seg['points'][-1]
                tangent = p2 - p1
                norm_vec = np.array([-tangent[1], tangent[0]]) 
                norm_vec /= (np.linalg.norm(norm_vec) + 1e-9)

            label_pos = mid_pt + (norm_vec * 0.5) 
            
            color = 'blue' if seg['type'] == 'line' else 'red'
            ax.plot(seg['points'][:,0], seg['points'][:,1], color=color, linewidth=2)
            ax.text(label_pos[0], label_pos[1], str(i), fontsize=12, color='white', 
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

        plt.title(f"Device Geometry: Found {len(segments)} segments")
        plt.show()
    
    # Return the segments so the next function can use them
    return segments
def add_terminals_by_id(device, segments, source_id, drain_id, layer, max_edge_length, width_pct=100, stripe_length=0.01, probe_depth=0.5):
    """
    Creates a new device with terminals added at specified segments.
    Automatically positions probe points near the new terminals.
    
    :param stripe_length: Thickness of the terminal (0.01).
    :param probe_depth: How far inside the film to place the probes (in um).
    """
    new_terminals = list(device.terminals)
    
    if source_id >= len(segments) or drain_id >= len(segments):
        raise ValueError(f"IDs must be less than {len(segments)}")

    # --- 1. Create Source ---
    seg_s = segments[source_id]
    term_s = create_terminal_from_segment(seg_s, "_new_source", pct=width_pct, stripe_length=stripe_length)
    new_terminals.append(term_s)
    
    # Calculate Probe 1 (Source Side)
    probe_s = get_inward_probe_point(seg_s, depth=probe_depth)
    
    # --- 2. Create Drain ---
    seg_d = segments[drain_id]
    term_d = create_terminal_from_segment(seg_d, "_new_drain", pct=width_pct, stripe_length=stripe_length)
    new_terminals.append(term_d)
    
    # Calculate Probe 2 (Drain Side)
    probe_d = get_inward_probe_point(seg_d, depth=probe_depth)

    # --- 3. Update Device Probe Points ---
    # We replace the old probes with these new ones tailored to the current path
    new_probes = np.array([probe_s, probe_d])

    # --- 4. Reconstruct Device ---
    new_film_parts = [device.film] + new_terminals
    new_film = tdgl.Polygon.from_union(new_film_parts, name="film_with_new_terminals")

    new_device = tdgl.Device(
        f"{device.name}_expanded",
        layer=layer,
        film=new_film,
        holes=device.holes,
        terminals=new_terminals,
        probe_points=new_probes, # <--- Updated here
        length_units=device.length_units
    )
    
    print(f"Remeshing with stripe_length={stripe_length} and probe_depth={probe_depth}...")
    new_device.make_mesh(max_edge_length=max_edge_length, smooth=100)
    
    return new_device


def segment_boundary(points, angle_tol=1.0, curve_tol=0.05):
    """
    Divides a list of points into geometric segments (Linear or Circular).
    """
    segments = []
    if len(points) < 2: return segments

    current_segment = [points[0]]
    
    # We iterate through points and check if the 'next' vector maintains the current trend
    # This is a simplified vector analysis.
    
    for i in range(1, len(points) - 1):
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[i+1]
        
        current_segment.append(p_curr)
        
        # Analyze vectors
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        # Angles in degrees
        ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
        ang2 = np.degrees(np.arctan2(v2[1], v2[0]))
        
        diff = abs(ang1 - ang2)
        if diff > 180: diff = 360 - diff
        
        # If angle changes significantly, the segment *might* be ending or it's a curve
        if diff > angle_tol:
            # If the current segment has enough points, we determine its type
            # For this logic, we break whenever there is a sharp corner.
            # Curves in TDGL polygons are usually many small segments with small angle changes.
            # If the change is abrupt (>30 deg), it's definitely a corner.
            if diff > 20: 
                segments.append(classify_segment(np.array(current_segment)))
                current_segment = [p_curr] # Start new segment from corner

    # Add the last bit
    current_segment.append(points[-1])
    segments.append(classify_segment(np.array(current_segment)))
    
    # Post-process: Merge continuous small curve segments if needed, 
    # but for standard TDGL polygons defined by boxes, this is usually sufficient.
    return segments

def classify_segment(pts):
    """Determines if points form a line or a circle arc."""
    if len(pts) <= 2:
        return {'type': 'line', 'points': pts, 'length': np.linalg.norm(pts[-1]-pts[0]), 'angle': get_line_angle(pts)}
    
    # Check linearity: distance of mid points to the line connecting start-end
    p_start = pts[0]
    p_end = pts[-1]
    line_vec = p_end - p_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0: return {'type': 'point', 'points': pts}
    
    line_unit = line_vec / line_len
    
    # Max deviation from straight line
    deviations = []
    for p in pts:
        vec_p = p - p_start
        proj = np.dot(vec_p, line_unit)
        perp_dist = np.linalg.norm(vec_p - proj * line_unit)
        deviations.append(perp_dist)
        
    if max(deviations) < 1e-3: # It's a line
         return {'type': 'line', 'points': pts, 'length': line_len, 'angle': get_line_angle(pts)}
    else:
        # It's likely a curve/circle
        # Fit a circle (simplified: circumcenter of start, mid, end)
        mid = pts[len(pts)//2]
        center, radius = define_circle(p_start, mid, p_end)
        return {'type': 'circle', 'points': pts, 'center': center, 'radius': radius}

def get_line_angle(pts):
    """Returns angle of a line segment."""
    d = pts[-1] - pts[0]
    return np.degrees(np.arctan2(d[1], d[0]))

def define_circle(p1, p2, p3):
    """Find center and radius from 3 points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-9: return np.array([0,0]), 0 # Collinear
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    center = np.array([Ux, Uy])
    radius = np.linalg.norm(center - p1)
    return center, radius
def create_terminal_from_segment(segment, name_suffix, pct=100, stripe_length=0.01):
    """
    Creates a terminal extension attached to the segment with a fixed stripe_length.
    """
    pts = segment['points']
    p_start = pts[0]
    p_end = pts[-1]
    
    # 1. Determine size scaling (width along the edge)
    if pct < 100:
        center_line = (p_start + p_end) / 2
        vec = (p_end - p_start)
        half_vec = vec * (pct / 200.0) 
        p_start = center_line - half_vec
        p_end = center_line + half_vec

    # 2. Determine Outward Normal
    segment_mid = (p_start + p_end) / 2
    
    if segment['type'] == 'line':
        tangent = p_end - p_start
        # Rotate 90 degrees to get a normal
        normal = np.array([-tangent[1], tangent[0]]) 
        normal = normal / np.linalg.norm(normal)
        
        # Ensure normal points OUTWARD (away from center 0,0)
        # We assume the device is centered roughly at (0,0)
        if np.dot(normal, segment_mid) < 0:
            normal = -normal
            
        # Extrude by exactly STRIPE_LENGTH
        extrusion = normal * stripe_length 
        
        # Create box points
        t_pts = [
            p_start,
            p_end,
            p_end + extrusion,
            p_start + extrusion
        ]
        
    elif segment['type'] == 'circle':
        center = segment['center']
        radius = segment['radius']
        # Extrude outward radially by STRIPE_LENGTH
        r_out = radius + stripe_length
        
        ang_start = np.arctan2(p_start[1]-center[1], p_start[0]-center[0])
        ang_end = np.arctan2(p_end[1]-center[1], p_end[0]-center[0])
        
        num_arc_pts = len(pts)
        theta = np.linspace(ang_start, ang_end, num_arc_pts)
        
        outer_arc = []
        for t in theta:
            outer_arc.append(center + np.array([np.cos(t), np.sin(t)]) * r_out)
        outer_arc = np.array(outer_arc)
        
        t_pts = np.vstack([pts, outer_arc[::-1]])

    return tdgl.Polygon(f"term{name_suffix}", points=t_pts)
def get_inward_probe_point(segment, depth=1.0):
    """
    Calculates a point 'depth' units inside the device from the segment center.
    """
    pts = segment['points']
    p_start = pts[0]
    p_end = pts[-1]
    segment_mid = (p_start + p_end) / 2
    
    if segment['type'] == 'line':
        tangent = p_end - p_start
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / np.linalg.norm(normal)
        
        # We want the INWARD normal (towards 0,0)
        # If dot product is positive, it points outward, so flip it
        if np.dot(normal, segment_mid) > 0:
            normal = -normal
            
        return segment_mid + (normal * depth)
        
    elif segment['type'] == 'circle':
        # Move radially towards the center
        center = segment['center']
        vec_to_arc = segment_mid - center
        vec_unit = vec_to_arc / np.linalg.norm(vec_to_arc)
        # Move backwards (inward) from the arc
        return segment_mid - (vec_unit * depth)
def add_multiple_terminals(device, segments, terminal_configs, layer, max_edge_length, width_pct=100, stripe_length=0.01, central_probe_separation=3.0, orientation="horizontal",sep_constant=0.7):
    """
    Adds multiple terminals and REPLACES existing probes with exactly 2 central probes.
    
    :param orientation: "horizontal" (default) aligns probes along X-axis. 
                        "vertical" aligns probes along Y-axis.
    """
    new_terminals = list(device.terminals)
    
    # 1. Add All Requested Terminals
    for config in terminal_configs:
        seg_id = config['id']
        name_suffix = f"_{config['name']}"
        
        if seg_id >= len(segments):
            print(f"Warning: Segment ID {seg_id} out of range. Skipping.")
            continue

        seg = segments[seg_id]
        
        # Create Terminal Polygon
        term_poly = create_terminal_from_segment(
            seg, 
            name_suffix, 
            pct=width_pct, 
            stripe_length=stripe_length
        )
        new_terminals.append(term_poly)

    # 2. Probe Placement (Central Intersection)
    all_points = np.vstack([s['points'] for s in segments])
    device_center = np.mean(all_points, axis=0)
    
    # Determine direction based on orientation parameter
    if orientation.lower() == "vertical":
        # Align along Y-axis
        flow_dir = np.array([0.0, 1.0])
    else:
        # Default: Align along X-axis (Horizontal)
        flow_dir = np.array([1.0, 0.0])

    # Place probes
    # Using the 0.7 multiplier from your snippet
    p1 = device_center - (flow_dir * (central_probe_separation * sep_constant))
    p2 = device_center + (flow_dir * (central_probe_separation * sep_constant))
    
    # THIS is the key: we only put these 2 into the array
    final_probes = np.array([p1, p2])

    # 3. Reconstruct Device
    new_film_parts = [device.film] + new_terminals
    new_film = tdgl.Polygon.from_union(new_film_parts, name="film_with_extra_terminals")

    new_device = tdgl.Device(
        f"{device.name}_multi_term",
        layer=layer,
        film=new_film,
        holes=device.holes,
        terminals=new_terminals,
        probe_points=final_probes,
        length_units=device.length_units
    )
    
    # 4. Remesh & Plot
    print(f"Remeshing... Probes placed {central_probe_separation}um apart at center ({orientation}).")
    new_device.make_mesh(max_edge_length=max_edge_length, smooth=100)
    
    fig, ax = new_device.plot(mesh=True)
    
    # Visual confirmation of probes
    ax.scatter(final_probes[:,0], final_probes[:,1], c='red', marker='x', s=100, label='Probes', zorder=10)
    
    return new_device
# =================================================================================
## Simulation functions 
# ================================================================================

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
        fig = plt.gcf()
        if current_path != None:
            
            fig.savefig(current_path,facecolor='white', bbox_inches='tight', pad_inches=0)
        plt.show()
     
    #Second plot
    # Plot a snapshot of the order parameter in the middle of a phase slip
    t0 = 155
    solution.solve_step = solution.closest_solve_step(t0)
    if order_title == None:
        fig, axes = solution.plot_order_parameter(figsize=(10, 4))
    else:
        fig, axes = solution.plot_order_parameter(figsize=(10, 4),subtitle = order_title)
    fig = plt.gcf()
    if order_path != None:        
            fig.savefig(order_path,facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.show()

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
def solve_field(device,field,file_path,d=0.1):
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
    with tempfile.TemporaryDirectory(prefix="electro2_", suffix="_data") as temp_dir:
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
    mag_path = os.path.join(file_path,"magnetization_vs_B.txt")
    susc_path = os.path.join(file_path,"suceptibilidad_vs_B.txt")
    np.savetxt(mag_path, np.column_stack((field, magnetizations)),header="B[mT] M[uA/um^3]")
    np.savetxt(susc_path, np.column_stack((field, suceptibility)),header="B[mT] dM/dB [uA/(um^3·mT)]")
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
    
def current_application(device,currents,file_path,B_field = 0):
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
    with tempfile.TemporaryDirectory(prefix='electro2_') as temp_dir:
        for I in currents:
            filename = f'solution_I_{I:.1f}.h5'
            applied_currents = {
                "term_s": I,
                "term_d": -I
            }
            solution_c = default_solution(
            device,
            filename,
            vector_potential=B_field,
            terminal_currents_applied=applied_currents,
           )
            dynamics = solution_c.dynamics
            indices = dynamics.time_slice(tmin=120)
            voltage = np.abs(np.mean(dynamics.voltage()[indices]))
            voltages.append(voltage)
            j+=1
            print(f"I = {I:.1f} µA, <V> = {voltage:.4f} V₀,progress: {np.round(j/np.size(currents)*100,2)}%", end='\r')
            if os.path.exists(filename):
                     os.remove(filename)
    resistances = find_resistance(currents,voltages)
    dd.save_data((currents,voltages),file_path+'/voltage_vs_current.txt',"currents(µA)  Voltages(V0)")
    dd.save_data((currents,resistances),file_path +'/resistance_vs_current.txt',"currents(µA)  resistances(R0)")
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


def varying_increments(geometry_used,layer,MAX_EDGE_LENGTH_IV,dimensions,displacement,currents,file_path,deltay = 1,field = 1.0,terminals = [8,2],length=0.3):

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
    
# 2. Define your list of terminals
# Example: A 4-terminal measurement setup

    my_terminals = [
        {"id": terminals[0],  "name": "s"},
        {"id": terminals[1], "name": "d"}
    ]
    segments_found = visualize_segments(device_l,view = False)

# 3. Create the device
# This will now work and place probes at the first 2 terminals automatically
    device_final = add_multiple_terminals(
        device_l, 
        segments_found, 
        my_terminals, 
        layer, 
        MAX_EDGE_LENGTH_IV,
        stripe_length=length
    )
    voltages,resistance =  current_application(device_final, currents,file_path,B_field = field)
    return voltages,resistance

        

