from tdgl.visualization.animate import create_animation
from scipy.signal import find_peaks, savgol_filter
from shapely.geometry import LineString, Point
from IPython.display import HTML, display
from IPython.display import clear_output
from tdgl.sources import ConstantField
from tdgl.geometry import box, circle
import default_directories as dd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import tempfile
import string
import tdgl
import time
import os
def create_terminal_dictionary(terminals,names):
        my_terminals = []
        for term_id, term_name in zip(terminals, names):
            my_terminals.append({"id": term_id, "name": term_name})
        return my_terminals
def create_dictionary(ids_names,values):
        my_dict = {}
        for id_name, value in zip(ids_names, values):
            my_dict[id_name] = value
        return my_dict
# #################################################################################
# =================================================================================
## Device functions 
# =================================================================================
# #################################################################################
def create_device(geometry_added,layer,dimensions,incrementx=0.0,incrementy=0.0,translationy=0,device_view=True,clear_device_view=True):
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
        length_units="um",
    )
    #device.make_mesh(max_edge_length=max_edge_length, smooth=smoothing_steps)
    #Remove to se more details about the mesh
    #There are 4 malformed cells as of now , 4/5030
    if clear_device_view == True:
        clear_output(wait=True)
    #print(f"  Malla creada: {len(device.mesh.sites)} puntos")
    if device_view == True:
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
def add_multiple_terminals(device, segments, terminal_configs, layer, max_edge_length, width_pct=100, stripe_length=0.01, central_probe_separation=3.0, orientation="horizontal",sep_constant=0.7,view_device=True):
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
    if view_device == True:
        fig, ax = new_device.plot(mesh=True)
    
         # Visual confirmation of probes
        ax.scatter(final_probes[:,0], final_probes[:,1], c='red', marker='x', s=100, label='Probes', zorder=10)
    
    return new_device


class building :
    def __init__(self,name,geometry,layer,terminals,terminal_names,gp,segments_found):
        self.name = name
        self.geometry = geometry
        self.layer = layer
        self.terminals = terminals
        self.terminal_names = terminal_names
        self.terminal_dict = create_terminal_dictionary(terminals,terminal_names)
        self.tempdir = tempfile.TemporaryDirectory()
        self.gp = gp
        max_edge_length_iv =gp["xi"] / 1.5
        self.max_edge_length_iv=  max_edge_length_iv
        self.max_edge_length_vortex = max_edge_length_iv
        self.smoothing_steps = 100
        device_prototype = create_device(geometry,layer,gp["dimensions"],incrementx=0,incrementy=0)

        # --- FIX START ---
        # 1. Create the configuration list of dictionaries
        terminal_configs = []
        for t_id, t_name in zip(terminals, terminal_names):
            terminal_configs.append({"id": t_id, "name": t_name})

        # 2. Pass 'terminal_configs' instead of 'terminals'
        self.default_device = add_multiple_terminals(
            device_prototype,
            segments_found,
            terminal_configs,  # <--- CHANGED FROM terminals TO terminal_configs
            layer,
            max_edge_length_iv,
            stripe_length=gp["stripe_length"],
            orientation=gp["orientation"],
        )

 
    def default_options(self,d_filename,skip_t=200,solve_t=200,saves=200):
   
        options =  tdgl.SolverOptions(
        skip_time=skip_t,  # initial relaxation time
        solve_time=solve_t,  # Real simulation time
        output_file=os.path.join(self.tempdir.name,d_filename),  # file route
        field_units=self.gp["field_units"],  #Units of the applied field (miliTesla)
        current_units=self.gp["current_units"],  # Units of the applied current (microamperios)
        save_every=saves,
        )
        return options
    
    def default_solution(self, file_name, terminal_currents_applied, device=None, vector_potential=0):
        '''
        This function allows the user to apply different solution cases based on the applied current/field 
        '''
        if device is None:
            device = self.default_device
            
        options = self.default_options(file_name)
        
        external_field = ConstantField(
            vector_potential, 
            field_units=options.field_units, 
            length_units=self.gp["length_units"]
        )
        
        solution = tdgl.solve(
            device,
            options,
            applied_vector_potential=external_field,
            terminal_currents=terminal_currents_applied
            
        )
        return solution
    ######################################################################################
    #1) PLOTS
    ######################################################################################     
    def plot_solution(self, solution, order_title=None, current_title=None, currentBool=True, order_path=None, current_path=None, view=True):
            '''
            Graphs the applied current on the device and the phase for a fixed current/constant field 
            '''
            # The plot_solution is only used on the 1st simulation section
            
            if currentBool:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Wider figure
                
                # --- LEFT PLOT: Magnitude (Heatmap) ---
                if current_title is None:
                    solution.plot_currents(ax=axes[0], streamplot=False)
                else:
                    solution.plot_currents(ax=axes[0], streamplot=False, title=current_title)
                    
                # --- RIGHT PLOT: Streamlines (Lines) ---
                # FIX: Explicitly set streamplot=True to ensure lines appear
                if current_title is None:
                    solution.plot_currents(ax=axes[1], streamplot=True)  
                else:
                    solution.plot_currents(ax=axes[1], streamplot=True, title=current_title)  
                
                plt.subplots_adjust(wspace=0.4)
                plt.tight_layout()
                fig = plt.gcf()
                
                if current_path is not None:
                    fig.savefig(current_path, facecolor='white', bbox_inches='tight', pad_inches=0)
                
                if view:
                    plt.show()
                else:
                    plt.close()

            # Second plot: Order Parameter
            # Plot a snapshot of the order parameter in the middle of a phase slip
            t0 = 155
            # Ensure we don't crash if t0 is out of bounds
            if solution.solve_step is not None:
                solution.solve_step = solution.closest_solve_step(t0)
            
            if order_title is None:
                fig, axes = solution.plot_order_parameter(figsize=(10, 4))
            else:
                fig, axes = solution.plot_order_parameter(figsize=(10, 4), subtitle=order_title)
                
            fig = plt.gcf()
            if order_path is not None:        
                fig.savefig(order_path, facecolor='white', bbox_inches='tight', pad_inches=0)
                
            if view:
                plt.show()
            else:
                plt.close()
    
  
    def plot_group(self, solution, figure_size, used_titles, currentBool=True, titleBool=False, order_path=None, current_path=None, view=False):
            '''
            Graphs a group of plots including the current, order parameter, vorticity and scalar potential
            '''
            if titleBool:
                # FIX: Added self.
                self.plot_solution(
                    solution,
                    currentBool=currentBool,
                    order_title=used_titles["order_parameter"],
                    current_title=used_titles["sheet_current"],
                    order_path=order_path,
                    current_path=current_path,
                    view=view
                )
                if view:
                    solution.plot_vorticity(figsize=figure_size, title=used_titles["vorticity"])
                    solution.plot_scalar_potential(figsize=figure_size, title=used_titles["scalar_potential"])
            else:
                # FIX: Added self.
                self.plot_solution(
                    solution, 
                    currentBool=currentBool, 
                    order_path=order_path, 
                    current_path=current_path, 
                    view=view
                )
                if view:
                    solution.plot_vorticity(figsize=figure_size)
                    solution.plot_scalar_potential(figsize=figure_size)
    
    def plot_parameters(self,p1,p2,plot_labels,plot_type="plot",color_applied="teal",dir_path = None):
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
        
            if dir_path != None:
                plt.savefig(dir_path,facecolor='white', bbox_inches='tight', pad_inches=0)
            plt.show()
        #################################################################################################
        # Custom Plots 
        # #################################################################################################
    def plot_order_parameter(self,
        solution: tdgl.Solution,
        squared: bool = False,
        subtitle: str = "Order Parameter",
        mag_cmap: str = "viridis",
        phase_cmap: str = "twilight_shifted",
        shading: str = "gouraud",
        **kwargs,
    ) :
        """Plots the magnitude (or the magnitude squared) and
        phase of the complex order parameter, :math:`\\psi=|\\psi|e^{i\\theta}`.

        .. seealso:

            :meth:`tdgl.Solution.plot_order_parameter`

        Args:
            solution: The solution for which to plot the order parameter.
            squared: Whether to plot the magnitude squared, :math:`|\\psi|^2`.
            mag_cmap: Name of the colormap to use for the magnitude.
            phase_cmap: Name of the colormap to use for the phase.
            shading: May be ``"flat"`` or ``"gouraud"``. The latter does some interpolation.

        Returns:
            matplotlib Figure and an array of two Axes objects.
        """
        kwargs.setdefault("figsize", (8, 3))
        kwargs.setdefault("constrained_layout", True)
        device = solution.device
        psi = solution.tdgl_data.psi
        mag = np.abs(psi)
        if squared:
            mag = mag**2
        phase = np.angle(psi) / np.pi
        points = device.points
        triangles = device.triangles
        fig, axes = plt.subplots(1, 2, **kwargs)

        im = axes[0].tripcolor(
            points[:, 0],
            points[:, 1],
            mag,
            triangles=triangles,
            vmin=0,
            vmax=1,
            cmap=mag_cmap,
            shading=shading,
        )
        cbar = fig.colorbar(im, ax=axes[0])
        im = axes[1].tripcolor(
            points[:, 0],
            points[:, 1],
            phase,
            triangles=triangles,
            vmin=-1,
            vmax=1,
            cmap=phase_cmap,
            shading=shading,
        )
        cbar = fig.colorbar(im, ax=axes[1])
        length_units = device.ureg(device.length_units).units
        for ax in axes:
            ax.set_aspect("equal")
        return fig, axes
            

    def plot_phase_gradient(self,solution, ax=None):
        """
        Calculates and plots the magnitude of the Phase Gradient.
        Physically, this represents the superfluid velocity.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()

        # 1. Get Data
        # J_s is the supercurrent density
        J_s = solution.supercurrent_density 
        # psi is the order parameter
        psi = solution.tdgl_data.psi
        
        # 2. Calculate Density |psi|^2
        rho = np.abs(psi)**2
        
        # 3. Calculate Phase Gradient Magnitude
        # From GL Theory: J_s ~ rho * grad(phi)  ->  grad(phi) ~ J_s / rho
        # We calculate the magnitude of the current vector
        J_mag = np.linalg.norm(J_s, axis=1)
        
        # We divide J by rho to get the velocity (phase gradient)
        # We add a tiny epsilon (1e-6) to rho to avoid dividing by zero in vortex cores
        phase_grad_mag = J_mag / (rho + 1e-6)
        
        # 4. Mask Vortex Cores
        # Inside a vortex, density is 0, so the gradient is mathematically infinite/undefined.
        # We mask these out for a cleaner plot.
        phase_grad_mag[rho < 0.05] = np.nan

        # 5. Plot
        device = solution.device
        x, y = device.points[:, 0], device.points[:, 1]
        triangles = device.triangles

        im = ax.tripcolor(
            x, y, triangles, 
            phase_grad_mag, 
            shading="gouraud", 
            cmap="plasma",
            vmax=np.nanpercentile(phase_grad_mag, 95) # Auto-scale to ignore spikes
        )
        
        # Add Colorbar
        cbar = plt.colorbar(im, ax=ax)
        #cbar.set_label(r"$|\nabla \phi|$ (Superfluid Velocity)")
        

        ax.set_aspect("equal")   
        return fig, ax


    def plot_parameter_sweep(self,solutions, labels, title="", order_path=None, orientation="vertical",cbar_y_offset=0.0,cbar_x_offset=0.0,c_value="J=0.0mA , y = 3.0$mu_ m$"):
        """
    Creates a grid of plots showing the order parameter (magnitude and phase)
        parameters:
        params:solutions: Object list of tdgl.Solution
        params:labels: List of strings with labels for each solution
        params:title: String, title for the entire figure   
        params:order_path: String, path to save the figure
        params:orientation: String, "vertical" or "horizontal" layout
        Returns:
        None  
        """
        n_sols = len(solutions)
        
        # NEW CHECK: Prevent crash if list is empty
        if n_sols == 0:
            raise ValueError("The 'solutions' list is empty. Please provide at least one solution to plot.")

        if len(labels) != n_sols:
            raise ValueError(f"El número de etiquetas ({len(labels)}) debe coincidir con las soluciones ({n_sols}).")

        
        if len(labels) != n_sols:
            raise ValueError(f"El número de etiquetas ({len(labels)}) debe coincidir con las soluciones ({n_sols}).")

        # --- Plot dimensions  ---
        if orientation == "vertical":
            n_rows = n_sols
            n_cols = 2
            figsize = (5, 2.0 * n_rows)
        else: # horizontal
            n_rows = 2
            n_cols = n_sols
            #Amount of columns
            figsize = (1 * n_cols, 4.0)
        
        # --- CREACIÓN DE LA FIGURA ---
        # wspace=0.0 es CRÍTICO para pegar las columnas (distancia horizontal 0)
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=figsize, 
            constrained_layout=False,
            gridspec_kw={'wspace': 0.0, 'hspace': 0.05}
        )
        
        # Remove external margins
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0.05)
        
        # Indexable axes on all cases
        if n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1 and n_rows > 1:
            axes = axes.reshape(-1, 1)
        elif n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        
        alphabet = string.ascii_lowercase
        
        #Colorbar references
        im_psi_ref = None
        im_phase_ref = None

        # Índice para saber dónde dibujar la colorbar (última simulación en horizontal, fila específica en vertical)
        cbar_idx_vert = 1 if n_sols > 1 else 0

        # --- Principal loop ---
        for i, (sol, label_text) in enumerate(zip(solutions, labels)):
            device = sol.device
            x, y = device.points[:, 0], device.points[:, 1]
            triangles = device.triangles
            
            # Data
            psi = sol.tdgl_data.psi
            if hasattr(psi, "magnitude"): psi = psi.magnitude
            rho = np.abs(psi)**2
            phase = np.angle(psi) / np.pi
            
            # ---Axes selection ---
            # Vertical: axes[i, 0]=Psi, axes[i, 1]=Fase
            # Horizontal: axes[0, i]=Psi, axes[1, i]=Fase
            if orientation == "vertical":
                ax_psi = axes[i, 0]
                ax_phase = axes[i, 1]
            else:
                ax_psi = axes[0, i]
                ax_phase = axes[1, i]
            
            # --- GRAPH PSI ---
            im_psi = ax_psi.tripcolor(x, y, triangles, rho, shading="gouraud", cmap="viridis", vmin=0, vmax=1)
            if i == 0: im_psi_ref = im_psi 
            ax_psi.set_aspect("equal")
            ax_psi.axis('off')
            
            # --- A) B) C) ---
            # En horizontal, 'i' avanza por columnas, así que la etiqueta va arriba de cada columna
            seq_char = alphabet[i]
            combined_label = f"{seq_char}) {label_text}"
            
            ax_psi.text(
                0.0, 1.0, 
                combined_label, 
                transform=ax_psi.transAxes, 
                fontsize=7, fontweight='normal', color='black', 
                va='bottom', ha='left'
            )
        
            # --- Graph phase ---
            im_phase = ax_phase.tripcolor(x, y, triangles, phase, shading="gouraud", cmap="twilight", vmin=-1, vmax=1)
            if i == 0: im_phase_ref = im_phase 
            ax_phase.set_aspect("equal")
            ax_phase.axis('off')
            
            # --- Axes Titles (Psi^2, Phi) ---
            if orientation == "vertical":
                # Títulos above the first row
                if i == 0:
                    ax_psi.set_title(r"$|\psi|^2$", fontsize=13, pad=17)
                    ax_phase.set_title(r"$\Delta \phi $", fontsize=11, pad=17)
            else:
                #Titles left of the first column
                if i == 0:
                    ax_psi.text(-0.1, 0.5, r"$|\psi|^2$", transform=ax_psi.transAxes, 
                                fontsize=13, va='center', ha='right', rotation=90)
                    ax_phase.text(-0.1, 0.5, r"$\Delta \phi $", transform=ax_phase.transAxes, 
                                fontsize=11, va='center', ha='right', rotation=90)

            # --- ANCHORS(spacing 0) ---
            if orientation == "vertical":

                ax_psi.set_anchor('E')
                ax_phase.set_anchor('W')
            else:

                ax_psi.set_anchor('S')
                ax_phase.set_anchor('N')

        # --- Color Bars---
            draw_cbar = False
            if orientation == "vertical" and i == cbar_idx_vert:
                draw_cbar = True
            elif orientation == "horizontal" and i == n_sols - 1:
                draw_cbar = True

            if draw_cbar:
                # 1. BARRA DE FASE (Derecha de fase)
                cax_phase = ax_phase.inset_axes([1.02, cbar_y_offset, 0.05, 1])
                cbar_phase = fig.colorbar(im_phase_ref, cax=cax_phase)
                cbar_phase.ax.tick_params(labelsize=8, length=2, pad=1)
                cbar_phase.set_ticks([-1, 0, 1])
                cbar_phase.set_ticklabels([r"$0$", "0.5", "1"])
                cbar_phase.outline.set_visible(False) 

                # 2. BARRA DE PSI (Derecha de psi) - RESTAURADA
                cax_psi = ax_psi.inset_axes([1.02, cbar_y_offset, 0.05, 1])
                cbar_psi = fig.colorbar(im_psi_ref, cax=cax_psi)
                cbar_psi.ax.tick_params(labelsize=8, length=2, pad=1)
                cbar_psi.set_ticks([0, 0.5, 1])
                cbar_psi.outline.set_visible(False)



        ax_top_left = axes.flat[0]
        ax_top_left.text(
            0.0, 1.15, 
            c_value, 
            transform=ax_top_left.transAxes, 
            fontsize=11, fontweight='normal', color='black', 
            va='bottom', ha='left'
        )

        # Main Title (Optional, positioned higher to not overlap J)
        if title:
            fig.suptitle(title, fontsize=12, y=1.05)
        
        if order_path is not None:        
            fig.savefig(order_path, facecolor='white', bbox_inches='tight', pad_inches=0.05)
            return fig, axes
        
        return fig, axes

  

    ######################################################################################
    #Varying Functions
    ######################################################################################
    # =========================
    # 2) Magnetization function
    # =========================
    def solve_field(self,field_o,field_f,field_steps,file_path,save_dir,device=None,d=0.1):
        '''
        A function that applies a magnetic field sweep to a device and returns the corresponding magnetizations and magnetic moments.
        
        :param device: tdgl.device object
        :param field: List or array of magnetic field values to be applied.
        :param d: Double, depth of the superconductor in micrometers (default is 0.1 µm).
        :return: Two lists containing the total magnetic moments and volumetric magnetizations for each applied
        '''
        if device == None :
            device = self.default_device
        field = np.linspace(field_o,field_f,field_steps)  # External field on mT (B)
        d = 0.1  # Superconductor depth in micrometers (µm)
        area = np.sum(device.areas)  # effective area of the device ( µm²)
        # =========================
        # 3) External Field sweep
        # =========================
        # Defines a list of 10 values for the external magnetic field form 0 to 1 mT
        moments = [] #total magnetic moment
        magnetizations = []  # volumetric magnetization
        # Loop for each value of B
        dynamic_terminal_names = [t.name for t in device.terminals]
        terminal_currents = {}
                        
                        # Map currents to the available terminals
                        # We assume the last added terminals are the ones we want to drive, 
                        # or that the order matches [Source, Drain].
        current_values = [0, -0]
                        
                        # We map values to the detected names in order
        for idx, t_name in enumerate(dynamic_terminal_names):
            if idx < len(current_values):
                terminal_currents[t_name] = current_values[idx]
       
        with tempfile.TemporaryDirectory(prefix="electro2_", suffix="_data") as temp_dir:
            for B in field:
                # Creates a uniform magnetic field of magnitude B
                # Solves Ginzburg–Landau equations with an applied field
                solution_field= self.default_solution("Bscan.h5",terminal_currents,device,vector_potential=B)
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
        mag_path = os.path.join(save_dir,"magnetization_vs_B.txt")
        susc_path = os.path.join(save_dir,"suceptibilidad_vs_B.txt")
        np.savetxt(mag_path, np.column_stack((field, magnetizations)),header="B[mT] M[uA/um^3]")
        np.savetxt(susc_path, np.column_stack((field, suceptibility)),header="B[mT] dM/dB [uA/(um^3·mT)]")
        plot_info = plot_labels={"fig_name":"currents.jpg","title":f'applied field vs magnetizations({field_o}–{field_f} mT)',"x":"B mT","y":"magnetization"}
        self.plot_parameters(field, -magnetizations,color_applied="green",plot_labels = plot_info,dir_path = save_dir+'/applied_field_vs_magnetization.jpg')
        plt.plot(field, -suceptibility,color="orange")
        plt.xlabel("B [$mT$]")
        plt.ylabel("susceptibility [mu_m$]")
        plt.grid()
        plt.title("applied field vs susceptibility")
        plt.savefig(save_dir+'/applied_field_vs_susceptibility.jpg')
        return moments,magnetizations, suceptibility
    def find_resistance(self,currents,voltages):
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
    
    ######################################################################################
    #3)Varying height function for current increments
    ########################################################

    def find_critical_currents(self,currents, voltages, quantity=5, smooth_window=11, poly_order=3, prominence=0.1):
        """
        Finds critical currents (Ic) by analyzing peaks in the differential resistance (dV/dI).
        It is more robust to noise than the simple threshold method.

    parameters:
    param: currents: np.array
    param: voltages: np.array
    param: quantity: int
    param: smooth_window: int
    param: poly_order: int
    param: prominence: float Minimum prominence of peaks as a fraction of the maximum peak height.
        Returns:
    
        np.array
            Array with current values where critical transitions occur.
        """
        # 1. Calculate Differential Resistance (dV/dI)
        # Use np.gradient which handles edges better than np.diff
        dV_dI = np.gradient(voltages, currents)
        
        # 2. Smooth the signal to remove numerical noise
        # This is crucial for discrete simulations
        if len(dV_dI) > smooth_window:
            dV_dI_smooth = savgol_filter(dV_dI, window_length=smooth_window, polyorder=poly_order)
        else:
            dV_dI_smooth = dV_dI

        # 3. Find peaks
        # 'prominence' ensures it is a real peak and not just noise
        # The threshold is calculated as a percentage of the maximum peak found
        height_threshold = np.max(np.abs(dV_dI_smooth)) * prominence
        peak_indices, properties = find_peaks(dV_dI_smooth, height=height_threshold)
        
        # 4. Select the best candidates if there are too many
        if len(peak_indices) > quantity:
            # Sort by peak height (largest voltage jumps first)
            heights = properties['peak_heights']
            # Get indices of the 'quantity' highest peaks
            best_indices = np.argsort(heights)[-quantity:]
            peak_indices = peak_indices[best_indices]
            # Reorder to keep chronological current order
            peak_indices.sort()
            
        return currents[peak_indices]

    def calculate_diode_efficiency(self,ic_positive, ic_negative):
        """
        Calculates the superconducting diode efficiency (eta) for arrays of critical currents.
        
        Formula: eta = (Ic+ - |Ic-|) / (Ic+ + |Ic-|)

        param: ic_positive: np.array
        param: ic_negative: np.array
        
        Returns:
        --------
        np.array
            Array with efficiency values (-1 to 1).
        """
        # Ensure they are numpy arrays
        I_plus = np.array(ic_positive)
        
        # Take absolute value for safety, in case -15 uA is passed instead of 15 uA
        I_minus = np.abs(np.array(ic_negative))
        
        # Verify they have the same size
        if I_plus.shape != I_minus.shape:
            # Ideally handle mismatch or raise error. 
            # For simplicity, we raise an error here to alert the user.
            raise ValueError(f"Arrays must have the same size. Ic+: {I_plus.shape}, Ic-: {I_minus.shape}")

        # Calculate denominator avoiding division by zero
        denominator = I_plus + I_minus
        
        # Handle zeros: if both currents are 0, efficiency is 0 (or undefined, we set 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = (I_plus - I_minus) / denominator
            # Where denominator is 0, set eta to 0
            eta[denominator == 0] = 0.0
            
        return eta
    def varying_increments(self, heights, currents, save_dir, field=[1.0], device_view=True,del_info=True):
            '''
            This function applies a current sweep to devices with varying heights.
            '''
            size = np.size(heights)
            voltages_arr = []
            resistance_arr = []    
            for h in range(0,size):
                print(h)
                for B in range(len(field)):
                    file_path = save_dir + f'/vert_dy_{h}'
                    print(f" the height is {heights[h]}")
                    
                    # 1. Create the base prototype
                    
                    device_prototype = create_device(
                        self.geometry,
                        self.layer,
                        self.gp["dimensions"],
                        incrementy=heights[h],
                        device_view=device_view,
                        clear_device_view=del_info
                    )

                    # 2. Re-calculate segments
                    current_segments = visualize_segments(device_prototype, view=False)
                    
                    # 3. Add terminals and MESH
                    device = add_multiple_terminals(
                        device_prototype,
                        current_segments,
                        self.terminal_dict,
                        self.layer,
                        self.max_edge_length_iv,
                        view_device=device_view,
                        stripe_length=self.gp.get("stripe_length", 0),
                        orientation=self.gp.get("orientation", "vertical")
                    )
                    
                    # Check terminals
                    dynamic_terminal_names = [t.name for t in device.terminals]
                    if not dynamic_terminal_names:
                        print("\n[WARNING] Device has no terminals!")

                    # FIX: Arguments passed in the correct order (device, currents, file_path)
                    voltages, resistance = self.current_application(
                        device, 
                        currents, 
                        file_path, 
                        B_field=field[B],
                        del_info=del_info
                    )
                    
                    # Define plot info
                    self.plot_info1 = {"fig_name": "currents.jpg", "title": f'current vs voltage for y={h+3}, field={field[B]}', "x": "current [$mA]", "y": "voltage V0"}
                    self.plot_info2 = {"fig_name": "currents.jpg", "title": f'current vs resistance for dy={h+3}, field={field[B]}', "x": "current [$mA]", "y": "resistance R0"}
                    
                    # Plot
                    self.plot_parameters(currents, voltages, self.plot_info1, plot_type="plot", dir_path=save_dir + f'/voltage_vs_current_deltay{h}_fiel{field[B]}.jpg', color_applied="blue")
                    self.plot_parameters(currents, resistance, self.plot_info2, plot_type="plot", dir_path=save_dir + f'/voltage_vs_resistance_deltay{h}_fiel{field[B]}.jpg', color_applied="orange")
                    
                    # Append to lists
                    voltages_arr.append(voltages)
                    resistance_arr.append(resistance)

                print(f"Result size: {np.size(voltages_arr)}")

                
            return voltages_arr, resistance_arr

    def current_application(self, device, currents, file_path, B_field=0,del_info=True):
        '''
        A function that applies a current sweep to a device and returns the corresponding voltages.
        '''
        voltages = []
        # =========================================================
        # Simulation
        # =========================================================
        start_time = time.time()
        total_simulations = len(currents)
        j = 0
        
        with tempfile.TemporaryDirectory(prefix='electro2_') as temp_dir:
            for I in currents:
                filename = f'solution_I_{I:.1f}.h5'
                applied_currents = {
                    "term_s": I,
                    "term_d": -I
                }
                
                solution_c = self.default_solution(
                    filename,
                    applied_currents, 
                    device=device,
                    vector_potential=B_field
                )
                
                dynamics = solution_c.dynamics
                indices = dynamics.time_slice(tmin=120)
                voltage = np.abs(np.mean(dynamics.voltage()[indices]))
                voltages.append(voltage)
                
                j += 1
                print(f"I = {I:.1f} µA, <V> = {voltage:.4f} V₀, progress: {np.round(j/len(currents)*100,2)}%", end='\r')
                
                if os.path.exists(filename):
                    os.remove(filename)
        
        # Ensure lists are converted to numpy arrays if required by find_resistance
        # But keeping lists as per your structure implies find_resistance handles them.
        resistances = self.find_resistance(currents, voltages)
        
        dd.save_data((currents, voltages), file_path+'voltage_vs_current.txt', "currents(µA)  Voltages(V0)")
        dd.save_data((currents, resistances), file_path +'resistance_vs_current.txt', "currents(µA)  resistances(R0)")
        if del_info == True:
            clear_output(wait=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time/60
        
        print(" " * 60, end='\r') 
        print("-" * 50)
        print(f"✅ The simulation was completed with {total_simulations} steps.")
        print(f"⏱️ The elapsed time was: {elapsed_time:.2f} seconds.")
        print(f"⏱️ The elapsed time was: {elapsed_minutes:.2f} minutes.")
        print(f"📊 Mean time per step was: {(elapsed_time / total_simulations):.2f} seconds.")
        print("-" * 50)
        
        return voltages, resistances