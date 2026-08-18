from scipy.signal import find_peaks, savgol_filter
from IPython.display import clear_output
from tdgl.sources import ConstantField
from tdgl.geometry import box
import default_directories as dd
from diode_analysis import (
    critical_current_at_voltage,
    differential_resistance,
    diode_efficiency,
    finite_1d as _as_finite_1d,
    validate_iv_data as _validate_iv_data,
)
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import string
import tdgl
import time
import os

def create_terminal_dictionary(terminals,names):
        if len(terminals) != len(names):
            raise ValueError("terminals and names must have the same length")
        return [
            {"id": int(term_id), "name": str(term_name)}
            for term_id, term_name in zip(terminals, names)
        ]
def create_dictionary(ids_names,values):
        if len(ids_names) != len(values):
            raise ValueError("ids_names and values must have the same length")
        return dict(zip(ids_names, values))
# #################################################################################
# =================================================================================
## Device functions 
# =================================================================================
# #################################################################################
def create_device(
    geometry_added,
    layer,
    dimensions,
    incrementx=0.0,
    incrementy=0.0,
    translationy=0.0,
    device_view=True,
    clear_device_view=True,
    length_units="um",
):
    """Build the asymmetric bridge used by the simulations.

    ``geometry_added`` contains the central film and left arm. A right arm is
    added with width ``dimensions['width_x2']`` and baseline height
    ``dimensions.get('right_height_y', 3.0)``. The historical 3 µm baseline is
    retained, but is now explicit and configurable.

    All geometry values are expressed in ``device.length_units`` (micrometres in
    the supplied notebook).
    """
    width_x = float(dimensions["width_x"])
    width_x2 = float(dimensions["width_x2"])
    right_height_y = float(dimensions.get("right_height_y", 3.0))
    real_size_x = width_x2 + float(incrementx)
    real_size_y = right_height_y + float(incrementy)
    if width_x <= 0 or real_size_x <= 0 or real_size_y <= 0:
        raise ValueError("device widths and heights must be positive")

    film_poly_up = tdgl.Polygon(
        "film_right_arm", points=box(width=real_size_x, height=real_size_y)
    ).translate(dx=width_x / 2, dy=float(translationy))
    combined_geometry = geometry_added.union(film_poly_up)
    combined_film = combined_geometry
    device = tdgl.Device(
        "asymmetric_bridge",
        layer=layer,
        film=combined_film,
        holes=[],
        length_units=length_units,
    )
    if clear_device_view:
        clear_output(wait=True)
    if device_view:
        device.draw(figsize=(10, 4))
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
    
    if source_id == drain_id:
        raise ValueError("source_id and drain_id must identify different segments")
    if any(index < 0 or index >= len(segments) for index in (source_id, drain_id)):
        raise ValueError(f"segment IDs must be in the range [0, {len(segments) - 1}]")
    device_center = np.mean(np.asarray(device.film.points), axis=0)

    # --- 1. Create Source ---
    seg_s = segments[source_id]
    term_s = create_terminal_from_segment(
        seg_s, "_new_source", pct=width_pct, stripe_length=stripe_length,
        device_center=device_center,
    )
    new_terminals.append(term_s)
    
    # Calculate Probe 1 (Source Side)
    probe_s = get_inward_probe_point(seg_s, depth=probe_depth, device_center=device_center)
    
    # --- 2. Create Drain ---
    seg_d = segments[drain_id]
    term_d = create_terminal_from_segment(
        seg_d, "_new_drain", pct=width_pct, stripe_length=stripe_length,
        device_center=device_center,
    )
    new_terminals.append(term_d)
    
    # Calculate Probe 2 (Drain Side)
    probe_d = get_inward_probe_point(seg_d, depth=probe_depth, device_center=device_center)

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
def create_terminal_from_segment(
    segment, name_suffix, pct=100, stripe_length=0.01, device_center=None
):
    """
    Creates a terminal extension attached to the segment with a fixed stripe_length.
    """
    if not 0 < pct <= 100:
        raise ValueError("pct must be in the interval (0, 100]")
    if stripe_length <= 0:
        raise ValueError("stripe_length must be positive")

    pts = np.asarray(segment['points'], dtype=float)
    p_start = pts[0]
    p_end = pts[-1]
    if device_center is None:
        device_center = np.zeros(2)
    device_center = np.asarray(device_center, dtype=float)
    
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
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            raise ValueError("cannot create a terminal from a zero-length segment")
        # Rotate 90 degrees to get a normal
        normal = np.array([-tangent[1], tangent[0]]) 
        normal = normal / tangent_norm
        
        # Ensure normal points outward from the actual device center.
        if np.dot(normal, segment_mid - device_center) < 0:
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

    else:
        raise ValueError(f"unsupported segment type: {segment.get('type')!r}")

    return tdgl.Polygon(f"term{name_suffix}", points=t_pts)
def get_inward_probe_point(segment, depth=1.0, device_center=None):
    """
    Calculates a point 'depth' units inside the device from the segment center.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    pts = np.asarray(segment['points'], dtype=float)
    p_start = pts[0]
    p_end = pts[-1]
    segment_mid = (p_start + p_end) / 2
    if device_center is None:
        device_center = np.zeros(2)
    device_center = np.asarray(device_center, dtype=float)
    
    if segment['type'] == 'line':
        tangent = p_end - p_start
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            raise ValueError("cannot place a probe from a zero-length segment")
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / tangent_norm
        
        # We want the inward normal (towards the actual device center).
        # If dot product is positive, it points outward, so flip it
        if np.dot(normal, segment_mid - device_center) > 0:
            normal = -normal
            
        return segment_mid + (normal * depth)
        
    elif segment['type'] == 'circle':
        # Move radially towards the center
        center = segment['center']
        vec_to_arc = segment_mid - center
        vec_unit = vec_to_arc / np.linalg.norm(vec_to_arc)
        # Move backwards (inward) from the arc
        return segment_mid - (vec_unit * depth)
    raise ValueError(f"unsupported segment type: {segment.get('type')!r}")
def add_multiple_terminals(
    device, segments, terminal_configs, layer, max_edge_length, width_pct=100,
    stripe_length=0.01, central_probe_separation=3.0, orientation="horizontal",
    sep_constant=0.7, view_device=True, smoothing_steps=100,
):
    """
    Adds multiple terminals and REPLACES existing probes with exactly 2 central probes.
    
    :param orientation: "horizontal" (default) aligns probes along X-axis. 
                        "vertical" aligns probes along Y-axis.
    """
    if len(terminal_configs) < 2:
        raise ValueError("at least two terminal configurations are required")
    if central_probe_separation <= 0 or sep_constant <= 0:
        raise ValueError("probe separation parameters must be positive")
    if max_edge_length <= 0 or smoothing_steps < 0:
        raise ValueError("max_edge_length must be positive and smoothing_steps non-negative")
    orientation = orientation.lower()
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    new_terminals = list(device.terminals)
    all_points = np.vstack([s['points'] for s in segments])
    device_center = np.mean(all_points, axis=0)
    existing_names = {terminal.name for terminal in new_terminals}
    requested_names = [str(config["name"]) for config in terminal_configs]
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("terminal names must be unique")
    if existing_names.intersection(f"term_{name}" for name in requested_names):
        raise ValueError("a requested terminal name already exists on the device")
    
    # 1. Add All Requested Terminals
    for config in terminal_configs:
        seg_id = config['id']
        name_suffix = f"_{config['name']}"
        
        if seg_id < 0 or seg_id >= len(segments):
            raise ValueError(
                f"terminal segment ID {seg_id} is outside [0, {len(segments) - 1}]"
            )

        seg = segments[seg_id]
        
        # Create Terminal Polygon
        term_poly = create_terminal_from_segment(
            seg, 
            name_suffix, 
            pct=width_pct, 
            stripe_length=stripe_length,
            device_center=device_center,
        )
        new_terminals.append(term_poly)

    # 2. Probe Placement (Central Intersection)
    # Determine direction based on orientation parameter
    if orientation == "vertical":
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
    new_device.make_mesh(max_edge_length=max_edge_length, smooth=int(smoothing_steps))
    if view_device:
        fig, ax = new_device.plot(mesh=True)
    
         # Visual confirmation of probes
        ax.scatter(final_probes[:,0], final_probes[:,1], c='red', marker='x', s=100, label='Probes', zorder=10)
    
    return new_device


class Building:
    """Simulation façade for an asymmetric superconducting bridge."""

    def __init__(self,name,geometry,layer,terminals,terminal_names,gp,segments_found=None):
        required_parameters = {
            "xi", "dimensions", "stripe_length", "orientation",
            "field_units", "current_units", "length_units",
        }
        missing = required_parameters.difference(gp)
        if missing:
            raise ValueError(f"missing global parameters: {sorted(missing)}")
        if gp["xi"] <= 0:
            raise ValueError("coherence length xi must be positive")
        self.name = name
        self.geometry = geometry
        self.layer = layer
        self.terminals = terminals
        self.terminal_names = terminal_names
        self.terminal_dict = create_terminal_dictionary(terminals,terminal_names)
        self.tempdir = tempfile.TemporaryDirectory()
        self.gp = gp
        max_edge_length_iv = gp["xi"] / 1.5
        self.max_edge_length_iv=  max_edge_length_iv
        self.max_edge_length_vortex = max_edge_length_iv
        self.smoothing_steps = int(gp.get("smoothing_steps", 100))
        device_prototype = create_device(
            geometry, layer, gp["dimensions"], incrementx=0, incrementy=0,
            length_units=gp["length_units"],
        )
        # Recompute the boundary from this exact geometry. ``segments_found`` is
        # retained in the signature for compatibility with older notebooks.
        del segments_found
        current_segments = visualize_segments(device_prototype, view=False)
        self.default_device = add_multiple_terminals(
            device_prototype,
            current_segments,
            self.terminal_dict,
            layer,
            max_edge_length_iv,
            stripe_length=gp["stripe_length"],
            orientation=gp["orientation"],
            smoothing_steps=self.smoothing_steps,
        )

 
    def default_options(self,d_filename,skip_t=200,solve_t=200,saves=200):
        if skip_t < 0 or solve_t <= 0 or saves <= 0:
            raise ValueError("skip_t must be non-negative; solve_t and saves must be positive")
        output_path = (
            d_filename
            if os.path.isabs(d_filename)
            else os.path.join(self.tempdir.name, d_filename)
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        options = tdgl.SolverOptions(
        skip_time=skip_t,  # initial relaxation time
        solve_time=solve_t,  # Real simulation time
        output_file=output_path,
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

        terminal_names = [terminal.name for terminal in device.terminals]
        unknown_names = set(terminal_currents_applied).difference(terminal_names)
        if unknown_names:
            raise ValueError(f"unknown terminal names: {sorted(unknown_names)}")
        terminal_currents_applied = {
            name: float(terminal_currents_applied.get(name, 0.0))
            for name in terminal_names
        }
        total_current = sum(terminal_currents_applied.values())
        if not np.isclose(total_current, 0.0, atol=1e-12):
            raise ValueError(
                f"terminal currents must sum to zero; received {total_current:g} "
                f"{self.gp['current_units']}"
            )

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
    def perform_sweep(self, currents, fields, heights, save_dir="", individual=False, device_view=True):
        """
        The unified 3-loop function.
        Iterates over Heights -> Fields -> Currents.
        """
        currents = _as_finite_1d(currents, "currents")
        fields = _as_finite_1d(fields, "fields")
        heights = _as_finite_1d(heights, "heights")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        solutions = []
        labels = []
        file_name = f'sweep_B_{len(fields)}_H_{len(heights)}_I_{len(currents)}.h5'
        
        total_steps = len(currents) * len(fields) * len(heights)
        step_count = 0
        
        print(f"Starting sweep with {total_steps} simulations...")

        for h in heights:
            device_prototype = create_device(
                self.geometry,
                self.layer,
                self.gp["dimensions"],
                # self.gp, # Removed duplicate arg if not needed by your create_device
                incrementy=h,
                device_view=device_view,
                clear_device_view=device_view,
                length_units=self.gp["length_units"],
            )

            current_segments = visualize_segments(device_prototype, view=False)
            
            device = add_multiple_terminals(
                device_prototype,
                current_segments,
                self.terminal_dict,
                self.layer,
                self.max_edge_length_iv,
                view_device=device_view,
                stripe_length=self.gp["stripe_length"],
                orientation=self.gp["orientation"],
                smoothing_steps=self.smoothing_steps,
            )
            
            dynamic_terminal_names = [t.name for t in device.terminals]
            if len(dynamic_terminal_names) < 2:
                raise ValueError("the sweep device must have at least two terminals")

            for B in fields:
                for I in currents:
                    step_count += 1
                    print(f"Simulating [{step_count}/{total_steps}]: H={h}, B={B}, I={I}", end='\r')
                    
                    terminal_currents = dict.fromkeys(dynamic_terminal_names, 0.0)
                    terminal_currents[dynamic_terminal_names[0]] = I
                    terminal_currents[dynamic_terminal_names[1]] = -I

                    sol = self.default_solution(
                        file_name, 
                        terminal_currents, 
                        device=device, 
                        vector_potential=B
                    )
                    
                    solutions.append(sol)
                    
                    label_parts = []
                    if len(heights) > 1: label_parts.append(f"h={h}")
                    if len(fields) > 1: label_parts.append(f"B={B}mT")
                    if len(currents) > 1: label_parts.append(f"I={I}uA")
                    if not label_parts: label_parts.append(f"I={I}, B={B}")
                    labels.append(", ".join(label_parts))

                    if individual:
                        exp_vor = "$\\vec{\\omega}=\\vec{\\nabla}\\times\\vec{K}$"
                        exp_sp = "$\\mu/v_0$"
                        titles_group = {
                            "sheet_current": f'Sheet current density for {B} mT, {I} uA',
                            "order_parameter": f'Order parameter for {B} mT, {I} uA',
                            "vorticity": "Vorticity" + exp_vor,
                            "scalar_potential": "Scalar potential" + exp_sp
                        }
                        save_p_scd = os.path.join(save_dir, f'scd_B{B}_I{I}uA.jpg') if save_dir else None
                        save_p_op = os.path.join(save_dir, f'op_B{B}_I{I}uA.jpg') if save_dir else None
                        
                        self.plot_group(
                            sol, (5,4), titles_group, 
                            currentBool=True, titleBool=False, 
                            order_path=save_p_op, current_path=save_p_scd, view=True
                        )

        print("\nSweep complete.")
        
        if len(solutions) > 0:
            orient = "horizontal" if len(solutions) <= 4 else "vertical"
            full_save_path = None
            if save_dir:
                full_save_path = os.path.join(save_dir, file_name + "_sweep_result.jpg")
                
            # FIX: Removed self. if plot_parameter_sweep is global
            fig, axes = self.plot_parameter_sweep(
                solutions, 
                labels, 
                orientation=orient,
                order_path=full_save_path,
                c_value=f"I={currents} {self.gp['current_units']}, Δy={heights} µm"
            )
            plt.show()
            
        return solutions, labels
   
    ######################################################################################
    #1) PLOTS
    ######################################################################################     
    def plot_solution(
        self, solution, order_title=None, current_title=None, currentBool=True,
        order_path=None, current_path=None, view=True, snapshot_time=None,
    ):
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

            # Select a time explicitly when a phase-slip snapshot is desired.
            # Otherwise preserve the caller's current solve step.
            if snapshot_time is not None:
                solution.solve_step = solution.closest_solve_step(snapshot_time)
            
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
        Plot ``|K_s| / |psi|²`` as a phase-gradient-like diagnostic.

        This is a useful dimensionless proxy away from vortex cores, not a direct
        SI-valued superfluid velocity: the GL current also contains gauge and
        normalization factors.
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
        phase_grad_mag = np.divide(
            J_mag,
            rho,
            out=np.full_like(J_mag, np.nan, dtype=float),
            where=rho >= 0.05,
        )
        
        # 4. Mask Vortex Cores
        # Inside a vortex, density is 0, so the gradient is mathematically infinite/undefined.
        # We mask these out for a cleaner plot.
        # 5. Plot
        device = solution.device
        x, y = device.points[:, 0], device.points[:, 1]
        triangles = device.triangles

        finite_values = phase_grad_mag[np.isfinite(phase_grad_mag)]
        if finite_values.size == 0:
            raise ValueError("phase-gradient proxy is undefined because |psi|² is below 0.05 everywhere")
        im = ax.tripcolor(
            x, y, triangles, 
            phase_grad_mag, 
            shading="gouraud", 
            cmap="plasma",
            vmax=np.percentile(finite_values, 95) # Auto-scale to ignore spikes
        )
        
        # Add Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"$|K_s|/|\psi|^2$ (dimensionless proxy)")
        

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
            raise ValueError(f"labels ({len(labels)}) must match solutions ({n_sols})")
        if orientation not in {"vertical", "horizontal"}:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

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
            seq_char = alphabet[i] if i < len(alphabet) else f"{i + 1}"
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
                cbar_phase.set_ticklabels([r"$-1$", "0", "1"])
                cbar_phase.set_label(r"$\theta/\pi$")
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
    def solve_field(
        self, field_o, field_f, field_steps, file_path=None, save_dir=None,
        device=None, d=None,
    ):
        """Solve a zero-current field sweep and compute ``M=m/(area*d)``.

        Magnetic field is expressed in ``gp['field_units']`` and thickness ``d``
        in device length units. If ``d`` is omitted, the layer thickness from
        ``gp['d']`` is used. ``file_path`` is retained as the legacy output-dir
        argument; ``save_dir`` takes precedence when both are supplied.
        """
        if device is None:
            device = self.default_device
        if int(field_steps) != field_steps or field_steps < 2:
            raise ValueError("field_steps must be an integer greater than one")
        if not np.isfinite([field_o, field_f]).all() or field_o == field_f:
            raise ValueError("field bounds must be finite and different")

        thickness = float(self.gp.get("d", 0.1) if d is None else d)
        if thickness <= 0:
            raise ValueError("superconductor thickness d must be positive")
        area = float(np.sum(device.areas))
        if not np.isfinite(area) or area <= 0:
            raise ValueError("device area must be finite and positive")

        output_dir = save_dir if save_dir is not None else file_path
        output_dir = output_dir or "."
        os.makedirs(output_dir, exist_ok=True)

        field = np.linspace(float(field_o), float(field_f), int(field_steps))
        terminal_currents = {terminal.name: 0.0 for terminal in device.terminals}
        moments = []
        for index, field_value in enumerate(field):
            solution_field = self.default_solution(
                f"Bscan_{index:04d}.h5",
                terminal_currents,
                device,
                vector_potential=field_value,
            )
            moments.append(
                float(solution_field.magnetic_moment(units="uA * um**2", with_units=False))
            )
            if hasattr(solution_field, "close"):
                solution_field.close()

        moments = np.asarray(moments)
        magnetizations = moments / (area * thickness)
        susceptibility = np.gradient(magnetizations, field)

        dd.save_data(
            (field, magnetizations),
            os.path.join(output_dir, "magnetization_vs_B.txt"),
            "B[mT] M[uA/um^3]",
        )
        dd.save_data(
            (field, susceptibility),
            os.path.join(output_dir, "susceptibility_vs_B.txt"),
            "B[mT] dM/dB[uA/(um^3*mT)]",
        )
        plot_info = {
            "title": f"Magnetization vs applied field ({field_o}–{field_f} mT)",
            "x": "B [mT]",
            "y": "M [µA/µm³]",
        }
        self.plot_parameters(
            field,
            magnetizations,
            color_applied="green",
            plot_labels=plot_info,
            dir_path=os.path.join(output_dir, "applied_field_vs_magnetization.jpg"),
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(field, susceptibility, color="orange")
        ax.set(xlabel="B [mT]", ylabel="dM/dB [µA/(µm³·mT)]", title="Susceptibility")
        ax.grid(True)
        fig.savefig(
            os.path.join(output_dir, "applied_field_vs_susceptibility.jpg"),
            facecolor="white",
            bbox_inches="tight",
        )
        plt.show()
        return moments, magnetizations, susceptibility
    def find_resistance(self,currents,voltages):
        '''
        This function calculates the resistance at each point in the IV curve by computing the gradient of voltage with respect to current.
        Parameters:
        :param currents (np.array): Array of current values.
        :param voltages (np.array): Array of voltage values corresponding to the currents.
        Returns:
        np.array: Array of resistance values calculated as dV/dI.
        '''
        return differential_resistance(currents, voltages)
    
    ######################################################################################
    #3)Varying height function for current increments
    ########################################################

    def find_critical_currents(self,currents, voltages, quantity:int=5, smooth_window:int=11, poly_order=3, prominence=0.1):
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
        currents, voltages = _validate_iv_data(currents, voltages, min_size=3)
        if quantity < 1:
            raise ValueError("quantity must be at least one")
        if not 0 <= prominence <= 1:
            raise ValueError("prominence must be between zero and one")
        if poly_order < 0:
            raise ValueError("poly_order must be non-negative")

        # The magnitude works for both ascending positive sweeps and descending
        # negative sweeps, independent of probe polarity.
        differential_resistance = np.abs(np.gradient(voltages, currents))
        
        # 2. Smooth the signal to remove numerical noise
        # This is crucial for discrete simulations
        window = min(int(smooth_window), differential_resistance.size)
        if window % 2 == 0:
            window -= 1
        if window > poly_order and window >= 3:
            smoothed = savgol_filter(
                differential_resistance, window_length=window, polyorder=poly_order
            )
        else:
            smoothed = differential_resistance

        # 3. Find peaks
        # 'prominence' ensures it is a real peak and not just noise
        # The threshold is calculated as a percentage of the maximum peak found
        prominence_threshold = np.ptp(smoothed) * prominence
        peak_indices, properties = find_peaks(
            smoothed,
            prominence=prominence_threshold,
        )
        
        # 4. Select the best candidates if there are too many
        if len(peak_indices) > quantity:
            # Sort by peak height (largest voltage jumps first)
            heights = properties['prominences']
            # Get indices of the 'quantity' highest peaks
            best_indices = np.argsort(heights)[-quantity:]
            peak_indices = peak_indices[best_indices]
            # Reorder to keep chronological current order
            peak_indices.sort()
            
        return currents[peak_indices]

    def estimate_critical_current(self, currents, voltages, voltage_threshold):
        """Estimate the first threshold-crossing current by linear interpolation.

        This single-criterion estimate is preferable to peak finding when an
        experimental or numerical voltage criterion is known. The threshold is
        applied to ``abs(voltage)`` and the signed current is returned.
        """
        return critical_current_at_voltage(currents, voltages, voltage_threshold)

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
        return diode_efficiency(ic_positive, ic_negative)
    def varying_increments(self, heights, currents, save_dir, file_suffix="", field=0, device_view=True, del_info=True):
            '''
            Applies a current sweep to devices with varying heights.
            
            FIXES:
            1. Uses lists for storage to prevent index/size errors.
            2. Passes 'save_dir' down so temp files are created there (saving C: drive space).
            3. Correct arguments for current_application.
            '''
            heights = _as_finite_1d(heights, "heights")
            currents = _as_finite_1d(currents, "currents")
            voltages_arr = []
            resistance_arr = []
            
            # Ensure the main save directory exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Create a specific temp folder in your SAVE_DIR (not on C:)
            local_temp_root = os.path.join(save_dir, "_temp_files")
            if not os.path.exists(local_temp_root):
                os.makedirs(local_temp_root)

            for i, h in enumerate(heights):
                # Define specific sub-folder for this height
                # We use this path for both saving results AND temporary files
                current_save_path = (
                    save_dir
                    if len(heights) == 1
                    else os.path.join(save_dir, f'dy_{h:g}')
                )
                if not os.path.exists(current_save_path):
                    os.makedirs(current_save_path)
                
                print(f"--- Height iteration: {h} ---")
                
                # 1. Create Device
              
                device_prototype = create_device(
                    self.geometry, 
                    self.layer, 
                    self.gp["dimensions"], 
                    incrementy=h, 
                    device_view=device_view,
                    length_units=self.gp["length_units"],
                )

                # 2. Add Terminals
                current_segments = visualize_segments(device_prototype, view=False)
                device = add_multiple_terminals(
                    device_prototype, 
                    current_segments, 
                    self.terminal_dict, 
                    self.layer, 
                    self.max_edge_length_iv,
                    view_device=device_view, stripe_length=self.gp["stripe_length"],
                    orientation=self.gp.get("orientation", "vertical"),
                    smoothing_steps=self.smoothing_steps,
                )

                # 3. Apply Currents
                # We pass 'local_temp_root' to force temp files to be created on your data drive
                voltages, resistance = self.current_application(
                    device, 
                    currents, 
                    current_save_path, 
                    B_field=field,
                    temp_dir_root=local_temp_root,
                    del_info=del_info
                )
                
                # 4. Save & Plot
                self.plot_info1 = {"fig_name": "currents.jpg", "title": f'I vs V (dy={h})', "x": "current [uA]", "y": "voltage [V0]"}
                self.plot_info2 = {"fig_name": "currents.jpg", "title": f'I vs R (dy={h})', "x": "current [uA]", "y": "resistance [R0]"}
                
                plot_filename_v = f'voltage_vs_current_dy{i}_{file_suffix}.jpg'
                plot_filename_r = f'voltage_vs_resistance_dy{i}_{file_suffix}.jpg'
                
                self.plot_parameters(currents, voltages, self.plot_info1, plot_type="plot", 
                                    dir_path=os.path.join(current_save_path, plot_filename_v), color_applied="blue")
                self.plot_parameters(currents, resistance, self.plot_info2, plot_type="plot", 
                                    dir_path=os.path.join(current_save_path, plot_filename_r), color_applied="orange")
                
                # Store results
                voltages_arr.append(voltages)
                resistance_arr.append(resistance)

            return voltages_arr, resistance_arr

    def current_application(
        self, device, currents, file_path, B_field=0, temp_dir_root=None,
        del_info=True, averaging_tmin=120, absolute_voltage=False,
    ):
        '''
        Apply a current sweep and return voltage and differential resistance.

        Voltage is signed by default so probe polarity and non-reciprocity are not
        discarded. Set ``absolute_voltage=True`` only for legacy magnitude plots.
        ``del_info`` is retained for notebook compatibility.
        '''
        del del_info
        currents = _as_finite_1d(currents, "currents")
        if averaging_tmin < 0:
            raise ValueError("averaging_tmin must be non-negative")
        terminal_names = [terminal.name for terminal in device.terminals]
        if len(terminal_names) < 2:
            raise ValueError("current sweeps require at least two device terminals")
        voltages = []
        
        # If no temp root provided, fallback to file_path, else current dir
        if temp_dir_root is None:
            temp_dir_root = file_path if os.path.exists(file_path) else os.getcwd()
        os.makedirs(temp_dir_root, exist_ok=True)
        os.makedirs(file_path, exist_ok=True)

        start_time = time.time()
        
        # FIX: Force temp dir to be in your specific storage path
        with tempfile.TemporaryDirectory(dir=temp_dir_root, prefix="sim_temp_") as temp_dir:
            for j, I in enumerate(currents):
                
                # Unique filename per step
                filename = os.path.join(temp_dir, f'solution_step_{j}.h5')
                
                applied_currents = dict.fromkeys(terminal_names, 0.0)
                applied_currents[terminal_names[0]] = I
                applied_currents[terminal_names[1]] = -I
                
                # Run Solver
                solution_c = self.default_solution(
                    filename,
                    applied_currents, 
                    device=device,
                    vector_potential=B_field
                )
                
                # Extract Data
                dynamics = solution_c.dynamics
                indices = dynamics.time_slice(tmin=averaging_tmin)
                voltage_samples = np.asarray(dynamics.voltage()[indices])
                if voltage_samples.size == 0:
                    raise ValueError(
                        f"no voltage samples exist after t={averaging_tmin}; "
                        "lower averaging_tmin or increase solve_time"
                    )
                voltage = float(np.mean(voltage_samples))
                if absolute_voltage:
                    voltage = abs(voltage)
                voltages.append(voltage)
                
                # Progress
                progress = (j + 1) / len(currents) * 100
                print(f"I={I:.1f}uA, V={voltage:.4f} [Progress: {progress:.1f}%]", end='\r')
                
                # CRITICAL: Close solution to release file handle immediately
                if hasattr(solution_c, 'close'):
                    solution_c.close()
                del solution_c 
                
                # Try to remove file immediately to save space
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except OSError:
                    pass # TemporaryDirectory cleanup will retry.

        # Calculate Resistance
        if len(currents) > 1:
            resistances = self.find_resistance(currents, voltages)
        else:
            resistances = np.array([voltages[0] / currents[0] if currents[0] else 0.0])

        # Save Data
        dd.save_data((currents, voltages), os.path.join(file_path, 'voltage_vs_current.txt'), "currents(uA) Voltages(V0)")
        dd.save_data((currents, resistances), os.path.join(file_path, 'resistance_vs_current.txt'), "currents(uA) resistances(R0)")
        
        # Print Stats
        total_time = time.time() - start_time
        print("\n" + "-"*50)
        print(f"✅ Simulation Complete. Time: {total_time:.2f}s ({total_time/len(currents):.2f}s/step)")
        print("-"*50)
        
        return np.asarray(voltages), np.asarray(resistances)


# Backward-compatible alias used throughout the original notebook.
building = Building
