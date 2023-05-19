import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

updatecount = 0

# Define a function to get the x and y range based on the chosen shape and its dimensions
def get_range(shape, dimensions):
    if shape == 'parallel_plate':
        l, w, d = dimensions
        return l * 1.5, w * 5
    elif shape in ['line_charge_vertical', 'line_charge_horizontal']:
        length, _, _ = dimensions
        return length * 1.5, length * 1.5
    elif shape == 'rect':
        width, height, _ = dimensions
        return width * 1.5, height * 1.5
    elif shape == 'circle':
        radius, *_ = dimensions
        return radius * 2.5, radius * 2.5
    elif shape == 'annulus':
        inner_radius, outer_radius, _ = dimensions
        return outer_radius * 2.5, outer_radius * 2.5
    else:
        return 1, 1

    
# Functions to create rho vectors for different shapes
def parallel_plate_rho(l, w, d, X, Y, sigma):
    mask_upper = np.logical_and(X > -l/2, X < l/2)
    mask_upper &= np.logical_and(Y > d/2, Y < w + d/2)
    mask_lower = np.logical_and(X > -l/2, X < l/2)
    mask_lower &= np.logical_and(Y < -d/2, Y > -w-d/2)

    rho = np.zeros_like(X)
    rho[mask_upper] = sigma
    rho[mask_lower] = -sigma
    rho = rho.ravel()
    return rho

def rho_rect(x, y, width, height, charge_density):
    mask = np.logical_and(np.abs(x) <= width/2, np.abs(y) <= height/2)
    rho = np.zeros_like(x)
    rho[mask] = charge_density
    rho = rho.ravel()
    return rho

def rho_circle(x, y, radius, charge_density):
    r = np.sqrt(x**2 + y**2)
    rho = np.zeros_like(x)
    rho[r <= radius] = charge_density
    rho = rho.ravel()
    return rho

def rho_annulus(x, y, inner_radius, outer_radius, charge_density):
    r = np.sqrt(x**2 + y**2)
    rho = np.zeros_like(x)
    rho[np.logical_and(r >= inner_radius, r <= outer_radius)] = charge_density
    rho = rho.ravel()
    return rho

def rho_line_charge(x, y, length, charge_density, orientation='horizontal'):
    if orientation == 'horizontal':
        mask = np.logical_and(np.abs(y) <= dy, np.abs(x) <= length/2)
    else:
        mask = np.logical_and(np.abs(x) <= dx, np.abs(y) <= length/2)
    rho = np.zeros_like(x)
    rho[mask] = charge_density
    rho = rho.ravel()
    return rho

def rho_point_charge(x, y, charge):
    rho = np.zeros_like(x)
    mask = np.logical_and(np.abs(x) <= dx, np.abs(y) <= dy)
    rho[mask] = charge
    rho = rho.ravel()
    return rho

def read_custom_rho_file(file_path, N, x_range, y_range):
    data = np.loadtxt(file_path)
    X, Y, rho_values = data[:, 0], data[:, 1], data[:, 2]

    x_range = abs(X.max() - X.min())
    y_range = abs(Y.max() - X.min())
    x_range *= 1.5
    y_range *= 1.5
    x = np.linspace(-x_range, x_range, N)
    y = np.linspace(-y_range, y_range, N)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Interpolate the custom rho values onto the grid
    rho = np.zeros_like(X_grid)
    for x_val, y_val, rho_val in zip(X, Y, rho_values):
        ix = int(np.round(N * (x_val + x_range) / (2 * x_range)))
        iy = int(np.round(N * (y_val + y_range) / (2 * y_range)))
        rho[iy, ix] = rho_val

    return rho.ravel()

def plot_graph_with_input(shape, dimensions, efield,N,V0=None,custom_rho_file=None):
    x_range, y_range = get_range(shape, dimensions)

    # Grid inputs
    global dx
    global dy
    dx = 2 * x_range / N
    dy = 2 * y_range / N

    # Create grid
    x = np.linspace(-x_range, x_range, N)
    y = np.linspace(-y_range, y_range, N)
    X, Y = np.meshgrid(x, y)

    # Constants
    eps0 = 8.85e-12

    # Create sparse matrix for Laplacian operator
    A = diags([-4, 1, 1, 1, 1], [0, -1, 1, -N, N], shape=(N**2, N**2))
    A.setdiag(1, k=N)
    A.setdiag(1, k=-N)

    # Create charge density vector
    rho = np.zeros_like(X).ravel()

    # Choose the shape for rho
    if shape == 'parallel_plate':
        l, w, d = dimensions
        sigma = V0 * eps0 / d
        rho += parallel_plate_rho(l, w, d, X, Y, sigma)
    elif shape == 'line_charge_vertical':
        l, c, _ = dimensions
        rho += rho_line_charge(X, Y, l, c, orientation="Vertical")
    elif shape == 'rect':
        w, h, c = dimensions
        rho += rho_rect(X, Y, w, h, c)
    elif shape == 'circle':
        r, c,*_ = dimensions
        rho += rho_circle(X, Y, r, c)
    elif shape == 'annulus':
        inr, outr, c = dimensions
        rho += rho_annulus(X, Y, inr, outr, c)
    elif shape == 'line_charge_horizontal':
        l, c, _ = dimensions
        rho += rho_line_charge(X, Y, l, c, orientation='horizontal')
    elif shape == 'point_charge':
        c, _, _ = dimensions
        rho += rho_point_charge(X, Y, c)
    elif shape == 'custom':
        rho += read_custom_rho_file(custom_rho_file, N, x_range, y_range)

    rho_max = np.max(rho)
    if rho_max == 0.0:
        output_label_text.set(f"Zero Error: Something has made charge density 0.")
        
        return
    #print(f"row max {rho_max}")
    rho_normalized = rho / (rho_max)

    # Solve for potential
    V, info = cg(A, rho_normalized, tol=1e-10)

    # Reshape potential into 2D array for plotting
    V = V.reshape(N, N)
    V = np.flipud(V)
    V *= rho_max

    # Calculate the electric field
    Ex, Ey = np.gradient(-V)
    Ex /= dx
    Ey /= dy
    Ex *= rho_max
    Ey *= rho_max
    E = np.sqrt(Ex**2 + Ey**2)

    # Plot scalar potential and electric field
    fig, ax = plt.subplots()
    if x_range == y_range:
        ax.set_aspect('equal', adjustable='box')
    if efield:
        #ax.quiver(X, Y, Ey, Ex)
        im = ax.pcolormesh(X, Y, E, cmap='jet', shading='auto')
        plt.title("Electric Field")
    else:
        im = ax.pcolormesh(X, Y, V, cmap='jet', shading='auto')
        plt.title("Scalar Potential")

    cbar = plt.colorbar(im)
    
    plt.xlabel("x")
    plt.ylabel("y")

    # Define a function to display potential or electric field at clicked point
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            col = int(np.round(N * (1 - (iy + y_range) / (2 * y_range))))
            row = int(np.round(N * (ix + x_range) / (2 * x_range)))
        if efield:
            output_label_text.set("Electric Field at ({:.3f}, {:.3f}): Ex = {:.3e} V/m, Ey = {:.3e} V/m".format(ix, iy, Ex[row, col], Ey[row, col]))
        else:
            output_label_text.set("Potential at ({:.3f}, {:.3f}): {:.3e} V".format(ix, iy, V[row, col]))

    # Connect the onclick function to the figure
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


def submit():
    output_label_text.set('')
    shape = shape_var.get()
    shape = shape_var.get()
    efield = efield_var.get()

    try:
        V0 = float(V0_entry.get())
    except:
        V0 = None

    try:
        dimensions = tuple(float(e.get()) for e in dimension_entries)
        N = int(N_entry.get())

        # Check if dimensions and N are positive
        if any(dim <= 0 for dim in dimensions) and shape != 'custom':
            raise ValueError("Dimensions must be positive numbers.")
        if N <= 10:
            raise ValueError("Have at least 10 gridpoints")
            

    except ValueError as ve:
        output_label_text.set(f"Error: {ve}")
    except Exception as e:
        output_label_text.set(f"Unexpected Error: {e}")

    if shape == "annulus":
        innr,outr,c = dimensions
        if outr - innr < 0:
            output_label_text.set(f"For annulus, the outer radius > inner radius")
            return
    
    if shape == 'custom':
        try:
            custom = filedialog.askopenfilename()
            if not custom:
                raise Exception("File not selected")
            #plot_graph_with_input(shape, dimensions, efield, N, V0, custom)
        except FileNotFoundError:
            output_label_text.set(f"File not found.")
        except Exception as e:
            output_label_text.set(f"Error: {e}")
    else:
        custom = None
        
    plot_graph_with_input(shape, dimensions, efield, N, V0, custom)
    

# Create the main Tkinter window
root = tk.Tk()
root.title("Electrostatics GUI")

# Add a label and a dropdown menu to select the shape
shape_var = tk.StringVar(root)
shape_var.set("parallel_plate")
shape_options = [
    "parallel_plate",
    "line_charge_vertical",
    "line_charge_horizontal",
    "rect",
    "circle",
    "annulus",
    "point_charge",
    "custom",
]
shape_menu = tk.OptionMenu(root, shape_var, *shape_options)
shape_menu.grid(row=0, column=0, padx=10, pady=10)
shape_label = tk.Label(root, text="Dimensions")
shape_label.grid(row=0, column=1, padx=10, pady=10)

# Create V0 input field and label for parallel plate
V0_label = tk.Label(root, text="Plate Voltage(Volts)")
V0_entry = tk.Entry(root)


N_label = tk.Label(root, text="Grid Points N")
N_label.grid(row=9, column=0, padx=10, pady=10)
N_entry = tk.Entry(root)
N_entry.grid(row=9, column=1, padx=10, pady=10)
N_entry.insert(0, "300")  # Default value



def update_input_fields(*args):
    global updatecount
    shape = shape_var.get()

    if shape == "parallel_plate":
        labels = ["Length(Meters)", "Width(Meters)", "Distance between Plates(Meters)"]
        V0_label.grid(row=7, column=0, padx=10, pady=10)
        V0_entry.grid(row=7, column=1, padx=10, pady=10)
    else:
        labels = ["Length(Meters)", "Charge Density(C⋅m^-2)"] if shape in ["line_charge_vertical", "line_charge_horizontal"] else ["Width", "Height", "Charge Density"]
        V0_label.grid_forget()
        V0_entry.grid_forget()

    if shape in ["line_charge_vertical", "line_charge_horizontal"]:
        labels = ["Length(Meters)", "Charge Density(C⋅m^-2)"]
    elif shape == "rect":
        labels = ["Width(Meters)", "Height(Meters)", "Charge Density(C⋅m^-2)"]
    elif shape == "circle":
        labels = ["Radius(Meters)", "Charge Density(C⋅m^-2)"]
    elif shape == "annulus":
        labels = ["Inner Radius(Meters)", "Outer Radius(Meters)", "Charge Density(C⋅m^-2)"]
    elif shape == "point_charge":
        labels = ["Charge(Coulomb)"]
    elif shape == "custom":
        labels = []

    for i in range(len(dimension_labels)):
        if i < len(labels):
            dimension_labels[i].config(text=labels[i])
            dimension_entries[i].config(state=tk.NORMAL)
        else:
            dimension_labels[i].config(text="-")
            dimension_entries[i].delete(0, tk.END)
            dimension_entries[i].insert(0, "0")
            dimension_entries[i].config(state=tk.DISABLED)
    for i, (label, entry) in enumerate(zip(dimension_labels, dimension_entries)):
        label.grid(row=i+1, column=0, padx=10, pady=10)
        entry.grid(row=i+1, column=1, padx=10, pady=10)
    if updatecount == 0:
        # Add default values for the input fields
        dimension_entries[0].insert(0, "1")
        dimension_entries[1].insert(0, "0.2")
        dimension_entries[2].insert(0, "0.01")
        V0_entry.insert(0, "1")
        updatecount += 1
        
shape_var.trace("w", update_input_fields)

# Add input fields for each shape
dimension_entries = [
    tk.Entry(root),
    tk.Entry(root),
    tk.Entry(root)
]
dimension_labels = [
    tk.Label(root, text="Length"),
    tk.Label(root, text="Width"),
    tk.Label(root, text="Distance")
]

for i, (label, entry) in enumerate(zip(dimension_labels, dimension_entries)):
    label.grid(row=i+1, column=0, padx=10, pady=10)
    entry.grid(row=i+1, column=1, padx=10, pady=10)

# Add a checkbox to choose between electric field and scalar potential
efield_var = tk.BooleanVar()
efield_checkbox = tk.Checkbutton(root, text="Electric Field", variable=efield_var)
efield_checkbox.grid(row=10, column=1, padx=10, pady=10)

# Add a button to submit the form
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.grid(row=10, column=0, padx=10, pady=10)

# Add a label to display the output of the onclick event
output_label_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_label_text)
output_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

update_input_fields()
root.mainloop()
