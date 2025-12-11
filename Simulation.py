import numpy as np
import tomllib
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle


def optimal_params():
    with open("System_Params.toml", "rb") as f:
        system_params = tomllib.load(f)

    wavelength = system_params["wavelength"]*1e-6  # in mm
    crystal_length = system_params["crystal_length"]

    confocal_param = 2.84
    optimal_rayleigh = crystal_length/ (2*confocal_param)
    optimal_waist = math.sqrt((optimal_rayleigh*wavelength)/math.pi)

    return (optimal_rayleigh, optimal_waist, wavelength)


# ----------------------------------------------------------
# Gaussian beam radius function
# ----------------------------------------------------------
def gaussian_w(z, w0, zR, z0):
    return w0 * np.sqrt(1 + ((z - z0)/zR)**2)

# ----------------------------------------------------------
# Field for overlap calculation
# ----------------------------------------------------------
def gaussian_field_xy(x, y, w):
    r2 = x**2 + y**2
    return np.exp(-r2 / w**2)

# ----------------------------------------------------------
# Overlap efficiency at a given z
# ----------------------------------------------------------
def overlap_efficiency(z, A_params, B_params, grid_size=4.0, resolution=600):
    w0A, zRA, z0A = A_params
    w0B, zRB, z0B = B_params

    wA = gaussian_w(z, w0A, zRA, z0A)
    wB = gaussian_w(z, w0B, zRB, z0B)

    xs = np.linspace(-grid_size, grid_size, resolution)
    ys = np.linspace(-grid_size, grid_size, resolution)
    X, Y = np.meshgrid(xs, ys)

    EA = gaussian_field_xy(X, Y, wA)
    EB = gaussian_field_xy(X, Y, wB)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    overlap = np.sum(EA * EB) * dx * dy
    powerA = np.sum(EA**2) * dx * dy
    powerB = np.sum(EB**2) * dx * dy

    return (overlap**2) / (powerA * powerB)

# ----------------------------------------------------------
# Fixed Beam A (Optimal Beam)
# ----------------------------------------------------------
z0A = 0  # waist at rectangle center
zRA, w0A, lamA = optimal_params()
A_params = (w0A, zRA, z0A)

# ----------------------------------------------------------
# Initial Beam B
# ----------------------------------------------------------
w0B_init = w0A
z0B_init = 0
zRB_init = zRA

# ----------------------------------------------------------
# Plot setup
# ----------------------------------------------------------
z_min, z_max = -100, 100  # propagation along z
z = np.linspace(z_min, z_max, 400)

fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(bottom=0.35)
ax.set_title("SHG Efficiency by Deviation from Optimal")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Beam radius w(z) (mm)")

# Rectangle (crystal) in center
rect_length = 40
rect_height = 50
rect = Rectangle((-rect_length/2, -rect_height/2), rect_length, rect_height,
                 facecolor='skyblue', alpha=0.5)
ax.add_patch(rect)

# Initial beam curves
wA_z = gaussian_w(z, w0A, zRA, z0A)
wB_z = gaussian_w(z, w0B_init, zRB_init, z0B_init)

curveA, = ax.plot(z, wA_z, 'r', label='Beam A', linewidth=2)
ax.plot(z, -wA_z, 'r', linewidth=2)  # lower envelope

curveB, = ax.plot(z, wB_z, 'g', label='Beam B', linewidth=2)
ax.plot(z, -wB_z, 'g', linewidth=2)

ax.legend()
ax.set_xlim(z_min, z_max)
ax.set_ylim(-100, 100)

# Overlap text at rectangle center
eta0 = overlap_efficiency(0, A_params, (w0B_init, zRB_init, z0B_init))
eta_text = ax.text(0, 90, f"Overlap: {eta0*100:.3f}%", ha='center', fontsize=14, fontweight='bold')

# ----------------------------------------------------------
# Sliders for Beam B parameters
# ----------------------------------------------------------
ax_w0B = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_zRB = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_z0B = plt.axes([0.2, 0.15, 0.65, 0.03])

slider_w0B = Slider(ax_w0B, "Beam B waist w0", 0, 1.5, valinit=w0B_init)
slider_zRB = Slider(ax_zRB, "Rayleigh length zR_B", 0, 50, valinit=zRB_init)
slider_z0B = Slider(ax_z0B, "Waist position z0_B", -50, 50, valinit=z0B_init)

# ----------------------------------------------------------
# Slider update function
# ----------------------------------------------------------
def update(val):
    w0B = slider_w0B.val
    zRB = slider_zRB.val
    z0B = slider_z0B.val

    B_params = (w0B, zRB, z0B)
    wB_z = gaussian_w(z, w0B, zRB, z0B)
    curveB.set_ydata(wB_z)
    # lower envelope
    ax.lines[3].set_ydata(-wB_z)

    # Update overlap at rectangle center (z=0)
    eta = overlap_efficiency(0, A_params, B_params)
    eta_text.set_text(f"Overlap: {eta*100:.3f}%")

    fig.canvas.draw_idle()

slider_w0B.on_changed(update)
slider_zRB.on_changed(update)
slider_z0B.on_changed(update)

plt.show()
