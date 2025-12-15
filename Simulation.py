import numpy as np
import tomllib
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

with open("System_Params.toml", "rb") as f:
    system_params = tomllib.load(f)

crystal_length = system_params["crystal_length"]  # mm
optimal_efficiency = system_params["optimal_efficiency"]  # %/Wcm


def optimal_params():
    with open("System_Params.toml", "rb") as f:
        system_params = tomllib.load(f)

    wavelength = system_params["wavelength"]*1e-6  # in mm
    crystal_length = system_params["crystal_length"]

    confocal_param = 2.84
    optimal_rayleigh = crystal_length/ (2*confocal_param)
    optimal_waist = math.sqrt((optimal_rayleigh*wavelength)/math.pi)

    return optimal_rayleigh, optimal_waist

def optimal_power():
    power_in = slider_pw.val  # mW
    power_eff = (optimal_efficiency/100) * crystal_length
    power_eff_pump = power_eff/100 * power_in
    power_out = power_eff_pump/100 * power_in

    return power_out

def current_power():
    power_in = slider_pw.val  # mW
    power_eff = (optimal_efficiency / 100) * crystal_length
    power_eff_pump = power_eff / 100 * power_in
    power_out = power_eff_pump / 100 * power_in

    return power_out*eta0


# ----------------------------------------------------------
# Gaussian beam radius function
# ----------------------------------------------------------
def gaussian_w(z, w0, zR, z0 = 0):
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
def overlap_efficiency(z, A_params, B_params, grid_size=4.0, resolution=600): # calculates once at z, (zRB irrelevant when z = 0)
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


def average_overlap_through_crystal(A_params, B_params,
                                    crystal_length=20,
                                    N=20,  # number of slices
                                    grid_size=4.0,
                                    resolution=400):
    """
    Compute the average overlap efficiency over the crystal length.

    A_params, B_params = (w0, zR, z0)
    crystal_length = total length of crystal in mm
    """
    # Slice positions (z values)
    z_vals = np.linspace(-crystal_length / 2, +crystal_length / 2, N)

    overlaps = []

    for z in z_vals:
        overlaps.append(
            overlap_efficiency(z, A_params, B_params,
                               grid_size=grid_size,
                               resolution=resolution)
        )

    return np.mean(overlaps)


# ----------------------------------------------------------
# Fixed Beam A (Optimal Beam)
# ----------------------------------------------------------
z0A = 0  # waist at rectangle center
zRA, w0A = optimal_params()
A_params = (w0A, zRA, z0A)
pw_init = 20 # mW

# ----------------------------------------------------------
# Initial Beam B
# ----------------------------------------------------------
w0B_init = w0A
z0B_init = 0
zRB_init = zRA

# ----------------------------------------------------------
# Plot setup
# ----------------------------------------------------------
z_min, z_max = -200, 200  # propagation along z
z = np.linspace(z_min, z_max, 400)

fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(bottom=0.35)
ax.set_title("SHG Efficiency by Deviation from Optimal")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Beam radius w(z) (mm)")

# Rectangle (crystal) in center
rect_length = 200 # mm
rect_height = 200 # mm
rect = Rectangle((-rect_length/2, -rect_height/2), rect_length, rect_height,
                 facecolor='skyblue', alpha=0.5)
ax.add_patch(rect)
# Channel in center
channel_height = rect_height/10 # mm
channel = Rectangle((-rect_length/2, -channel_height/2), rect_length, channel_height,
                 facecolor='snow', alpha=0.5)
ax.add_patch(channel)

# Initial beam curves
wA_z = gaussian_w(z, w0A, zRA, z0A)
wB_z = gaussian_w(z, w0B_init, zRB_init, z0B_init)

curveA, = ax.plot(z, wA_z, 'r', label='Optimal Beam', linewidth=1)
ax.plot(z, -wA_z*10, 'r', linewidth=1)  # lower envelope
curveA.set_ydata(wA_z*10)



curveB, = ax.plot(z, wB_z, 'g', label='Adjusted Beam', linewidth=1)
ax.plot(z, -wB_z*10, 'g', linewidth=1)
curveB.set_ydata(wB_z*10)


ax.legend()
ax.set_xlim(z_min, z_max)
ax.set_ylim(-200, 200)

# Overlap text
eta0 = average_overlap_through_crystal( A_params, (w0B_init, zRB_init, z0B_init))
eta_text = ax.text(-195, 180, f"Overlap: {eta0*100:.3f}%", ha='left', fontsize=8)


# ----------------------------------------------------------
# Sliders for Beam B parameters
# ----------------------------------------------------------
ax_w0B = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_zRB = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_pw = plt.axes([0.2, 0.15, 0.65, 0.03])

slider_w0B = Slider(ax_w0B, "Beam B waist w0 (mm)", 0, 1.5, valinit=w0B_init)
slider_zRB = Slider(ax_zRB, "Rayleigh length zR_B (mm)", 0, 50, valinit=zRB_init)
slider_pw = Slider(ax_pw, "Pump Power (mW)", 0, 50, valinit=pw_init)

pw_op_text = ax.text(-195, 155, f"Pump power: {pw_init:.1f} mW -> Optimal output power: {optimal_power()*1000:.3f} μW", ha='left', fontsize=8)
pw_curr_text = ax.text(-195, 130, f"Pump power: {pw_init:.1f} mW -> Current output power: {current_power()*1000:.3f} μW", ha='left', fontsize=8)
op_param_text = ax.text(-195, 105, f"Optimal Rayleigh: {A_params[1]:.3f}, Optimal Waist: {A_params[0]:.3f}", ha='left', fontsize=8)

# ----------------------------------------------------------
# Slider update function
# ----------------------------------------------------------
def update(val):
    w0B = slider_w0B.val
    zRB = slider_zRB.val
    pw = slider_pw.val

    B_params = (w0B, zRB, 0)
    wB_z = gaussian_w(z, w0B, zRB, 0)
    curveB.set_ydata(wB_z*10)
    # lower envelope
    ax.lines[3].set_ydata(-wB_z*10)

    # Update overlap at rectangle center (z=0)
    eta = average_overlap_through_crystal( A_params, B_params)
    op = optimal_power()
    cp = op*eta
    eta_text.set_text(f"Overlap: {eta*100:.3f}%")
    pw_op_text.set_text(f"Pump power: {pw:.1f} mW-> Optimal output power: {op*1000:.3f} μW")
    pw_curr_text.set_text(f"Pump power: {pw:.1f} mW -> Current output power: {cp*1000:.3f} μW")

    fig.canvas.draw_idle()

slider_w0B.on_changed(update)
slider_zRB.on_changed(update)
slider_pw.on_changed(update)

plt.show()
