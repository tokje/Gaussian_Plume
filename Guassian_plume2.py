"""
E230 - Group Assignment 5
Gaussian Plume Dispersion Model

PURPOSE:
This program investigates how atmospheric stability, smokestack
height, and wind speed affect pollutant concentrations at ground level.

The calculations follow the Gaussian plume equation and the
Pasquill-Gifford dispersion parameters in the assignment.

THE PROGRAM (cases refer to Table 2 in the report):
1. Implements the Gaussian plume model.
2. Case 1: compares atmospheric stability classes A-F.
3. Case 2: investigates stack heights of 25, 50 and 100 m
   with a logarithmic wind speed profile.
4. Case 3: calculates the relative error of assuming a constant
   wind speed, for z0 = 0.03 m and z0 = 1 m.
5. Case 4: calculates the concentration at x = 6 km, the
   straight-line distance from Loddefjord to Bergen centre.
6. Saves Figures 2 and 3 of the report in a folder named "figures".

UNITS:
Emission rate: g/s
Wind speed: m/s
Distance and height: m
Pollutant concentration: ug/m3
"""

# ============================================================
# 1. IMPORT LIBRARIES AND CREATE OUTPUT FOLDER
# ============================================================

# NumPy provides mathematical functions and array calculations.
import numpy as np

# Matplotlib creates the graphs.
import matplotlib.pyplot as plt

# Path helps us create folders and save files.
from pathlib import Path

# Find the folder containing this Python program.
program_folder = Path(__file__).resolve().parent

# Create a folder named "figures" beside the Python program.
output_folder = program_folder / "figures"

# Create the folder if it does not already exist.
# exist_ok=True prevents an error if the folder exists.
output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ATMOSPHERIC DISPERSION PARAMETERS
# ============================================================

def dispersion_parameters(x, stability):
    """
    Calculate horizontal and vertical dispersion parameters.

    INPUTS:
    x         : Downwind distance (m).
    stability : Atmospheric stability class A-F.

    OUTPUTS:
    sigma_y : Horizontal dispersion parameter (m).
    sigma_z : Vertical dispersion parameter (m).

    These parameters describe how the pollutant plume spreads
    horizontally and vertically as it travels downwind.

    The equations come from Table 1 in the assignment.
    """

    # Each stability class has two coefficients:
    # the first for horizontal dispersion,
    # and the second for vertical dispersion.
    coefficients = {
        "A": (0.22, 0.20),
        "B": (0.16, 0.12),
        "C": (0.11, 0.08),
        "D": (0.08, 0.06),
        "E": (0.06, 0.03),
        "F": (0.04, 0.016)
    }

    # Check whether the requested stability class is valid.
    if stability not in coefficients:
        raise ValueError("Stability must be A, B, C, D, E or F")

    # Retrieve the coefficients for the selected stability class.
    a, b = coefficients[stability]

    # Calculate horizontal dispersion.
    # All stability classes use the same general equation,
    # but with different coefficients.
    sigma_y = a * x / np.sqrt(1 + 0.0001 * x)

    # Calculate vertical dispersion.
    # The formula depends on the atmospheric stability class.

    if stability in ("A", "B"):
        # Very unstable and moderately unstable conditions.
        sigma_z = b * x

    elif stability == "C":
        # Slightly unstable conditions.
        sigma_z = b * x / np.sqrt(1 + 0.0002 * x)

    elif stability == "D":
        # Neutral atmospheric conditions.
        sigma_z = b * x / np.sqrt(1 + 0.0015 * x)

    else:
        # Slightly stable and stable conditions (E and F).
        sigma_z = b * x / (1 + 0.0003 * x)

    # Return both dispersion parameters.
    return sigma_y, sigma_z


# ============================================================
# 3. GAUSSIAN PLUME MODEL
# ============================================================

def gaussian_plume(S0, u, h, x, y, z, stability):
    """
    Calculate pollutant concentration using the Gaussian plume model.

    INPUTS:
    S0        : Emission rate (g/s).
    u         : Wind speed at effective source height (m/s).
    h         : Effective source height (m).
    x         : Downwind distance (m).
    y         : Crosswind distance from plume centreline (m).
    z         : Height above ground (m).
    stability : Atmospheric stability class A-F.

    OUTPUT:
    Pollutant concentration (ug/m3).

    ASSUMPTIONS:
    - Continuous emissions from a single point source.
    - Steady-state atmospheric conditions.
    - Constant horizontal wind direction.
    - Flat, unobstructed terrain.
    - No chemical reactions or pollutant removal.
    - Pollutants are reflected at ground level.
    """

    # Convert x to a NumPy array so the function can handle
    # both individual distances and multiple distances.
    x = np.asarray(x, dtype=float)

    # The Gaussian plume equation is undefined at x = 0
    # because the dispersion parameters become zero.
    if np.any(x <= 0):
        raise ValueError("Downwind distance must be positive")

    # Wind speed must be positive.
    if u <= 0:
        raise ValueError("Wind speed must be greater than zero")

    # Emission rate cannot be negative.
    if S0 < 0:
        raise ValueError("Emission rate cannot be negative")

    # Calculate horizontal and vertical dispersion.
    sigma_y, sigma_z = dispersion_parameters(x, stability)

    # PART A:
    # Describes the effect of emission rate, wind speed,
    # and the amount of horizontal and vertical spreading.
    A = S0 / (2 * np.pi * u * sigma_y * sigma_z)

    # PART B:
    # Describes horizontal spreading away from the centreline.
    # At y = 0, this factor equals 1.
    B = np.exp(-y**2 / (2 * sigma_y**2))

    # PART D:
    # Describes vertical dispersion.
    #
    # The first exponential represents the actual plume.
    # The second represents reflection at the ground.
    D = (
        np.exp(-(z - h)**2 / (2 * sigma_z**2))
        + np.exp(-(z + h)**2 / (2 * sigma_z**2))
    )

    # Multiply all three parts of the Gaussian plume equation.
    # The resulting concentration is initially in g/m3.
    concentration = A * B * D

    # Convert g/m3 into ug/m3.
    return concentration * 1_000_000


# ============================================================
# 4. CASE 1: ATMOSPHERIC STABILITY CLASSES A-F
# ============================================================

# Use the parameters specified in the assignment.
S0 = 5       # Emission rate (g/s).
u = 2        # Wind speed (m/s).
h = 30       # Effective stack height (m).
y = 0        # Plume centreline (m).
z = 0        # Ground level (m).

# Create 2500 distance values between 1 and 2500 metres.
# We exclude x = 0 because the equation is undefined there.
x = np.linspace(1, 2500, 2500)

# Define all six atmospheric stability classes.
stability_classes = [
    ("A", "very unstable"),
    ("B", "moderately unstable"),
    ("C", "slightly unstable"),
    ("D", "neutral"),
    ("E", "slightly stable"),
    ("F", "stable"),
]

# Create a new figure.
plt.figure(figsize=(10, 6))

print("\nCASE 1: ATMOSPHERIC STABILITY")
print("-" * 65)

# Repeat the calculations for every stability class.
for stability, description in stability_classes:

    # Calculate pollutant concentration at every distance.
    C = gaussian_plume(S0, u, h, x, y, z, stability)

    # Draw the concentration curve for this stability class.
    plt.plot(x, C, label=f"{stability}: {description}")

    # Find the index of the highest calculated concentration.
    maximum_index = np.argmax(C)

    # Retrieve the maximum concentration.
    maximum_concentration = C[maximum_index]

    # Find the distance where the maximum occurs.
    maximum_distance = x[maximum_index]

    # Print the results for this stability class.
    print(
        f"Class {stability} ({description}): "
        f"Maximum = {maximum_concentration:.1f} ug/m3, "
        f"Distance = {maximum_distance:.1f} m"
    )



# Label the horizontal and vertical axes.
plt.xlabel("Downwind distance, $x$ (m)",fontsize=15)
plt.ylabel("Ground-level concentration, $C$ (µg m$^{−3}$)",fontsize=15)

# Display a legend identifying the six stability classes.
plt.legend(fontsize=13)

# Add grid lines to improve readability.
plt.grid(True, alpha=0.3)

# Set the horizontal axis to the required range.
plt.xlim(0, 2500)

plt.ylim(-5,600)
# Automatically adjust spacing around the figure.
plt.tight_layout()

# Define the file path for Figure 2 in the report.
figure1_path = output_folder / "figure1_stability2.pdf"

# Save Figure 1 as a vector PDF suitable for a scientific report.
# bbox_inches="tight" prevents labels from being cut off.
plt.savefig(figure1_path, bbox_inches="tight")

# Confirm where the figure was saved.
print(f"\nFigure 2 (report) saved to: {figure1_path}")

# Display the figure.
plt.show()


# ============================================================
# 5. LOGARITHMIC WIND SPEED PROFILE
# ============================================================

def logarithmic_wind_speed(h, u_ref=10, z_ref=10, z0=0.03):
    """
    Calculate wind speed at a specified height.

    INPUTS:
    h     : Height above ground (m).
    u_ref : Wind speed at reference height (m/s).
    z_ref : Reference height (m).
    z0    : Surface roughness length (m).

    OUTPUT:
    Wind speed at height h (m/s).

    The logarithmic wind profile describes how wind speed
    increases with height due to decreasing surface friction.

    It is used here for neutral atmospheric conditions.
    """

    # Both heights must exceed the roughness length.
    if h <= z0 or z_ref <= z0:
        raise ValueError("Heights must exceed roughness length")

    # Calculate wind speed at the requested height.
    u_h = u_ref * np.log(h / z0) / np.log(z_ref / z0)

    # Return the calculated wind speed.
    return u_h


# ============================================================
# 6. CASE 2: INFLUENCE OF STACK HEIGHT
# ============================================================

# Define the three effective stack heights in the assignment.
stack_heights = [25, 50, 100]

# Use neutral atmospheric stability (class D).
stability = "D"

# Assume the same emission rate as Case 1.
# This assumption must be mentioned in the report.
S0 = 5

# Generate distances from 1 to 3000 metres.
x = np.linspace(1, 3000, 3000)

# Create a new figure.
plt.figure(figsize=(10, 6))

print("\nCASE 2: STACK HEIGHT")
print("-" * 65)

# Calculate the concentration for each stack height.
for h in stack_heights:

    # Calculate wind speed at the effective stack height.
    u = logarithmic_wind_speed(h)

    # Calculate ground-level concentration at every distance.
    C = gaussian_plume(S0, u, h, x, 0, 0, stability)

    # Plot the concentration curve.
    plt.plot(x, C, label=f"h = {h} m, u = {u:.1f} m s$^{{-1}}$")

    # Find the index of the maximum concentration.
    maximum_index = np.argmax(C)

    # Retrieve the maximum concentration.
    maximum_concentration = C[maximum_index]

    # Find the corresponding distance.
    maximum_distance = x[maximum_index]

    # Print the results for this stack height.
    print(
        f"Height = {h} m: "
        f"Wind speed = {u:.2f} m/s, "
        f"Maximum = {maximum_concentration:.1f} ug/m3, "
        f"Distance = {maximum_distance:.1f} m"
    )

# Add a descriptive title.

# Label both axes.
plt.xlabel("Downwind distance, $x$ (m)", fontsize=15)
plt.ylabel("Ground-level concentration, $C$ (µg m$^{−3}$)", fontsize=15)

# Display a legend showing the stack heights.
plt.legend(fontsize=13)

# Add grid lines.
plt.grid(True, alpha=0.3)

# Set the required horizontal range.
plt.xlim(0, 3000)

plt.ylim(-2,100)

# Adjust the figure layout.
plt.tight_layout()

# Define the path for Figure 3 in the report.
figure2_path = output_folder / "figure2_stack_height2.pdf"

# Save Figure 2 at high resolution.
plt.savefig(figure2_path, bbox_inches="tight")

# Confirm where the figure was saved.
print(f"\nFigure 3 (report) saved to: {figure2_path}")

# Display the figure.
plt.show()


# ============================================================
# 7. CASE 3: RELATIVE ERROR OF A CONSTANT WIND SPEED
# ============================================================

# The client wants to assume a constant wind speed of 10 m/s.
#
# We compare concentrations calculated with this assumption
# against concentrations calculated using the logarithmic
# wind profile at a distance of 1500 metres, for open terrain
# (z0 = 0.03 m) and urban terrain (z0 = 1 m).

# Define the distance at which the error is evaluated.
x_error = 1500

# Define the incorrectly assumed constant wind speed.
u_constant = 10

# Roughness lengths: farmland (assignment) and urban terrain.
roughness_lengths = [0.03, 1.0]

print("\nCASE 3: RELATIVE ERROR AT 1500 M (class D)")
print("-" * 65)

# Repeat the comparison for each roughness length and stack height.
for z0 in roughness_lengths:

    print(f"\nRoughness length z0 = {z0} m")

    for h in stack_heights:

        # Calculate the logarithmic wind speed at this height.
        u_actual = logarithmic_wind_speed(h, z0=z0)

        # Calculate concentration using the logarithmic profile.
        C_actual = gaussian_plume(
            S0, u_actual, h, x_error, 0, 0, "D"
        )

        # Calculate concentration using constant wind speed.
        C_constant = gaussian_plume(
            S0, u_constant, h, x_error, 0, 0, "D"
        )

        # Calculate relative error in percent (Eq. 9 in the report).
        #
        # Positive error means the constant wind speed
        # assumption overestimates pollutant concentration.
        relative_error = (
            (C_constant - C_actual) / C_actual
        ) * 100

        # Print the results for this stack height.
        print(
            f"h = {h:3d} m: u = {u_actual:5.2f} m/s, "
            f"C_log = {C_actual:6.2f} ug/m3, "
            f"C_const = {C_constant:6.2f} ug/m3, "
            f"error = {relative_error:+6.1f} %"
        )


# ============================================================
# 8. CASE 4: CONCENTRATION IN BERGEN CENTRE (x = 6 km)
# ============================================================

# Loddefjord lies about 6 km (straight line) from Bergen centre.
x_centre = 6000

# Wind speed used for all stability classes.
u_case4 = 2

print("\nCASE 4: CONCENTRATION AT x = 6 km")
print("-" * 65)
print("Class      h = 25 m   h = 50 m   h = 100 m   (ug/m3, u = 2 m/s)")

# Concentration for each stability class and stack height.
for stability, description in stability_classes:

    concentrations = [
        gaussian_plume(S0, u_case4, h, x_centre, 0, 0, stability)
        for h in stack_heights
    ]

    # Vertical dispersion at 6 km, used to interpret the results.
    sigma_y, sigma_z = dispersion_parameters(x_centre, stability)

    print(
        f"{stability:<9}"
        + "".join(f"{C:10.1f}" for C in concentrations)
        + f"    (sigma_z = {sigma_z:.0f} m)"
    )

# Neutral conditions with the logarithmic wind profile (z0 = 0.03 m).
concentrations = [
    gaussian_plume(S0, logarithmic_wind_speed(h), h, x_centre, 0, 0, "D")
    for h in stack_heights
]
print(f"{'D, log':<9}" + "".join(f"{C:10.1f}" for C in concentrations)
      + "    (logarithmic wind profile)")


# ============================================================
# 9. END OF PROGRAM
# ============================================================

print("\nAll calculations completed successfully.")
print("\nThe following figures have been saved:")

# Print the complete file paths so the images are easy to find.
print(f"Figure 2 (report): {figure1_path}")
print(f"Figure 3 (report): {figure2_path}")