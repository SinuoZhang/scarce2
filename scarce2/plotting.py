import fipy
from matplotlib import pyplot as plt
from matplotlib import colormaps
from scipy import interpolate
import numpy as np

CMAP = colormaps["coolwarm"]  # Choose colormap


def plot_potential(
    pot: fipy.CellVariable,
    n_pixel: int,
    plot_title: str = "Potential",
    colorbar_label: str = "",
):
    if not pot.solved:
        raise RuntimeWarning("Potential has not been solved yet!")
    # Interpolation
    X = np.linspace(min(pot.mesh.x), max(pot.mesh.x), 500)
    Y = np.linspace(min(pot.mesh.y), max(pot.mesh.y), 500)
    xx, yy = np.meshgrid(X, Y)
    potential = interpolate.griddata(
        np.transpose(pot.mesh.faceCenters),
        pot.arithmeticFaceValue,
        (xx, yy),
        method="linear",
    )

    E_x = interpolate.griddata(
        np.transpose(pot.mesh.faceCenters),
        pot.grad.arithmeticFaceValue[0],
        (xx, yy),
        method="linear",
    )
    E_y = interpolate.griddata(
        np.transpose(pot.mesh.faceCenters),
        pot.grad.arithmeticFaceValue[1],
        (xx, yy),
        method="linear",
    )

    # If potential grid has nan values from interpolation, fill with closest finite value
    nan_mask = np.isnan(potential)
    potential[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), potential[~nan_mask])

    aspect = pot.mesh.aspect2D
    fig_width = n_pixel
    fig, ax = plt.subplots(figsize=(fig_width, aspect * fig_width))
    fig.set_layout_engine("compressed")

    # Plot equipotential lines
    ax.contour(xx, yy, potential, 10, colors="black", linestyles="dashed", linewidths=1)

    # Plot potential strength as color
    im = ax.pcolormesh(xx, yy, potential, cmap=CMAP, rasterized=True)

    # Plot field lines
    E_tot = np.sqrt(E_x**2 + E_y**2)
    linewidth = 50 * E_tot / E_tot.max()
    ax.streamplot(xx, yy, E_x, E_y, linewidth=0.75, color="darkgray", arrowstyle="-")
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)

    ax.set_title(plot_title)
    ax.set_xlabel("Width [µm]")
    ax.set_ylabel("Depth [µm]")
    plt.show()


def plot_field(
    pot: fipy.CellVariable,
    n_pixel: int,
    plot_title: str = "Potential",
    colorbar_label: str = "",
):
    if not pot.solved:
        raise RuntimeWarning("Potential has not been solved yet!")
    # Interpolation
    X = np.linspace(min(pot.mesh.x), max(pot.mesh.x), 500)
    Y = np.linspace(min(pot.mesh.y), max(pot.mesh.y), 500)
    xx, yy = np.meshgrid(X, Y)
    potential = interpolate.griddata(
        np.transpose(pot.mesh.faceCenters),
        pot.arithmeticFaceValue,
        (xx, yy),
        method="linear",
    )

    E_x = interpolate.griddata(
        np.transpose(pot.mesh.faceCenters),
        pot.grad.arithmeticFaceValue[0],
        (xx, yy),
        method="linear",
    )
    E_y = interpolate.griddata(
        np.transpose(pot.mesh.faceCenters),
        pot.grad.arithmeticFaceValue[1],
        (xx, yy),
        method="linear",
    )

    # If potential grid has nan values from interpolation, fill with closest finite value
    nan_mask = np.isnan(potential)
    potential[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), potential[~nan_mask])

    E_tot = np.sqrt(E_x**2 + E_y**2)

    aspect = pot.mesh.aspect2D
    fig_width = n_pixel
    fig, ax = plt.subplots(figsize=(fig_width, aspect * fig_width))
    fig.set_layout_engine("compressed")

    # Plot equipotential lines
    # ax.contour(xx, yy, E_tot, 10, colors="black", linestyles="dashed", linewidths=1)

    # Plot potential strength as color
    im = ax.pcolormesh(xx, yy, E_tot, cmap=CMAP, rasterized=True)

    # Plot field lines
    # E_tot = np.sqrt(E_x**2 + E_y**2)
    # linewidth = 50 * E_tot / E_tot.max()
    ax.streamplot(xx, yy, E_x, E_y, linewidth=0.75, color="darkgray", arrowstyle="-")
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)

    ax.set_title(plot_title)
    ax.set_xlabel("Width [µm]")
    ax.set_ylabel("Depth [µm]")
    plt.show()
