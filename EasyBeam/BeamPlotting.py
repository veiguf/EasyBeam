import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mplcollect
import pyvista

# import matplotlib.colors as colors
import numpy as np


def _plotting2D(self, val, disp, title, colormap):

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect("equal")

    c = np.linspace(val.min(), val.max(), 5)
    norm = MidpointNormalizeNew(vmin=-np.abs(val).max(), vmax=np.abs(val).max())
    lcAll = colorline(
        disp[:, 0, :], disp[:, 1, :], val, cmap=colormap, plot=False, norm=norm
    )

    for i in range(self.nEl):
        xEl = self.Nodes[self.El[i, 0] - 1, 0], self.Nodes[self.El[i, 1] - 1, 0]
        yEl = self.Nodes[self.El[i, 0] - 1, 1], self.Nodes[self.El[i, 1] - 1, 1]
        plt.plot(xEl, yEl, c="gray", lw=0.5, ls=self.lineStyleUndeformed, clip_on=False)
    for i in range(self.nEl):
        lc = colorline(
            disp[i, 0, :], disp[i, 1, :], val[i, :], cmap=colormap, norm=lcAll.norm
        )

    cb = plt.colorbar(
         lcAll,
         ticks=c,
         shrink=0.5,
         ax=[ax],
         location="left",
         aspect=10,
         boundaries=np.linspace(val.min()-1e-9, val.max()+1e-9, 100),
    )
    cb.outline.set_visible(False)
    cb.set_label(title, labelpad=0, y=1.1, rotation=0, ha="left")
    # cb = plt.colorbar(lcAll, ticks=c, shrink=0.4, orientation="horizontal")
    xmin = disp[:, 0, :].min() - 1
    xmax = disp[:, 0, :].max() + 1
    ymin = disp[:, 1, :].min() - 1
    ymax = disp[:, 1, :].max() + 1
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    buff = 0.1
    plt.xlim(xmin - xdelta * buff, xmax + xdelta * buff)
    plt.ylim(ymin - ydelta * buff, ymax + ydelta * buff)
    # cb.ax.set_title(title)
    plt.show()

def _plotting3D(self, val, disp, title, colormap):

    pyvista.global_theme.axes.box = False
    pyvista.global_theme.axes.x_color = 'black'
    pyvista.global_theme.axes.y_color = 'black'
    pyvista.global_theme.axes.z_color = 'black'
    pyvista.global_theme.font.color = 'black'

    grid = pyvista.StructuredGrid(disp[:, 0, :].flatten(order="C"),
                                  disp[:, 1, :].flatten(order="C"),
                                  disp[:, 2, :].flatten(order="C"))
    grid.cell_data[title] = val.flatten(order="C")[:-1]
    grid.plot(off_screen=False,
              full_screen=False,
              interactive=True,
              parallel_projection=True,
              show_axes=True,
              show_bounds=False,
              # scalars=colors,
              render_lines_as_tubes=True,
              style='wireframe',
              line_width=10,
              cmap=colormap,
              lighting='three lights',
              show_scalar_bar=True,
              background='w')

def PlotStress(self, points="all", stress="all", scale=1):
    if not self.ComputedStress:
        self.ComputeStress()
    self.rS = self.r0S + self.uS * scale

    if points == "all":  # points should be "all" or a list of section points
        points = np.arange(0, self.nSec, 1).tolist()
    if self.nNDoF == 3:
        position = ["neutral fiber", "upper fiber", "lower fiber"]
        prefix = ["maximum stress\n$", "equivalent stress\n",
                  "axial stress\n", "bending stress\n"]
        suffix = ["|\\sigma_{max}|$ [MPa]", "\n$|\\sigma_{eqv}|$ [MPa]",
                  "\n$\\sigma_{ax}$ [MPa]", "\n$\\sigma_{b}$ [MPa]"]
    elif self.nNDoF == 6:
        position = ["neutral fiber", "central-right fiber", "upper-right fiber",
                    "upper-central fiber", "upper-left fiber", "central-left fiber",
                    "lower-left fiber", "lower-central fiber", "lower-right fiber"]
        prefix = ["maximum stress", "equivalent stress on ",
                  "axial stress on ", "bending stress in y on ",
                  "bending stress in z on ", "torsional stress on "]
        suffix = [" in MPa", " in MPa", " in MPa", " in MPa", " in MPa", " in MPa"]

    if stress.lower() in ["all", "max"]:
        self._plotting(
            self.sigmaEqvMax,
            self.rS,
            prefix[0]+suffix[0],
            self.colormap,
        )
    for i in points:
        if stress.lower() in ["all", "equivalent"]:
            self._plotting(
                self.sigmaEqv[:, :, i],
                self.rS,
                prefix[1]+position[i]+suffix[1],
                self.colormap,
            )
        if stress.lower() in ["all", "axial"]:
            self._plotting(
                self.sigma[:, :, i, 0],
                self.rS,
                prefix[2]+position[i]+suffix[2],
                self.colormap,
            )
        if stress.lower() in ["all", "bending", "bending_y"]:
            self._plotting(
                self.sigma[:, :, i, 1],
                self.rS,
                prefix[3]+position[i]+suffix[3],
                self.colormap,
            )
        if stress.lower() in ["all", "bending", "bending_z"] and self.nSVal == 4:
            self._plotting(
                self.sigma[:, :, i, 2],
                self.rS,
                prefix[4]+position[i]+suffix[4],
                self.colormap,
            )
        if stress.lower() in ["all", "torsional"] and self.nSVal == 4:
            self._plotting(
                self.sigma[:, :, i, 3],
                self.rS,
                prefix[5]+position[i]+suffix[5],
                self.colormap,
            )


def PlotDisplacement(self, component="all", scale=1):
    if not self.ComputedDisplacement:
        self.ComputeDisplacement()
    self.rS = self.r0S + self.uS * scale

    if self.nNPoC == 2:
        label = ["deformation\nmagnitude\n$|u|$ [mm]",
                 "$x$-deformation\n$u_x$ [mm]", "$y$-deformation\n$u_y$ [mm]"]
    elif self.nNPoC == 3:
        label = ["deformation magnitude in mm", "x-deformation in mm",
                 "y-deformation in mm", "z-deformation in mm"]

    if component.lower() in ["mag", "all"]:
        self._plotting(
            self.uSmag, self.rS, label[0], self.colormap
        )
    if component.lower() in ["x", "all"]:
        self._plotting(
            self.uS[:, 0, :], self.rS, label[1], self.colormap
        )
    if component.lower() in ["y", "all"]:
        self._plotting(
            self.uS[:, 1, :], self.rS, label[2], self.colormap
        )
    if component.lower() in ["z", "all"] and self.nNPoC == 3:
        self._plotting(
            self.uS[:, 2, :], self.rS, label[3], self.colormap
        )

def PlotInternalForces(self, scale=1):
    if not self.ComputedInternalForces:
        self.ComputeInternalForces()
    self.rS = self.r0S + self.uS * scale

    if self.nNDoF == 3:
        label = ["Normal force\n$F_N$ [N]", "Shear force\n$F_Q$ [N]", "Bending moment\n$M_b$ [Nmm]"]
    elif self.nNDoF == 6:
        label = ["Normal force in N", "Shear force in y in N",
                 "Shear force in z in N", "Torsional moment in Nmm",
                 "Bending moment in y in Nmm", "Bending moment in z in Nmm"]

    for i in range(self.nNDoF):
        self._plotting(
            self.QS[:, :, i],
            self.rS,
            label[i],
            self.colormap,
        )


def PlotMode(self, scale=1):
    Phii = np.zeros([self.nNDoF*self.nN])
    for ii in range(len(self.omega)):
        Phii[self.DoF] = self.Phi[:, ii]
        uE_Phi = np.zeros([self.nEl, 2*self.nNDoF])
        uS_Phi = np.zeros([self.nEl, self.nNPoC, self.nSeg + 1])
        for i in range(self.nEl):
            uE_Phi[i, :] = Phii[self.idx[i]]
            for j in range(self.nSeg + 1):
                ξ = j / (self.nSeg)
                S = self.ShapeMat(ξ, self.ell[i])
                uS_Phi[i, :, j] = self.TX[i] @ S @ self.T[i] @ uE_Phi[i, :]
        # deformation
        rPhi = self.r0S + uS_Phi * scale
        if self.nNDoF == 3:
            dPhi = np.sqrt(uS_Phi[:, 0, :]**2+uS_Phi[:, 1, :]**2)
        elif self.nNDoF == 6:
            dPhi = np.sqrt(uS_Phi[:, 0, :]**2+uS_Phi[:, 1, :]**2+uS_Phi[:, 2, :]**2)
        self._plotting(
            dPhi,
            rPhi,
            ("mode " + str(ii + 1) + " at " + str(round(self.f0[ii], 4)) + " Hz"),
            self.colormap,
        )


def PlotMesh2D(self, NodeNumber=True, ElementNumber=True, Loads=True, BC=True, FontMag=1):
    if not self.Initialized:
        self.Initialize()
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect("equal")
    deltaMax = max(
        self.Nodes[:, 0].max() - self.Nodes[:, 0].min(),
        self.Nodes[:, 1].max() - self.Nodes[:, 1].min(),
    )
    p = deltaMax * 0.0075
    for i in range(self.nEl):
        xEl = self.Nodes[self.El[i, 0] - 1, 0], self.Nodes[self.El[i, 1] - 1, 0]
        yEl = self.Nodes[self.El[i, 0] - 1, 1], self.Nodes[self.El[i, 1] - 1, 1]
        plt.plot(xEl, yEl, c="gray", lw=self.A[i] / np.max(self.A), ls="-")
    plt.plot(self.Nodes[:, 0], self.Nodes[:, 1], ".k", clip_on=False)
    if NodeNumber:
        for i in range(self.nN):
            ax.annotate(
                "N" + str(i + 1),
                (self.Nodes[i, 0] + p, self.Nodes[i, 1] + p),
                fontsize=5 * FontMag,
                clip_on=False,
            )
    if ElementNumber:
        for i in range(self.nEl):
            posx = (
                self.Nodes[self.El[i, 0] - 1, 0] + self.Nodes[self.El[i, 1] - 1, 0]
            ) / 2
            posy = (
                self.Nodes[self.El[i, 0] - 1, 1] + self.Nodes[self.El[i, 1] - 1, 1]
            ) / 2
            ax.annotate(
                "E" + str(i + 1),
                (posx + p, posy + p),
                fontsize=5 * FontMag,
                c="gray",
                clip_on=False,
            )
    if Loads:
        note = [r"$F_x$", r"$F_y$", r"$M$"]
        for i in range(len(self.Load)):
            comment = ""
            for ii in range(3):
                if isinstance(self.Load[i][1][ii], int) or isinstance(
                    self.Load[i][1][ii], float
                ):
                    if self.Load[i][1][ii] != 0:
                        comment += note[ii]
            ax.annotate(
                comment,
                (
                    self.Nodes[self.Load[i][0] - 1, 0] + p,
                    self.Nodes[self.Load[i][0] - 1, 1] - p,
                ),
                fontsize=5 * FontMag,
                c="red",
                clip_on=False,
                ha="left",
                va="top",
            )
    if BC:
        noteBC = [r"$x_f$", r"$y_f$", r"$\theta_f$"]
        noteDL = [r"$x_d$", r"$y_d$", r"$\theta_d$"]
        for i in range(len(self.Disp)):
            commentBC = ""
            commentDL = ""
            for ii in range(3):
                if isinstance(self.Disp[i][1][ii], int) or isinstance(
                    self.Disp[i][1][ii], float
                ):
                    if self.Disp[i][1][ii] == 0:
                        commentBC += noteBC[ii]
                    else:
                        commentDL += noteDL[ii]
            ax.annotate(
                commentBC,
                (
                    self.Nodes[self.Disp[i][0] - 1, 0] - p,
                    self.Nodes[self.Disp[i][0] - 1, 1] - p,
                ),
                fontsize=5 * FontMag,
                c="green",
                clip_on=False,
                ha="right",
                va="top",
            )
            ax.annotate(
                commentDL,
                (
                    self.Nodes[self.Disp[i][0] - 1, 0] - p,
                    self.Nodes[self.Disp[i][0] - 1, 1] + p,
                ),
                fontsize=5 * FontMag,
                c="blue",
                clip_on=False,
                ha="right",
                va="bottom",
            )
    xmin = self.Nodes[:, 0].min()
    xmax = self.Nodes[:, 0].max()
    if self.Nodes[:, 1].max() - self.Nodes[:, 1].min() < 0.1:
        ymin = -10
        ymax = 10
    else:
        ymin = self.Nodes[:, 1].min()
        ymax = self.Nodes[:, 1].max()
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    buff = 0.1
    plt.xlim(xmin - xdelta * buff, xmax + xdelta * buff)
    plt.ylim(ymin - ydelta * buff, ymax + ydelta * buff)
    plt.show()

def PlotMesh3D(self, NodeNumber=True, ElementNumber=True, Loads=True, BC=True, FontMag=1):
    pyvista.set_plot_theme('document')
    pyvista.global_theme.axes.box = False
    pyvista.global_theme.axes.x_color = 'black'
    pyvista.global_theme.axes.y_color = 'black'
    pyvista.global_theme.axes.z_color = 'black'
    pyvista.global_theme.font.color = 'black'
    pyvista.global_theme.transparent_background = True

    plotter = pyvista.Plotter(
        lighting='three lights',
        off_screen=False,
        )

    mesh = pyvista.PolyData(self.Nodes, np.vstack((np.ones(np.array(self.El).shape[0], int)*2, (np.array(self.El)-1).T)).T)
    plotter.add_mesh(
        mesh.copy(),
        render_lines_as_tubes=False,
        line_width=2-5,
        style='wireframe',
        cmap="turbo",
        color="#1f77b4",
        show_scalar_bar=False,
        culling=True,
        )
    plotter.enable_parallel_projection()
    plotter.add_mesh(
        mesh.copy(),
        render_points_as_spheres=True,
        point_size=5,
        style='points',
        cmap="turbo",
        color="#1f77b4",
        show_scalar_bar=False,
        culling=True,
        )
    plotter.show(
        full_screen=False,
        interactive=True,
        )
class MidpointNormalizeNew(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.vmax == 0:
            normalized_min = self.vmin
        else:
            normalized_min = max(
                0,
                1
                / 2
                * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))),
            )
        if self.vmin == 0:
            normalized_max = self.vmax
        else:
            normalized_max = min(
                1,
                1
                / 2
                * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))),
            )
        if self.vmax == 0 and self.vmin == 0:
            normalized_mid = 0
        else:
            normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min,
            normalized_mid,
            normalized_max,
        ]
        return np.ma.masked_array(np.interp(value, x, y))


# class MidpointNormalize(colors.Normalize):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         colors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y))


def colorline(x, y, z, cmap="jet", linewidth=2, alpha=1.0, plot=True, norm=None):
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    segments = make_segments(x, y)
    lc = mplcollect.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )
    if plot:
        ax = plt.gca()
        ax.add_collection(lc)
    return lc


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
