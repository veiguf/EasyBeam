#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:10:25 2020

@author: veit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, g
import matplotlib as mpl

class Beam2D:
    # define the nodes
    N = np.array([[   0, 0],
                  [ 100, 0],
                  [ 200, 0]])
    # define the elements: which nodes are connected?
    El = np.array([[0, 1],
                   [1, 2]])
    # boundary conditions and loads
    BC = []
    Load = []
    nStep = 1
    Scale = 1

    def Initialize(self):

        self.nEl = len(self.El[:, 0])     # number of elements
        self.nN = len(self.N[:, 0])       # number of nodes

        # initial displacements
        self.u = np.empty([3*self.nN,1])
        self.u[:] = np.nan
        self.u[self.BC] = 0

        self.DoF = []   # degrees-of-freedom
        for i in range(3*self.nN):
            if i not in self.BC:
                self.DoF.append(i)

        # initial forces
        self.F = np.empty([3*self.nN,1])
        self.F[:] = np.nan
        self.F[self.DoF] = 0
        for i in range(len(self.Load)):
            self.F[self.Load[i][0]] = self.Load[i][1]

    def Solve(self):
        kL = np.zeros([self.nEl, 6, 6])
        self.l = np.zeros([self.nEl, 1])
        self.θ = np.zeros([self.nEl, 1])
        kG = np.zeros([self.nEl, 6, 6])
        self.T = np.zeros([self.nEl, 6, 6])
        self.k = np.zeros([3*self.nN, 3*self.nN])

        for i in range(self.nEl):
            self.El[i, 1]
            self.l[i] = np.linalg.norm(self.N[self.El[i, 1], :]-self.N[self.El[i, 0], :])
            kL[i, :, :] = np.array([[ self.A[i]*self.E[i]/self.l[i],                                    0,                                   0, -self.A[i]*self.E[i]/self.l[i],                                    0,                                   0],
                                    [                             0,  12*self.E[i]*self.I[i]/self.l[i]**3,  6*self.E[i]*self.I[i]/self.l[i]**2,                              0, -12*self.E[i]*self.I[i]/self.l[i]**3,  6*self.E[i]*self.I[i]/self.l[i]**2],
                                    [                             0,   6*self.E[i]*self.I[i]/self.l[i]**2,     4*self.E[i]*self.I[i]/self.l[i],                              0,  -6*self.E[i]*self.I[i]/self.l[i]**2,     2*self.E[i]*self.I[i]/self.l[i]],
                                    [-self.A[i]*self.E[i]/self.l[i],                                    0,                                   0,  self.A[i]*self.E[i]/self.l[i],                                    0,                                   0],
                                    [                             0, -12*self.E[i]*self.I[i]/self.l[i]**3, -6*self.E[i]*self.I[i]/self.l[i]**2,                              0,  12*self.E[i]*self.I[i]/self.l[i]**3, -6*self.E[i]*self.I[i]/self.l[i]**2],
                                    [                             0,   6*self.E[i]*self.I[i]/self.l[i]**2,     2*self.E[i]*self.I[i]/self.l[i],                              0,  -6*self.E[i]*self.I[i]/self.l[i]**2,     4*self.E[i]*self.I[i]/self.l[i]]])
            if self.N[self.El[i, 1], 0] >= self.N[self.El[i, 0], 0]:
                self.θ[i] = np.arctan((self.N[self.El[i, 1], 1]-self.N[self.El[i, 0], 1])/(self.N[self.El[i, 1], 0]-self.N[self.El[i, 0], 0]))
            else:
                self.θ[i] = np.arctan((self.N[self.El[i, 1], 1]-self.N[self.El[i, 0], 1])/(self.N[self.El[i, 1], 0]-self.N[self.El[i, 0], 0]))+pi
            self.T[i, :, :] = np.array([[np.cos(self.θ[i]), -np.sin(self.θ[i]), 0,                 0,                  0, 0],
                                        [np.sin(self.θ[i]),  np.cos(self.θ[i]), 0,                 0,                  0, 0],
                                        [                0,                  0, 1,                 0,                  0, 0],
                                        [                0,                  0, 0, np.cos(self.θ[i]), -np.sin(self.θ[i]), 0],
                                        [                0,                  0, 0, np.sin(self.θ[i]),  np.cos(self.θ[i]), 0],
                                        [                0,                  0, 0,                 0,                  0, 1]])
            kG[i, :, :] = self.T[i]@kL[i]@self.T[i].T

            self.k[3*self.El[i, 0]:3*self.El[i, 0]+3, 3*self.El[i, 0]:3*self.El[i, 0]+3] += kG[i, 0:3, 0:3]
            self.k[3*self.El[i, 0]:3*self.El[i, 0]+3, 3*self.El[i, 1]:3*self.El[i, 1]+3] += kG[i, 0:3, 3:6]
            self.k[3*self.El[i, 1]:3*self.El[i, 1]+3, 3*self.El[i, 0]:3*self.El[i, 0]+3] += kG[i, 3:6, 0:3]
            self.k[3*self.El[i, 1]:3*self.El[i, 1]+3, 3*self.El[i, 1]:3*self.El[i, 1]+3] += kG[i, 3:6, 3:6]

        self.u[self.DoF] = np.linalg.solve(self.k[self.DoF,:][:,self.DoF], self.F[self.DoF])
        self.F[self.BC] = self.k[self.BC,:][:,self.DoF]@self.u[self.DoF]

    def ComputeStress(self):

        NL = np.zeros([self.nEl, 3, 6])
        self.r = np.zeros([self.nEl, 3, self.nStep+1])
        self.w = np.zeros([self.nEl, 3, self.nStep+1])
        BU = np.zeros([self.nEl, 6])
        BL = np.zeros([self.nEl, 6])
        v = np.zeros([self.nEl, 6])
        self.σU = np.zeros([self.nEl, self.nStep+1])
        self.σL = np.zeros([self.nEl, self.nStep+1])
        self.σMax = np.zeros([self.nEl, self.nStep+1])

        for i in range(self.nEl):
            v[i, :] = np.concatenate((self.u[3*self.El[i, 0]:3*self.El[i, 0]+3], self.u[3*self.El[i, 1]:3*self.El[i, 1]+3]), axis=0)[:,0]
            for j in range(self.nStep+1):
                ξ = j/(self.nStep)
                NL = np.array([[1-ξ,                     0,                      0, ξ,                     0,                      0],
                               [  0,       1-3*ξ**2+2*ξ**3, ξ*self.l[i,0]*(1-ξ)**2, 0,          ξ**2*(3-2*ξ), ξ**2*self.l[i,0]*(ξ-1)],
                               [  0, 6*ξ/self.l[i,0]*(ξ-1),           1-4*ξ+3*ξ**2, 0, 6*ξ/self.l[i,0]*(1-ξ),              ξ*(3*ξ-2)]])
                self.r[i, :, j] = np.concatenate((self.N[self.El[i, 0], :], self.θ[i]), axis=0)+self.T[i, 0:3, 0:3]@np.array([ξ*self.l[i], 0, 0])
                self.w[i, :, j] = self.T[i, 0:3, 0:3]@NL@self.T[i].T@v[i,:]
                # upper fibre
                BU[i, :] = np.array([-1/self.l[i, 0],
                                     1/self.l[i,0]**2*6*self.eU[i,0]*(1-2*ξ),
                                     1/self.l[i,0]*2*self.eU[i,0]*(2-3*ξ),
                                     1/self.l[i, 0],
                                     1/self.l[i,0]**2*6*self.eU[i,0]*(2*ξ-1),
                                     1/self.l[i,0]*2*self.eU[i,0]*(1-3*ξ)])
                self.σU[i,j] = self.E[i,0]*BU[i,:].T@(self.T[i].T@v[i,:])
                # lower fibre
                BL[i, :] = np.array([-1/self.l[i, 0],
                                     1/self.l[i,0]**2*6*self.eL[i,0]*(1-2*ξ),
                                     1/self.l[i,0]*2*self.eL[i,0]*(2-3*ξ),
                                     1/self.l[i, 0],
                                     1/self.l[i,0]**2*6*self.eL[i,0]*(2*ξ-1),
                                     1/self.l[i,0]*2*self.eL[i,0]*(1-3*ξ)])
                self.σL[i,j] = self.E[i,0]*BL[i,:].T@(self.T[i].T@v[i,:])
                self.σMax[i, j] = max(abs(self.σL[i, j]), abs(self.σU[i, j]))

        # deformation
        self.q = self.r+self.w*self.Scale

    def PlotStressUpperFibre(self):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        plt.title('Stress: upper fibre [MPa]')
        c = np.linspace(self.σU.min(), self.σU.max(), 5)  # np.linspace(σ.min(), σ.max(), 3)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        for i in range(self.nEl):
            plt.plot(self.r[i, 0, :], self.r[i, 1, :], c='gray', lw=3, ls='-', clip_on=False, marker='s')
            plt.plot(self.q[i, 0, :], self.q[i, 1, :], c='k', lw=1.5, ls='-', clip_on=False)
            for j in range(self.nStep+1):
                plt.plot(self.q[i, 0, j], self.q[i, 1, j], c=cmap.to_rgba(self.σU[i,j]), ls='', marker='o', clip_on=False)
        plt.colorbar(cmap, ticks=c)
    def PlotStressLowerFibre(self):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        plt.title('Stress: lower fibre [MPa]')
        c = np.linspace(self.σL.min(), self.σL.max(), 5)  # np.linspace(σ.min(), σ.max(), 3)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        for i in range(self.nEl):
            plt.plot(self.r[i, 0, :], self.r[i, 1, :], c='gray', lw=3, ls='-', clip_on=False, marker='s')
            plt.plot(self.q[i, 0, :], self.q[i, 1, :], c='k', lw=1.5, ls='-', clip_on=False)
            for j in range(self.nStep+1):
                plt.plot(self.q[i, 0, j], self.q[i, 1, j], c=cmap.to_rgba(self.σL[i,j]), ls='', marker='o', clip_on=False)
        plt.colorbar(cmap, ticks=c)
    def PlotStressMaximum(self):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        plt.title('Maximum stress [MPa]')
        c = np.linspace(self.σMax.min(), self.σMax.max(), 5)  # np.linspace(σ.min(), σ.max(), 3)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        for i in range(self.nEl):
            plt.plot(self.r[i, 0, :], self.r[i, 1, :], c='gray', lw=3, ls='-', clip_on=False, marker='s')
            plt.plot(self.q[i, 0, :], self.q[i, 1, :], c='k', lw=1.5, ls='-', clip_on=False)
            for j in range(self.nStep+1):
                plt.plot(self.q[i, 0, j], self.q[i, 1, j], c=cmap.to_rgba(self.σMax[i,j]), ls='', marker='o', clip_on=False)
        plt.colorbar(cmap, ticks=c)
    def PlotDisplacement(self):
        self.d = np.sqrt(self.w[:, 0, :]**2+self.w[:, 1, :]**2)
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        plt.title('Displacement [mm]')
        c = np.linspace(self.d.min(), self.d.max(), 5)  # np.linspace(σ.min(), σ.max(), 3)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        for i in range(self.nEl):
            plt.plot(self.r[i, 0, :], self.r[i, 1, :], c='gray', lw=3, ls='-', clip_on=False, marker='s')
            plt.plot(self.q[i, 0, :], self.q[i, 1, :], c='k', lw=1.5, ls='-', clip_on=False)
            for j in range(self.nStep+1):
                plt.plot(self.q[i, 0, j], self.q[i, 1, j], c=cmap.to_rgba(self.d[i,j]), ls='', marker='o', clip_on=False)
        plt.colorbar(cmap, ticks=c)
        plt.show()
        
if __name__ == '__main__':
    Cantilever = Beam2D()
    # Knoten
    Cantilever.N = np.array([[   0, 0],
                             [ 100, 0],
                             [ 100, 100]])
    # Elemente: welche Knoten werden verbunden?
    Cantilever.El = np.array([[0, 1],
                              [1, 2]])
    # Boundary conditions and loads
    Cantilever.BC = [0, 1, 2]
    Cantilever.Load = [[6,  100],
                       [7, -100]]
    Cantilever.Initialize()
    # Querschnitte
    b = 10      # mm
    h = 10      # mm
    Cantilever.eU = np.ones([Cantilever.nEl, 1])*h/2
    Cantilever.eL = np.ones([Cantilever.nEl, 1])*-h/2
    Cantilever.A = np.ones([Cantilever.nEl, 1])*b*h     # mm^2
    Cantilever.I = np.ones([Cantilever.nEl, 1])*b*h**3/12    # mm^4
    # Hier den E-Modul definieren!
    Cantilever.E = np.ones([Cantilever.nEl, 1])*210000        # MPa
    Cantilever.Solve()
    Cantilever.nStep = 8
    Cantilever.Scale = 10
    Cantilever.ComputeStress()
    Cantilever.PlotStressUpperFibre()
    Cantilever.PlotStressLowerFibre()
    Cantilever.PlotStressMaximum()
    Cantilever.PlotDisplacement()
