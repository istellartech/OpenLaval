# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 Interstellar Technologies Inc. All Rights Reserved.
Authors : Seiji Arther Murakami, Takahiro Inagawa

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from scipy import optimize
import configparser
import pandas as pd

plt.close('all')

class Blade():
    def __init__(self, setting_file, reload = False):
        if (reload):  # if reload, it doesn't read file
            pass
        else:
            self.setting_file = setting_file
            self.setting = configparser.ConfigParser()
            self.setting.optionxform = str
            self.setting.read(setting_file, encoding='utf8')
        setting = self.setting

        self.name = setting.get("Config", "Name")
        self.is_save_fig = setting.getboolean("Config", "SaveFig?")
        self.is_save_excel = setting.getboolean("Config", "SaveExcel?")
        self.num_output_points = setting.getint("Config", "number of output points")

        gamma = setting.getfloat("Turbine-Blade", "specific heat ratio")
        mach_in = setting.getfloat("Turbine-Blade", "inlet mach number")
        mach_out = setting.getfloat("Turbine-Blade", "outlet mach number")
        beta_in = setting.getint("Turbine-Blade", "inlet flow angle[deg]")
        vu = setting.getint("Turbine-Blade", "upper surface Prandtle-Meyer angle[deg]")
        vl = setting.getint("Turbine-Blade", "lower surface Prandtle-Meyer angle[deg]")

        # error mode
        assert mach_in > 1.0, "inlet mach number must be more than 1.0"

        self.gamma = gamma
        self.mach_in = mach_in
        self.mach_out = mach_out
        self.beta_in = beta_in
        self.vu = vu
        self.vl = vl

        self.Rstar_min = np.sqrt((gamma - 1)/(gamma + 1))
        self.const = self.chara_line(1)
        vi = int(round(self.get_Pr(mach_in)))
        vo = int(round(self.get_Pr(mach_out)))
        self.vi = vi
        self.vo = vo
        b1 = 1 + (gamma - 1) / 2 * (mach_out ** 2)
        b2 = 1 + (gamma - 1) / 2 * (mach_in ** 2)
        b3 = (gamma + 1) / (2 * (gamma - 1))
        beta_out_rad = - np.arccos(mach_in / mach_out * ((b1 / b2) ** b3) * np.cos(np.deg2rad(beta_in)))
        beta_out = np.rad2deg(beta_out_rad)
        self.beta_out = beta_out
        self.total_turn_ang = self.beta_in - self.beta_out
        self.alpha_lower_in = beta_in - (vi - vl)
        self.alpha_lower_out = beta_out + (vo - vl)
        self.alpha_upper_in = beta_in - (vu - vi)
        self.alpha_upper_out = beta_out + (vu - vo)
        self.shift = 0

        self.mach_upper = self.get_mach_from_prandtle_meyer(self.vu)
        self.mach_lower = self.get_mach_from_prandtle_meyer(self.vl)

        # Prandtle-Meyer angle limitation
        self.vlmin = 0
        self.vlmax = self.vi
        self.vumin = self.vi
        self.vumax = np.rad2deg((math.pi/2) * (np.sqrt((self.gamma + 1)/(self.gamma - 1)) - 1))

        self.Rstar_l = self.get_Ru(self.vl)
        self.Rstar_u = self.get_Ru(self.vu)

        print("Inlet Mach num = %.1f, Outlet Mach num = %.1f" % (mach_in, mach_out))
        print("Upeer Mach num = %.1f, Lower Mach num = %.1f" % (self.mach_upper, self.mach_lower))
        print("Inlet Prandtle-Meyer angle = %.1f [deg], Outlet Prandtle-Meyer angle = %.1f [deg]" % (self.vi, self.vo))
        print("Upper Prandtle-Meyer angle = %.1f [deg], Lower Prandtle-Meyer angle = %.1f [deg]" % (self.vu, self.vl))
        print("alpha lower in = %.1f [deg], alpha lower out = %.1f [deg]" % (self.alpha_lower_in, self.alpha_lower_out))
        print("alpha upper in = %.1f [deg], alpha upper out = %.1f [deg]" % (self.alpha_upper_in, self.alpha_upper_out))
        print("beta in = %.1f [deg], beta out = %.1f [deg]" % (self.beta_in, self.beta_out))
        print("total turn angle = %.1f [deg]" % (self.total_turn_ang))
        print("== Prandtle-Meyer angle limitation ==")
        print("upper max = %.2f, upper min = %.2f, lower max = %.2f, lower min = %.2f" % (self.vumax, self.vumin, self.vlmax, self.vlmin))
        print("R* lower = %.2f, R* upper = %.2f" % (self.Rstar_l, self.Rstar_u))

    # if Rstar value is less than self.Rstar_min, if will give math error
    def chara_line(self, Rstar):

        fai = 0.5 * (np.sqrt((self.gamma + 1)/(self.gamma - 1)) * np.arcsin((self.gamma - 1) / Rstar**2 - self.gamma) + np.arcsin((self.gamma + 1) * Rstar**2 - self.gamma))

        return fai

    def chara_x(self, Rstar, theta):
        return Rstar * np.sin(theta)

    def chara_y(self, Rstar, theta):
        return Rstar * np.cos(theta)

    # draws 2 characteristic lines. the angle starts from v1(compression wave) and v2(expansion wave).
    def draw_lines(self, v1, v2):

        v1 = np.deg2rad(v1)
        v2 = np.deg2rad(v2)

        counter = 1000
        i = np.arange(counter)
        Rstar = self.Rstar_min + i / counter
        Rstar = Rstar[Rstar < 1]
        theta1 = (self.chara_line(Rstar) - self.const) + v1
        theta2 = -(self.chara_line(Rstar) - self.const) + v2
        x0 = self.chara_x(Rstar, theta1)
        y0 = self.chara_y(Rstar, theta1)
        x1 = self.chara_x(Rstar, theta2)
        y1 = self.chara_y(Rstar, theta2)

        num = 360
        theta = np.arange(0, num)
        R_x = - np.sin(np.deg2rad(theta))
        R_y = np.cos(np.deg2rad(theta))
        R_x_min = - np.sin(np.deg2rad(theta)) * self.Rstar_min
        R_y_min = np.cos(np.deg2rad(theta)) * self.Rstar_min

        plt.plot(x0, y0)
        plt.plot(x1, y1)
        plt.plot(R_x, R_y)
        plt.plot(R_x_min, R_y_min)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_chara(self, angle, step,Rstar = 1):

        counter = 1000
        i = np.arange(counter)
        Rstar_tmp = self.Rstar_min + i / counter
        Rstar_tmp = Rstar_tmp[Rstar_tmp < 1]
        fai = self.chara_line(Rstar_tmp)
        for j in range(0, angle, step):
            x1 = self.chara_x(Rstar_tmp * Rstar, fai - self.const + np.deg2rad(j))
            y1 = self.chara_y(Rstar_tmp * Rstar, fai - self.const + np.deg2rad(j))
            x2 = self.chara_x(Rstar_tmp * Rstar, - (fai - self.const - np.deg2rad(j)))
            y2 = self.chara_y(Rstar_tmp * Rstar, - (fai - self.const - np.deg2rad(j)))

            plt.plot(x1, y1, "r")
            plt.plot(x2, y2, "k")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    # top arc angle is 0
    # v1 must be smaller than v2
    def get_R(self, v1, v2, R_star = 1):

        v1 = math.radians(v1)
        v2 = math.radians(v2)
        Rstar_tmp = 1	 # initial Rstar
        param = []
        temp_x = 1
        temp_y = 1
        myu_check = math.radians(90)

        # this for is very stupid. intersecting angle is always half the delta
        for num in range(1, 10):  # decimal precision of Rstar
            while(1):
                theta1 = (self.chara_line(Rstar_tmp) - self.const) + v1
                theta2 = -(self.chara_line(Rstar_tmp) - self.const) + v2
                # theta1 = (self.chara_line(Rstar_tmp) - self.chara_line(R_star)) + v1
                # theta2 = -(self.chara_line(Rstar_tmp) - self.chara_line(R_star)) + v2
                x0 = self.chara_x(Rstar_tmp * R_star, theta1)
                y0 = self.chara_y(Rstar_tmp * R_star, theta1)
                x1 = self.chara_x(Rstar_tmp * R_star, theta2)
                y1 = self.chara_y(Rstar_tmp * R_star, theta2)
                # print(Rstar, math.degrees(theta1), math.degrees(theta2), x0, x1)
                # print("R* = %.5f,\ttheta1 = %.5f,\ttheta2 = %.5f,\tx0 = %.5f,\tx1 = %.5f" % (Rstar_tmp, math.degrees(theta1), math.degrees(theta2), x0, x1))

                if (x0 == x1) and (y0 == y1):
                    myu_check = 0
                    break
                elif x1 - x0 < 0:
                    myu_check = np.arctan2(temp_y - y1, temp_x - x1)
                    # if myu_check == 0:
                    #     print(theta1, theta2)
                    temp_x = x1
                    temp_y = y1
                    Rstar_tmp += 1.0/(10**num)

                    break
                else:
                    Rstar_tmp -= 1.0/(10**num)

                    if Rstar_tmp < self.Rstar_min:
                        Rstar_tmp += 1.0/(10**num)
                        break

            if (x0 == x1) and (y0 == y1):
                break
        # print("Rstar_tmp = %f ,R_star = %f" % (Rstar_tmp,R_star))
        # print("prandtle default = %f, Prandtle vl = %f" % (self.chara_line(Rstar_tmp), self.chara_line(R_star)))
        # print("R* = %.5f,\ttheta1 = %.5f,\ttheta2 = %.5f,\tx0 = %.5f,\ty0 = %.5f" % (Rstar_tmp, math.degrees(theta1), math.degrees(theta2), x0, y0))
        param.append(Rstar_tmp)
        param.append(x0)
        param.append(y0)
        param.append(myu_check)
        return param

    def get_myu(self, M):
        return np.arcsin(1/M)

    def get_Mstar(self, Rstar):
        return 1/Rstar

    def get_mach(self, Mstar):
        return np.sqrt((2 * Mstar**2)/((self.gamma + 1) - (self.gamma - 1) * Mstar**2))

    def get_Ru(self, vu):
        return self.get_R(-vu, vu)[0]

    def rotate(self, x, y, angle):
        """ rotate the (x, y) coordinate by angle degrees """
        theta = np.deg2rad(angle)
        a = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        b = np.array((x, y))
        return np.dot(a, b)

    def get_Pr(self, Mach):

        Mstar = (((self.gamma + 1) / 2 * Mach**2) / (1 + (self.gamma - 1) / 2 * Mach**2))**0.5
        tmp1 = math.pi/4 * (math.sqrt((self.gamma + 1)/(self.gamma - 1)) - 1)
        tmp2 = self.chara_line(1/Mstar)

        Pr = math.degrees(tmp1 + tmp2)

        return Pr

    def get_mach_from_prandtle_meyer(self, v1):
        mach0 = 1.0

        def func(mach, v1):
            return self.get_Pr(mach) - v1

        sol = optimize.root(func, mach0, args=(v1))
        mach = sol.x[0]
        return mach

    def make_circular_arcs(self):
        """ make concentric circular arcs """
        alpha_l = np.arange(90 + self.alpha_lower_in, 90 + self.alpha_lower_out, - 0.1)
        self.lower_arc_x = self.Rstar_l * np.cos(np.deg2rad(alpha_l))
        self.lower_arc_y = self.Rstar_l * np.sin(np.deg2rad(alpha_l))
        alpha_u = np.arange(90 + self.alpha_upper_in, 90 + self.alpha_upper_out, - 0.1)
        self.upper_arc_x = self.Rstar_u * np.cos(np.deg2rad(alpha_u))
        self.upper_arc_y = self.Rstar_u * np.sin(np.deg2rad(alpha_u))

        self.lower_arc_x_shift = self.lower_arc_x
        self.lower_arc_y_shift = self.shift + self.lower_arc_y

    def transition_arc(self, Rstar0, vmin, vmax, angle, delta_v):
        """ lower transition arc function
        Args:
            Rstar0 (float) : initial R*
            vmin (float) : small v(Prandtle-Meyer angle) [deg]
            vmax (float) : large v(Prandtle-Meyer angle) [deg]
            angle (float) : rotate angle from Y* to y* [deg]
            delta_v (float) : interval Prandtle-Meyer angle  [deg]
        """
        kmin = 1
        kmax = int((vmax - vmin)/delta_v) + 1
        k = np.arange(kmin, kmax)

        xstar_l = np.zeros(k.size)
        ystar_l = np.zeros(k.size)
        xstar_l[-1] = 0
        ystar_l[-1] = Rstar0
        func_Rstar = 2 * np.radians(vmax) - np.pi / 2 * (np.sqrt((self.gamma + 1) / (self.gamma - 1)) - 1) - np.radians(2 * (k - 1) * delta_v)

        def func(Rstar, func_Rstar_i):
            return 2 * self.chara_line(Rstar) - func_Rstar_i

        Rstar = np.zeros(func_Rstar.size)
        for (i, f_Rstar) in enumerate(func_Rstar):
            sol = optimize.root(func, Rstar0, args=(f_Rstar))
            Rstar[i] = sol.x[0]

        fai_k = np.radians(vmax - vmin - (k - 1) * delta_v)
        xstar_k = - Rstar * np.sin(fai_k)
        ystar_k = Rstar * np.cos(fai_k)
        myu_k = - np.arcsin(np.sqrt(((self.gamma + 1) / 2) * Rstar**2 - (self.gamma - 1) / 2))

        m_k =[]
        for (i, j) in enumerate(k[:-1]):
            m_k += [np.tan((fai_k[i] + fai_k[j])/2 + (myu_k[i] + myu_k[j])/2)]
        mbar_k = np.tan(fai_k[1:])

        Xstar_l, Ystar_l = self.rotate(xstar_l[-1], ystar_l[-1], angle)
        x, y = [Xstar_l], [Ystar_l]
        for i in range(kmax - kmin - 2, 0, -1):
            a = ystar_l[i+1] - mbar_k[i] * xstar_l[i+1]
            b = ystar_k[i] - m_k[i] * xstar_k[i]
            c = m_k[i] - mbar_k[i]
            xstar_l[i] =  (a - b) / c
            ystar_l[i] = (m_k[i]*a -mbar_k[i]*b)/c
            if (angle > 0):
                xtemp = xstar_l[i]
            else:
                xtemp = - xstar_l[i]
            Xstar_l, Ystar_l = self.rotate(xtemp, ystar_l[i], angle)

            x += [(Xstar_l)]
            y += [(Ystar_l)]
        return x, y

    def make_lower_concave(self):
        """ make lower concave curve """
        delta_v = 0.5
        x, y = self.transition_arc(self.Rstar_l, self.vl, self.vi, self.alpha_lower_in, delta_v)

        self.lower_concave_in_x = x
        self.lower_concave_in_y = y
        self.lower_concave_in_x_end = x[-1]
        self.lower_concave_in_y_end = y[-1]
        self.lower_concave_in_x_shift = x
        self.lower_concave_in_y_shift = np.array(y) + self.shift

        x, y = self.transition_arc(self.Rstar_l, self.vl, self.vo, self.alpha_lower_out, delta_v)

        self.lower_concave_out_x = x
        self.lower_concave_out_y = y
        self.lower_concave_out_x_end = x[-1]
        self.lower_concave_out_y_end = y[-1]
        self.lower_concave_out_x_shift = x
        self.lower_concave_out_y_shift = np.array(y) + self.shift

    def make_upper_convex(self):
        """ make upper convex curve """
        xtmp = 0
        ytmp = self.Rstar_u
        x, y = [], []

        for num in range(0, int(round(self.beta_in - self.alpha_upper_in)*2) + 1):
            Xstar_b = xtmp
            Ystar_b = ytmp
            Rstar, Xstar_a, Ystar_a, myu_check = self.get_R(-self.vu, self.vu - num)
            myu = self.get_myu(self.get_mach(self.get_Mstar(Rstar)))
            a1 = math.tan(myu + math.radians(num/2.0))
            b1 = Ystar_a - a1 * Xstar_a
            a2 = math.tan(math.radians(num/2.0))
            b2 = Ystar_b - a2 * Xstar_b
            xtmp = ((b2 - b1) / (a1 - a2))
            ytmp = xtmp * a2 + b2
            rotx, roty = self.rotate(xtmp, ytmp, self.alpha_upper_in)

            x += [(rotx)]
            y += [(roty)]

            # print(num/2,xtmp,ytmp,Rstar)

        self.upper_convex_in_x = x
        self.upper_convex_in_y = y
        self.upper_convex_in_x_end = x[-1]
        self.upper_convex_in_y_end = y[-1]

        xtmp = 0
        ytmp = self.Rstar_u
        x, y = [], []
        for num in range(0, int(round((-(self.beta_out - self.alpha_upper_out))*2)) + 1):
            Xstar_b = xtmp
            Ystar_b = ytmp
            Rstar, Xstar_a, Ystar_a, myu_check = self.get_R(-self.vu, self.vu - num)
            myu = self.get_myu(self.get_mach(self.get_Mstar(Rstar)))
            a1 = math.tan(myu + math.radians(num/2.0))
            b1 = Ystar_a - a1 * Xstar_a
            a2 = math.tan(math.radians(num/2.0))
            b2 = Ystar_b - a2 * Xstar_b
            xtmp = ((b2 - b1) / (a1 - a2))
            ytmp = xtmp * a2 + b2
            rotx, roty = self.rotate(-xtmp, ytmp, self.alpha_upper_out)

            x += [(rotx)]
            y += [(roty)]

        self.upper_convex_out_x = x
        self.upper_convex_out_y = y
        self.upper_convex_out_x_end = x[-1]
        self.upper_convex_out_y_end = y[-1]

    def make_upper_straight_line(self):
        """ make upper straight line """
        targetx = self.lower_concave_in_x_end
        x = self.upper_convex_in_x_end
        y = self.upper_convex_in_y_end
        targety = np.tan(np.deg2rad(self.beta_in)) * targetx + y - np.tan(np.deg2rad(self.beta_in)) * x
        self.upper_straight_in_x = [targetx, x]
        self.upper_straight_in_y = [targety, y]
        self.shift = - abs(self.lower_concave_in_y_end - targety)

        targetx = self.lower_concave_out_x_end
        x = self.upper_convex_out_x_end
        y = self.upper_convex_out_y_end
        targety = np.tan(np.deg2rad(self.beta_out)) * targetx + y - np.tan(np.deg2rad(self.beta_out)) * x
        self.upper_straight_out_x = [targetx, x]
        self.upper_straight_out_y = [targety, y]

    def make_interpolate_curve(self):
        """ interpolate contour curves """
        lcx = np.zeros(0)
        lcy = np.zeros(0)
        lcx = np.append(lcx, np.array(self.lower_concave_in_x)[::-1])
        lcx = np.append(lcx, self.lower_arc_x)
        lcx = np.append(lcx, np.array(self.lower_concave_out_x))
        lcy = np.append(lcy, np.array(self.lower_concave_in_y)[::-1])
        lcy = np.append(lcy, self.lower_arc_y)
        lcy = np.append(lcy, np.array(self.lower_concave_out_y))
        self.lower_curve_x = lcx
        self.lower_curve_y = lcy
        self.lower_curve_x_shift = lcx
        self.lower_curve_y_shift = lcy + self.shift

        ucx = np.zeros(0)
        ucy = np.zeros(0)
        ucx = np.append(ucx, np.array(self.upper_straight_in_x))
        ucx = np.append(ucx, np.array(self.upper_convex_in_x)[::-1])
        ucx = np.append(ucx, self.upper_arc_x)
        ucx = np.append(ucx, np.array(self.upper_convex_out_x))
        ucx = np.append(ucx, np.array(self.upper_straight_out_x))
        ucy = np.append(ucy, np.array(self.upper_straight_in_y))
        ucy = np.append(ucy, np.array(self.upper_convex_in_y)[::-1])
        ucy = np.append(ucy, self.upper_arc_y)
        ucy = np.append(ucy, np.array(self.upper_convex_out_y))
        ucy = np.append(ucy, np.array(self.upper_straight_out_y))
        self.upper_curve_x = ucx
        self.upper_curve_y = ucy

        x = np.linspace(ucx.min(), ucx.max(), self.num_output_points)
        lcy_func = interp1d(self.lower_curve_x, self.lower_curve_y)
        lcy_shift_func = interp1d(self.lower_curve_x_shift, self.lower_curve_y_shift)
        ucy_func = interp1d(self.upper_curve_x, self.upper_curve_y)
        self.lower_curve_x_interp = x
        self.lower_curve_y_interp = lcy_func(x)
        self.lower_curve_x_shift_interp = x
        self.lower_curve_y_shift_interp = lcy_shift_func(x)
        self.upper_curve_x_interp = x
        self.upper_curve_y_interp = ucy_func(x)

        # make pandas DataFrame to save contour
        tmp = [x, self.lower_curve_y_shift_interp,
               self.upper_curve_y_interp, self.lower_curve_y_interp]
        self.dfc = pd.DataFrame(tmp, index = ["x", "lower curve1", "upper curve", "lower curve2"])
        if(self.is_save_excel):
            """ save contour in Excel file """
            writer = pd.ExcelWriter("result/turbine_contour_" + self.name + ".xlsx")
            self.dfc.T.to_excel(writer, "contour")
            writer.save()

    def calc(self):
        """ wrapper of making lines and curves """
        self.make_circular_arcs()
        self.make_lower_concave()
        self.make_upper_convex()
        self.make_upper_straight_line()
        self.make_circular_arcs()
        self.make_lower_concave()

        self.make_interpolate_curve()

    def plot_contour(self):
        """ Plot contour """
        plt.figure()
        plt.plot(self.lower_arc_x, self.lower_arc_y)
        plt.plot(self.upper_arc_x, self.upper_arc_y)
        plt.plot(self.lower_arc_x_shift, self.lower_arc_y_shift)
        plt.plot(self.lower_concave_in_x, self.lower_concave_in_y)
        plt.plot(self.lower_concave_in_x_shift, self.lower_concave_in_y_shift)
        plt.plot(self.lower_concave_out_x, self.lower_concave_out_y)
        plt.plot(self.lower_concave_out_x_shift, self.lower_concave_out_y_shift)
        plt.plot(self.upper_convex_in_x, self.upper_convex_in_y)
        plt.plot(self.upper_convex_out_x, self.upper_convex_out_y)
        plt.plot(self.upper_straight_in_x, self.upper_straight_in_y)
        plt.plot(self.upper_straight_out_x, self.upper_straight_out_y)
        # plt.plot([],[], color='k', label="hoge")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("turbine wing contour : " + self.name)
        # plt.legend(loc="best")
        # plt.grid()
        if(self.is_save_fig):plt.savefig("result/turbine_contour_" + self.name + ".png")
        plt.show()

    def plot_contour_simple(self, color="k"):
        """ Plot contour that all lines are mono color """
        plt.figure()
        plt.plot(self.lower_curve_x, self.lower_curve_y, color=color)
        plt.plot(self.lower_curve_x_shift, self.lower_curve_y_shift, color=color)
        plt.plot(self.upper_curve_x, self.upper_curve_y, color=color)
        # plt.plot([],[], color='k', label="hoge")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("turbine wing contour : " + self.name)
        # plt.legend(loc="best")
        # plt.grid()
        if(self.is_save_fig):plt.savefig("result/turbine_contour_simple" + self.name + ".png")
        plt.show()

    def change_setting_value(self, section, key, value):
        """ change value in setting file
        Args:
            section (str) : section of setting_file that is in []
            key (str) : key of setting_file
            value (str or float) : value to change
        """
        self.setting.set(section, key, str(value))
        self.__init__(self.setting_file, reload=True)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        setting_file = 'setting.ini'
    else:
        setting_file = sys.argv[1]
        assert os.path.exists(setting_file), "no file exist"
    plt.close("all")
    plt.ion()
    print("Design Supersonic Turbine")

    f = Blade(setting_file)
    f.calc()
    f.plot_contour()
    f.plot_contour_simple()
    plt.show()

    mach = np.linspace(1, 5)
    angle = np.zeros(mach.size)
    for (i, m) in enumerate(mach):
        angle[i] = f.get_Pr(m)
    plt.figure()
    plt.plot(mach, angle)
    plt.title("Prandtle-Meyer angle in specific heat ratio = %.2f" % (f.gamma))
    plt.xlabel("Mach number")
    plt.ylabel("Prandtle-Meyer angle [deg]")
    plt.grid()
    print("finish")
