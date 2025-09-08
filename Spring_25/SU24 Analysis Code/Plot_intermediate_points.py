import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev, CubicSpline
from scipy import interpolate
from scipy.optimize import curve_fit
import ShapeAnalysisVerification as sav
import math


intermediate_file = "D:\\Gordon\\Automate FEB Runs\\2024_10_28\\2024_10_29_intermediate.csv"
header_pairs = [('inner_y', 'inner_z'), ('outer_y', 'outer_z'), ('innerShape_x', 'innerShape_y'), ('outerShape_x', 'outerShape_y')]

def plotIntermediatePoints(numrows, file):
        """ TODO: This function will use the intermediate file and the number of rows to plot the
        # the points that are in that row. The idea is that it will go through each header and collect
        # the x,y, or z and then use the points to plot what those points are representing. This should
        include some type of way to read the header files and the row that it is currently on and then
        take those numbers, put it in an array, and then plot using each element in that array. """
        df = pd.read_csv(file)
        df = df.head(numrows)

        for pair in header_pairs:
                x_header, y_header = pair
                x_coords = []
                y_coords = []

                print(f"\nProcessing pair: {x_header} and {y_header}")

                for col in df.columns:
                        if col.startswith(x_header):
                                x_coords.append(df[col].values)
                                #print(f"Found x column: {col}")
                        elif col.startswith(y_header):
                                y_coords.append(df[col].values)
                                #print(f"Found y column: {col}")


                x_cords_flat = [cord for sublist in x_coords for cord in sublist]
                y_cords_flat = [cord for sublist in y_coords for cord in sublist]
                print("X_VALS: ", x_cords_flat)
                print("Y_VALS: ", y_cords_flat)

                if len(x_cords_flat) != len(y_cords_flat):
                        raise ValueError("Mismatch in number of x and y coordinates.")

                coordinates = list(zip(x_cords_flat,y_cords_flat))

                #print(coordinates)

                x_vals, y_vals = zip(*coordinates)

                plt.figure()


                plt.scatter(x_vals, y_vals, label=f'{x_header} vs {y_header}', color='blue', marker='o')

                plt.xlabel(f'{x_header}')
                plt.ylabel(f'{y_header}')
                plt.title(f'Plot ({x_header}, {y_header}) Points First {numrows} Rows')
                plt.legend()


                plt.show()

def find_circle_center(points):
        """Finds the approximate center of a circle given a set of points.

        Args:

            points: A list of 2D points.

        Returns:
            A tuple (x, y) representing the center of the circle.
        """

        x = np.array([p[0] for p in points])

        y = np.array([p[1] for p in points])

        x_m, y_m = np.mean(x), np.mean(y)

        return x_m, y_m


def create_spline(points, center, ouput_points=10):
        """Creates a spline that passes through the given points.
        Args:
            points: A list of 2D points.
            center: A tuple (x, y) representing the center of the circle.

        Returns:
            A NumPy array of points representing the spline.
        """
        points = np.array(points)

        # Sort points by angle relative to the center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

        # Sort the points to match them to the angle
        sorted_points = np.argsort(angles)
        sorted_angles = angles[sorted_points]
        sorted_xs = points[sorted_points, 0]
        sorted_ys = points[sorted_points, 1]

        # adds the starting point to the end to close the circle
        sorted_angles = np.append(sorted_angles, sorted_angles[0] + 2 * np.pi)
        sorted_xs = np.append(sorted_xs, sorted_xs[0])
        sorted_ys = np.append(sorted_ys, sorted_ys[0])
        # this is what we used before, keeping here for reference but if not needed can delete
        '''curve_x = interpolate.UnivariateSpline(sorted_angles, sorted_xs, k=5)
        curve_y = interpolate.UnivariateSpline(sorted_angles, sorted_ys, k=5)'''

        # creates the splines for the xs and ys
        curve_x = CubicSpline(sorted_angles, sorted_xs, bc_type = 'periodic')
        curve_y = CubicSpline(sorted_angles, sorted_ys, bc_type = 'periodic')

        # defines the evenly spaced angles to for the spline points
        spaced_angles = np.linspace(0,2 * math.pi, ouput_points)
        spaced_angles = spaced_angles[:-1]
        

        #generates the x and y values using the splines
        xnew = curve_x(spaced_angles)
        ynew = curve_y(spaced_angles)

        #checking the angles and graphing the points for verification
        # spaced_angles_deg = spaced_angles*180/math.pi
        # print("***Angles: ", spaced_angles_deg)
        # plt.plot(xnew,ynew)
        # plt.show()

        # returns the two arrays as a 2D array
        return np.column_stack((xnew, ynew))


def angle_spline_driver(inner_radius, outer_radius, output_points = 10):
        """This is the "main" function that gets us the spline as of 11/21
                Args:
                    inner_radius: a dictionary of the inner radius points
                    outer_radius: a dictionary of the outer radius points

                Returns:
                    the splines for both the inner and outer radius in np 2d array
                """
        #converts the dictionary of the edge points to a 2d coordinate array
        inner_radius = sav.get_2d_coords_from_dictionary(inner_radius)
        # print("inner_radius", inner_radius)
        outer_radius = sav.get_2d_coords_from_dictionary(outer_radius)

        # print("here are the coords that are going to be going into function: ")
        # print("inner radius: ", [inner_list[0] for inner_list in inner_radius])
        # print("inner radius coords for spline: ", inner_radius)
        # print("type: ", type(inner_radius))
        # print("inner radius: ", [inner_list[1] for inner_list in inner_radius])
        
        # print("outer radius: ", [inner_list[0] for inner_list in outer_radius])
        # print("outer radius: ", [inner_list[1] for inner_list in outer_radius])
        
        # print("inner type:", type(inner_radius))
        # print("outer radius: ", outer_radius)

        # calculates the center of points
        center_inner = find_circle_center(inner_radius)
        # print("center_inner:", center_inner)
        center_outer = find_circle_center(outer_radius)

        #Gets the spline of those points
        spline_points_inner = create_spline(inner_radius, center_inner, output_points)
        spline_points_outer = create_spline(outer_radius, center_outer, output_points)


        #converts to a np array if not already
        #TODO: find out what to do to be able to plot the points
        outer_radius = np.array(outer_radius)
        inner_radius = np.array(inner_radius)
        #plots the spline and any points that are needed on the graph.
        #plot_spline(center_inner, spline_points_inner, inner_radius)
        #plot_spline(center_outer, spline_points_outer, outer_radius)

        return spline_points_inner, spline_points_outer

def plot_spline(center, spline_points, radius):
        plt.scatter(radius[:, 0], radius[:, 1])

        plt.scatter(center[0], center[1], color='red')

        plt.plot(spline_points[:, 0], spline_points[:, 1], color='green')

        #plt.scatter(equally_spaced_points[:, 0], equally_spaced_points[:, 1], color='blue')

        plt.show()
