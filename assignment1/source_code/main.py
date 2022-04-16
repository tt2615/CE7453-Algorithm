import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class Config():
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_name = 'lessthan4.txt'
        self.output_name = 'lessthan4_out.txt'
        self.output_fig_name = 'lessthan4.png'
        self.degree = 3
        self.delta = 1e-5


class InputParser():
    """Read in input files"""
    def __init__(self, path):
        self.path = path
        self.coors = None

    def parse(self):
        coors = []
        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                coor = line.split()
                coors.append(np.array([int(coor[0]), int(coor[1])]))
                line = f.readline()
        self.coors = np.array(coors)

class BSpline():
    """
        Store all nessensary information for a B-Spline
        Can write B-Spline data into file, or draw it
    """
    def __init__(self, degree, knots, ctl_points, data_points):
        self.degree = degree,
        self.u_list = knots,
        self.ctl_p_list = ctl_points
        self.data_p_list = data_points
    
    def write(self, output_path):
        with open(output_path, 'w') as f:
            f.write(f"{self.degree[0]}\n")
            f.write(f"{len(self.ctl_p_list)}\n\n")
            f.write(' '.join([str(x.round(3)) for x in list(self.u_list[0])]) + '\n\n')
            for p in self.ctl_p_list:
                f.write(f"{p[0].round(3)} {p[1].round(3)}\n")        

    def draw(self):
        ctr = np.array(self.ctl_p_list)
        c_x = ctr[:,0]
        c_y = ctr[:,1]

        l=len(c_x)
        t=np.linspace(0,1,l-2,endpoint=True)
        t=np.append([0,0,0],t)
        t=np.append(t,[1,1,1])

        tck=[t,[c_x,c_y],3]
        u3=np.linspace(0,1,(max(l*2,70)),endpoint=True)
        out = interpolate.splev(u3,tck) 

        p_x = self.data_p_list[:,0]
        p_y = self.data_p_list[:,1]

        plt.plot(c_x,c_y,'k--',label='Control polygon',marker='s',markerfacecolor='red')
        plt.plot(p_x, p_y, 'ro', label='Points')
        plt.plot(out[0],out[1],'b',linewidth=2.0,label='B-spline curve')
        plt.legend(loc='best')
        plt.axis([min(c_x)-1, max(c_x)+1, min(c_y)-1, max(c_y)+1])
        plt.title('Cubic B-spline Interpolation')
        plt.savefig(f"{config.dir_path}/{config.output_fig_name}")
        plt.show()

class InterpolationAlgo():
    """C2 cubic B-spline interpolation algo"""
    def __init__(self, input, degree=3):
        self.input = input
        self.degree = degree

    def _parameterize(self, coors):
        """chord length parameterization"""

        def __get_distance(coors):
            """calculate distance for each consecutive points"""
            dist_list = []

            prev_p = coors[0]
            for p in coors[1:]:
                dist = np.linalg.norm(p-prev_p)
                dist_list.append(dist)
                prev_p = p

            return np.array(dist_list)

        dist_list = __get_distance(coors)
        L = sum(dist_list)

        t_list = []
        t_list.append(0)
        cur_dist_total = 0
        for cur_dist in dist_list:
            cur_dist_total += cur_dist
            t = cur_dist_total/L
            t_list.append(t)
        return np.array(t_list)

    def _get_knots(self, t_list, degree):
        """
            add multiplicity at first and last point
            also add delta for multiplicit knot to prevent 0/0
        """
        list_head = np.repeat([t_list[0]], degree)
        list_tail = np.repeat([t_list[-1]], degree)

        joint_list = np.concatenate((list_head, t_list, list_tail))

        prev_knot = joint_list[0]
        delta_count = 1
        for i in range(1, len(joint_list)):
            if joint_list[i] == prev_knot:
                joint_list[i] += config.delta * delta_count
                delta_count += 1
            else:
                prev_knot = joint_list[i]
                delta_count = 1

        return joint_list

    def _get_ctl_points(self, knots, data, degree):
        
        def __compute_coefficient(u, t, i, k):
            """compute coefficient for control points, N_i^k(t), in a recursive manner"""
            try:
                if k <= 0:
                    return int(t >= u[i] and t < u[i+1])

                n = (t-u[i])/(u[i+k]-u[i]) * __compute_coefficient(u, t, i, k-1) + \
                    (u[i+k+1]-t)/(u[i+k+1]-u[i+1]) * __compute_coefficient(u, t, i+1, k-1)

                return n
            except IndexError:
                print("Index out of bound of knot list, pls check if i is correct")

        def __compute_cubic_2_deriv(u, t, i):
            """compute second derivative of the boundary conditions"""
            try:
                if t >= u[i] and t < u[i+1]:
                    deriv = (6 * (t - u[i])) / ((u[i+1] - u[i]) * (u[i+2] - u[i]) * (u[i+3] - u[i]))
                elif t >= u[i+1] and t < u[i+2]:
                    deriv = (-6*t + 4*u[i] + 2*u[i+2]) / ((u[i+2] - u[i+1]) * (u[i+3] - u[i]) * (u[i+2] - u[i])) \
                        + (2*u[i] + 2*u[i+1] + 2*u[i+3] - 6*t) / ((u[i+2] - u[i+1]) * (u[i+3] - u[i+1]) * (u[i+3] - u[i])) \
                        + (-6*t + 4*u[i+1] + 2*u[i+4]) / ((u[i+2] - u[i+1]) * (u[i+4] - u[i+1]) * (u[i+3] - u[i+1]))
                elif t >= u[i+2] and t < u[i+3]:
                    deriv = (6*t - 2*u[i] - 4*u[i+3]) / ((u[i+3] - u[i+2]) * (u[i+3] - u[i+1]) * (u[i+3] - u[i])) \
                        + (6*t - 2*u[i+1] - 2*u[i+3] - 2*u[i+4]) / ((u[i+3] - u[i+2]) * (u[i+4] - u[i+1]) * (u[i+3] - u[i+1])) \
                        + (6*t - 2*u[i+2] - 4*u[i+4]) / ((u[i+3] - u[i+2]) * (u[i+4] - u[i+2]) * (u[i+4] - u[i+1]))
                elif t >= u[i+3] and t < u[i+4]:
                    deriv = (6 * (u[i+4] - t)) / ((u[i+4] - u[i+3]) * (u[i+4] - u[i+2]) * (u[i+4] - u[i+1]))
                else:
                    deriv = 0
                return deriv
            except IndexError:
                print("Index out of bound of knot list, pls check if i is correct")

        ##generate n+1 equations for each data point
        #coefficient matrix
        data_num, point_num = len(data), len(data)+2
        N = np.zeros((data_num, point_num))
        for i in range(data_num): #each row corresponds to a data point
            for j in range(point_num): #each column corresponds to a control point i's coefficient N_i^k(t_i)
                N[i][j] = __compute_coefficient(u = knots, t = knots[i+degree], i = j, k = degree)
        #dependent matrix
        D_x = data.transpose()[0]
        D_y = data.transpose()[1]

        ##add boundary conditions
        cond1, cond2 = np.zeros(point_num), np.zeros(point_num)
        for i in range(degree):
            cond1[i] = __compute_cubic_2_deriv(u = knots, t = knots[degree], i = i)
            cond2[data_num-1+i] = __compute_cubic_2_deriv(u = knots, t = knots[data_num-1+degree], i = data_num-1+i)
        #insert to coefficient matrix
        N = np.insert(N, 1, cond1, axis=0)
        N = np.insert(N, -1, cond2, axis=0)
        N = N.round(3)
        #dependent matrix
        D_x = np.insert(D_x, 1, 0)
        D_x = np.insert(D_x, -1, 0)
        D_y = np.insert(D_y, 1, 0)
        D_y = np.insert(D_y, -1, 0)

        ##solve for control points
        x_list = np.linalg.solve(N, D_x)
        x_list = x_list.round(3)
        y_list = np.linalg.solve(N, D_y)
        y_list = y_list.round(3)
        points = [coor for coor in zip(x_list, y_list)]

        return points

    def run(self):
        """generate a b-spline object which interpolates input points"""

        #set knot vector
        t_list = self._parameterize(self.input.coors)
        knots = self._get_knots(t_list, self.degree)

        #solve control points
        ctl_points = self._get_ctl_points(knots, data = self.input.coors, degree=self.degree)

        #instantiate b_spline
        b_spline = BSpline(self.degree, knots, ctl_points, self.input.coors)
        return b_spline


if __name__ == '__main__':

    config = Config()

    #parse input files
    file_path = f"{config.dir_path}/{config.input_name}"
    input = InputParser(file_path)
    input.parse()

    #run interpolation algo
    algo = InterpolationAlgo(input, degree=config.degree)
    b_spline = algo.run()

    b_spline.write(f"{config.dir_path}/{config.output_name}")
    b_spline.draw()