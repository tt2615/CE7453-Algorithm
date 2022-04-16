from dis import dis
from re import M
from turtle import distance
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import ParseSyntaxException
from scipy import optimize, stats

from interpolation import Config, InputParser, BSpline, InterpolationAlgo
from t_interpolation import Trigono_Interpolation

def plot_graph(h):

    u = np.linspace(0, 1, 100)

    x = 1.5 * (np.exp(1.5 * np.sin(6.2*u - 0.027*h)) + 0.1) * np.cos(12.2*u)
    y = (np.exp(np.sin(6.2*u - 0.027*h)) + 0.1) * np.sin(12.2*u)
    print(x)

    fig  = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    plt.title('Original Parametric Curve')

    plt.plot(x, y, 'b')
    plt.savefig('./original.png')
    plt.show()
#end plot_graph


def least_square_fit(h, sampling_method = 'uniform'):

    def __get_distance(coors):
            """calculate distance for each consecutive points"""
            # print(coors)
            dist_list = [0]

            prev_p, accum_dist = coors[0], 0
            for p in coors[1:]:
                dist = np.linalg.norm(p-prev_p)
                accum_dist += dist
                dist_list.append(accum_dist)
                prev_p = p

            return np.array(dist_list)

    # sampling of points
    if sampling_method == 'random':
        u_s = np.random.rand(100)
        x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.cos(12.2*u_s)
        y_s = (np.exp(np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.sin(12.2*u_s)
    elif sampling_method == 'uniform':
        u_s = np.linspace(0, 1, 100)
        x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.cos(12.2*u_s)
        y_s = (np.exp(np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.sin(12.2*u_s)
    elif sampling_method == 'normal':
        lower, upper = 0, 1
        mu, sigma = 0.5,1
        u_s = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=100)
        x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.cos(12.2*u_s)
        y_s = (np.exp(np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.sin(12.2*u_s)
    elif sampling_method == 'equal':
        u_10s = np.linspace(0, 1, 1000)
        x_10s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_10s - 0.027*h)) + 0.1) * np.cos(12.2*u_10s)
        y_10s = (np.exp(np.sin(6.2*u_10s - 0.027*h)) + 0.1) * np.sin(12.2*u_10s)

        distances = __get_distance(np.vstack([x_10s, y_10s]).T)
        # print(distances)
        avg_dist = distances[-1]/100
        # print(avg_dist)

        u_s, x_s, y_s = [u_10s[0]], [x_10s[0]], [y_10s[0]]
        prev_dist = 0
        for i, dist in enumerate(distances):
            if dist - prev_dist > avg_dist:
                u_s.append(u_10s[i])
                x_s.append(x_10s[i])
                y_s.append(y_10s[i])
                prev_dist = dist
        u_s = np.array(u_s)
        x_s = np.array(x_s)
        y_s = np.array(y_s)
    
    A = np.vstack([np.ones(len(u_s)), u_s, u_s**2, u_s**3]).T
    c_x = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), x_s) #c_x = ((A^T A)^-1) A^T x
    c_y = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y_s)

    u = np.linspace(0, 1, 100)
    x = 1.5 * (np.exp(1.5 * np.sin(6.2*u - 0.027*h)) + 0.1) * np.cos(12.2*u)
    y = (np.exp(np.sin(6.2*u - 0.027*h)) + 0.1) * np.sin(12.2*u)
    new_x = c_x[0]+ c_x[1]*u + c_x[2]*u**2 + c_x[3]*u**3
    new_y = c_y[0]+ c_y[1]*u + c_y[2]*u**2 + c_y[3]*u**3
    # print(c_x, c_y)

    #plot graph
    fig  = plt.figure()
    plt.title('Cubic Least Square Fitting')
    # ax = fig.add_subplot(1,1,1)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')

    plt.plot(x_s, y_s, 'o', color='grey', label='Points')
    plt.plot(x, y, '--', color='black' ,label='Original Curve')
    plt.plot(new_x, new_y, color='red',label='Fitted Curve')
    plt.legend(loc='best')
    plt.savefig('./ls_fit.png')
    plt.show()
#end least_square_fit


def b_interpolate(h, sampling_method = 'uniform'):
    config = Config()
    config.dir_path = '.'
    config.output_fig_name = "interpolation.png"
    config.input_name = 'interpolation.txt'
    input_file_path = f"{config.dir_path}/{config.input_name}"
    fig_path = f"{config.dir_path}/{config.output_fig_name}"

    def __get_distance(coors):
            """calculate distance for each consecutive points"""
            print(coors)
            dist_list = [0]

            prev_p, accum_dist = coors[0], 0
            for p in coors[1:]:
                dist = np.linalg.norm(p-prev_p)
                accum_dist += dist
                dist_list.append(accum_dist)
                prev_p = p

            return np.array(dist_list)

    #prepare input
    ##sampling of points
    if sampling_method == 'random':
        u_s = np.random.rand(10)
        u_s = np.sort(u_s)
        x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.cos(12.2*u_s)
        y_s = (np.exp(np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.sin(12.2*u_s)
    elif sampling_method == 'uniform':
        u_s = np.linspace(0, 1, 10)
        x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.cos(12.2*u_s)
        y_s = (np.exp(np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.sin(12.2*u_s)
    elif sampling_method == 'normal':
        lower, upper = 0, 1
        mu, sigma = 0.5,1
        u_s = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=10)
        u_s = np.sort(u_s)
        x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.cos(12.2*u_s)
        y_s = (np.exp(np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.sin(12.2*u_s)
    elif sampling_method == 'equal':
        u_10s = np.linspace(0, 1, 100)
        x_10s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_10s - 0.027*h)) + 0.1) * np.cos(12.2*u_10s)
        y_10s = (np.exp(np.sin(6.2*u_10s - 0.027*h)) + 0.1) * np.sin(12.2*u_10s)
        distances = __get_distance(np.vstack([x_10s, y_10s]).T)
        # print(distances)
        avg_dist = distances[-1]/10
        print(avg_dist)

        u_s, x_s, y_s = [u_10s[0]], [x_10s[0]], [y_10s[0]]
        prev_dist = 0
        for i, dist in enumerate(distances):
            if dist - prev_dist > avg_dist:
                u_s.append(u_10s[i])
                x_s.append(x_10s[i])
                y_s.append(y_10s[i])
                prev_dist = dist
        u_s = np.array(u_s)
        x_s = np.array(x_s)
        y_s = np.array(y_s)

    u = np.linspace(0, 1, 100)
    x = 1.5 * (np.exp(1.5 * np.sin(6.2*u - 0.027*h)) + 0.1) * np.cos(12.2*u)
    y = (np.exp(np.sin(6.2*u - 0.027*h)) + 0.1) * np.sin(12.2*u)

    ##write points to file
    with open(input_file_path, 'w') as f:
        for i in range(len(x_s)):
            f.write(f"{x_s[i]} {y_s[i]}\n")


    #parse input files
    input = InputParser(input_file_path)
    input.parse()

    #run interpolation algo
    algo = InterpolationAlgo(input, degree=config.degree)
    b_spline = algo.run()

    b_spline.draw(fig_path, x, y)



def t_interpolate(h):
    algo = Trigono_Interpolation(h)
    algo.run()
    algo.draw("./t_interpolate.png")


def integrate(h):
    m = 100
    u_s = np.linspace(0, 1, 2*m+1)
    print(u_s)
    height = 1 / 200
    x_s = 1.5 * (np.exp(1.5 * np.sin(6.2*u_s - 0.027*h)) + 0.1) * np.cos(12.2*u_s)

    
    auc = 0
    for i in range(m):
        auc += height * (x_s[2*i]/3 + x_s[2*i+1]/3*4 + x_s[2*i+2]/3)

    print("Integral reuslt: ", auc)


if __name__=='__main__':

    h = 45

    #1. plot function
    # plot_graph(h)

    #2. least square fit
    # least_square_fit(h, sampling_method='equal')

    #3. b-spline interpolation
    b_interpolate(h, sampling_method='uniform')

    #4. trigonometric interpolation
    # t_interpolate(h)

    #5. simpson integral
    # integrate(h)
    