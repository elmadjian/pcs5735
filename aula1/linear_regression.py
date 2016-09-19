import sys
import matplotlib.pyplot as plt
import numpy as np

theta_list = []
x_list = []
y_list = []

def main():
    if len(sys.argv) != 2:
        print("modo de usar: <este_programa> <arquivo_csv>")
        sys.exit()
    csv_file = sys.argv[1]

    with open(csv_file, "r") as arquivo:
        classes = arquivo.readline().split(",")
        theta_list = [0.0 for i in range(len(classes))]
        for line in arquivo:
            values = line.split(",")
            curr_x = [float(i) for i in values]
            curr_x[-1] = 1.0
            x_list.append(curr_x)
            y_list.append(float(values[-1]))

        #print("x_list:", x_list, "\n\ny_list:", y_list, "\n\ntheta_list:", theta_list)


    #batch_gradient_descent(theta_list, x_list, y_list, 0.000005, 0.00001)
    stochastic_gradient_descent(theta_list, x_list, y_list, 0.000005, 0.00001)
    #theta_list = normal_equations(x_list, y_list)
    print(theta_list)
    print(J(theta_list, x_list, y_list))
    plot(theta_list, x_list, y_list)


#--------------------------------
def J(theta_list, x_list, y_list):
    sigma = 0
    for i in range(len(x_list)):
        sigma += (h_theta(theta_list, x_list[i]) - y_list[i])**2
    return sigma / 2


#--------------------------------
def h_theta(theta_list, x_list_i):
    return np.dot(theta_list, x_list_i)


#--------------------------------
def batch_gradient_descent(theta_list, x_list, y_list, alpha, epsilon):
    J_prev = 0
    J_curr = J(theta_list, x_list, y_list)
    count = 0
    while (abs(J_curr - J_prev) > epsilon):
        count+=1
        if count > 10000:
            print("too much iterations")
            break
        for j in range(len(theta_list)):
            sigma = 0
            for i in range(len(x_list)):
                h = h_theta(theta_list, x_list[i])
                sigma += (h - y_list[i]) * x_list[i][j]
                #print("h>>", h)
            theta_list[j] = theta_list[j] - alpha * sigma
        J_prev = J_curr
        J_curr = J(theta_list, x_list, y_list)
    print("iterations:", count)

#--------------------------------
def stochastic_gradient_descent(theta_list, x_list, y_list, alpha, epsilon):
    J_prev = 0
    J_curr = J(theta_list, x_list, y_list)
    count = 0
    while (abs(J_curr - J_prev) > epsilon):
        count+=1
        if count > 10000:
            print("too much iterations")
            break
        for j in range(len(theta_list)):
            for i in range(len(x_list)):
                diff = (h_theta(theta_list, x_list[i]) - y_list[i])
                theta_list[j] = theta_list[j] - alpha * diff * x_list[i][j]
        J_prev = J_curr
        J_curr = J(theta_list, x_list, y_list)
    print("iterations:", count)

#--------------------------------
def normal_equations(x_list, y_list):
    X = np.array(x_list)
    y = np.array(y_list)
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

#--------------------------------
def plot(theta_list, x_list, y_list):
    new_x_list = [i[1] for i in x_list]
    plt.plot(new_x_list, y_list, 'ro')
    #x_list.sort()
    #for x in x_list:
    #    print("x:", x, "thetaTx:", np.dot(theta_list, x))
    plt.plot(new_x_list, [np.dot((theta_list[1], theta_list[3]), (i[1], i[3])) for i in x_list])
    plt.title("Regressão SGD sobre os dados em 'Iris Dataset'")
    plt.xlabel("Largura da sépala")
    plt.ylabel("Largura da pétala")
    plt.show()



if __name__=="__main__":
    main()
