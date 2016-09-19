import sys, re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
            curr_x = [float(i) for i in values[:-1]]
            curr_x.append(1.0)
            x_list.append(curr_x)
            y_list.append(1.0) if values[-1].startswith("yes") else y_list.append(0.0)

        #print("x_list:", x_list, "\n\ny_list:", y_list, "\n\ntheta_list:", theta_list)


    logistic_regression(theta_list, x_list, y_list, 0.0005, 0.0000001)
    print(theta_list)
    print(J(theta_list, x_list, y_list))
    plot(theta_list, x_list, y_list)

def logistic_regression(theta_list, x_list, y_list, alpha, epsilon):
    J_prev = 0
    J_curr = J(theta_list, x_list, y_list)
    count = 0
    while abs(J_curr - J_prev) > epsilon:
        if count == 10000:
            print("too much iterations")
            break
        count += 1
        for j in range(len(theta_list)):
            for i in range(len(x_list)):
                diff = (h_theta(theta_list, x_list[i]) - y_list[i])
                theta_list[j] = theta_list[j] - alpha * diff * x_list[i][j]
        J_prev = J_curr
        J_curr = J(theta_list, x_list, y_list)


#--------------------------------
def J(theta_list, x_list, y_list):
    sigma = 0
    for i in range(len(x_list)):
        sigma += (h_theta(theta_list, x_list[i]) - y_list[i])**2
    return sigma / 2


#--------------------------------
def h_theta(theta, x):
    return 1.0/(1.0 + np.exp(-np.dot(theta, x)))


#--------------------------------
def predict(theta, x, y):
    return (h_theta(theta, x)**y) * ((1.0-h_theta(theta, x))**(1.0-y))


#--------------------------------
def plot(theta_list, x_list, y_list):
    new_x_list = [i[0] for i in x_list]
    new_y_list = [i[1] for i in x_list]
    hit, p1, p2, p3, p4 = 0, 0, 0, 0, 0
    for i in range(len(y_list)):

        if y_list[i] == 1.0:
            if predict(theta_list, x_list[i], y_list[i]) >= 0.5:
                p1, = plt.plot(np.dot(theta_list, x_list[i]), y_list[i], 'go')
                #plt.plot(new_x_list[i], new_y_list[i], 'go')
                hit += 1
            else:
                p2, = plt.plot(np.dot(theta_list, x_list[i]), y_list[i], 'gx')
                #plt.plot(new_x_list[i], new_y_list[i], 'gx')
        elif y_list[i] == 0.0 :
            if predict(theta_list, x_list[i], y_list[i]) >= 0.5:
                p3, = plt.plot(np.dot(theta_list, x_list[i]), y_list[i], 'ro')
                #plt.plot(new_x_list[i], new_y_list[i], 'ro')
                hit += 1
            else:
                p4, = plt.plot(np.dot(theta_list, x_list[i]), y_list[i], 'rx')
                #plt.plot(new_x_list[i], new_y_list[i], 'rx')
    #plt.plot([np.dot(theta_list, i) for i in x_list], [h_theta(theta_list, i) for i in x_list], "yo")
    plt.title("Regressão logística sobre os dados de 'students.csv'")
    plt.xlabel("z")
    plt.ylabel("g(z)")
    hit_true = 'P(y=admitido) = admitido'
    hit_false = 'P(y=admitido) = não admitido'
    miss_true = 'P(y=não admitido) = não admitido'
    miss_false ='P(y=não admitido) = admitido'

    plt.legend([p1,p2,p3,p4],[hit_true, hit_false, miss_true, miss_false])
    print("hit rate:", hit/len(y_list))
    plt.show()



if __name__=="__main__":
    main()
