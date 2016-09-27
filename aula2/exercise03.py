#important: python2!
import cv2, os
from ffnet import ffnet, mlgraph

#------------------------
def main():
    #creating network
    conec = mlgraph((960, 128, 4))
    net   = ffnet(conec)

    #loadgin dataset
    dataset, target = load_dataset('/faces_4')

    #training the network
    net.train_tnc(dataset, target, nproc='ncpu', maxfun=1000, messages=1)
    #net.train_bfgs(dataset, target, maxfun=1500)

    # #testing the network
    testset, test_target = load_dataset('/faces_test')
    output, regression = net.test(testset, test_target, iprint = 2)


#------------------------
def load_dataset(folder_path):
    """
    left     --> 0
    right    --> 1
    up:      --> 2
    straight --> 3
    """
    dataset, target = [], []
    for root, dirs, files in os.walk(os.getcwd() + folder_path):
        for name in files:
            img = cv2.imread(os.path.join(root, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = img.flatten()
            dataset.append(features)
            if "_left_" in name:
                target.append([1,0,0,0])
            elif "_right_" in name:
                target.append([0,1,0,0])
            elif "_up_" in name:
                target.append([0,0,1,0])
            elif "_straight_" in name:
                target.append([0,0,0,1])
    return dataset, target




#-----------------------
if __name__=="__main__":
    main()
