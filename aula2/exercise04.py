#important: python2!
import sys
from ffnet import ffnet, mlgraph

def main():
    if len(sys.argv) != 2:
        print "usage: <this_program> <dataset_file>"
        sys.exit()

    #creating network
    conec = mlgraph((24, 12, 12, 2))
    net   = ffnet(conec)

    #loading the dataset
    dataset, target = load_dataset(sys.argv[1])

    #training the network
    net.train_bfgs(dataset[:-25], target[:-25], maxfun=5000)
    #net.train_momentum(dataset[:-25], target[:-25], maxiter=5000, disp=True)

    # # #testing the network
    testset, test_target = dataset[-25:], target[-25:]
    output, regression = net.test(testset, test_target, iprint = 2)


#-----------------------
def load_dataset(filename):
    dataset, target = [], []
    with open(filename, 'r') as f:
        for line in f:
            values  = line.split()
            numeric = [float(v) for v in values]
            dataset.append(numeric[:-1])
            target.append([1,0]) if numeric[-1] == 1.0 else target.append([0,1])
    return dataset, target


#-----------------------
if __name__=="__main__":
    main()
