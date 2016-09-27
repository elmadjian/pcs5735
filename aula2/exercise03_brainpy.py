import sys, cv2, os
from pybrain.structure import FeedForwardNetwork, SigmoidLayer,\
    FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

#------------------------
def main():
    DS = open_faces_dataset()
    fnn = create_FNN(DS.indim, 32, DS.outdim)

    trainer = BackpropTrainer(fnn, dataset=DS, momentum=0.1,
                               verbose=True, weightdecay=0.01)
    #for i in range(10):
    trainer.trainEpochs(20)
    trnresult = percentError(trainer.testOnClassData(),
                          DS['class'])
    # tstresult = percentError(trainer.testOnClassData(
    #     dataset=tstdata), tstdata['class'])

    print ("epoch: %4d" % trainer.totalepochs,
           "  train error: %5.2f%%" % trnresult)
        #    "  test error: %5.2f%%" % tstresult)




#------------------------
def create_FNN(n_in, n_hidden, n_out):
    #creating neurons
    n = FeedForwardNetwork()
    in_layer = SigmoidLayer(n_in)
    hidden_layer = SigmoidLayer(n_hidden)
    out_layer = SigmoidLayer(n_out)

    #adding layers
    n.addInputModule(in_layer)
    n.addModule(hidden_layer)
    n.addOutputModule(out_layer)

    #adding connections
    in_to_hidden  = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)
    n.sortModules()

    return n


#------------------------
def open_faces_dataset():
    """
    left     --> 0
    right    --> 1
    up:      --> 2
    straight --> 3
    """
    DS = ClassificationDataSet(960, class_labels=['left','right','up','straight'])
    for root, dirs, files in os.walk(os.getcwd() + '/faces_4'):
        for name in files:
            img = cv2.imread(os.path.join(root, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = img.flatten().tolist()
            if "_left_" in name:
                DS.appendLinked(features, [0])
            elif "_right_" in name:
                DS.appendLinked(features, [1])
            elif "_up_" in name:
                DS.appendLinked(features, [2])
            elif "_straight_" in name:
                DS.appendLinked(features, [3])
    return DS



#-----------------------
if __name__=="__main__":
    main()
