import numpy
from sklearn.model_selection import train_test_split

def split_train_test(raw_data_file, train_file, test_file, train_size, test_size):

    data = numpy.genfromtxt(raw_data_file, delimiter=',')
    numpy.random.shuffle(data)

    train, test = train_test_split(data, train_size=train_size, test_size=test_size)

    numpy.savetxt(train_file, train, delimiter=",")
    numpy.savetxt(test_file, test, delimiter=",")


