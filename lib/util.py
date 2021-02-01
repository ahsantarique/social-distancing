'''Utilities'''
import matplotlib.pyplot as plt

def plot(x, y, label, title, xlabel, ylabel, showplot = True):
    # plot
    lines = plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if(showplot):
      plt.show()
    return lines

def plot_hist(x, bins, label, title, xlabel, ylabel, showplot = False):
    # plot
    plt.hist(x = x, bins = bins, density = True, histtype='bar', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if(showplot):
      plt.show()

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    return list(set(a) & set(b))

def union(a, b):
    return list(set(a) | set(b))