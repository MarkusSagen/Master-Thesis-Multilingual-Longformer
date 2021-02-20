from sacred import Experiment
from tracking import sacred_observer

ex = Experiment('y=kx+m')
ex.observers.append(sacred_observer())



@ex.config
def my_config():
    """
    Defines semi 'GLobal' variables that can be changed before running via the command line argument
    """
    k = .9
    m = 1

@ex.capture
def some_func(k, m, x, y):
    print(k, m, x, y)

@ex.automain
def my_main(k, m):
    x = 18
    y = k*x+m
    some_func(k, m, x, y)
    k = k+2
    return y


if __name__ == '__main__':
    # my_main()
    r = ex.run()
    print(r.config)
