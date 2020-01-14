Setup

Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    Open an anaconda prompt and run
        conda install theano pygpu
    then, from an anaconda prompt, the following python script should successfully execute.

        from __future__ import absolute_import, print_function, division
        import numpy
        import theano
        import theano.tensor as T
        from theano import function
        x = T.dscalar('x')
        y = T.dscalar('y')
        z = x + y
        f = function([x, y], z)
        print(f(2, 3)) # 5.0
        print(numpy.allclose(f(16.3, 12.1), 28.4)) # True

        a = theano.tensor.vector()  # declare variable
        b = theano.tensor.vector()  # declare variable
        out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
        f = theano.function([a, b], out)   # compile function
        print(f([1, 2], [4, 5]))  # prints [ 25.  49.]
