from random import gammavariate
from random import random
import numpy as np

"""
Samples from a Dirichlet distribution with parameter @alpha using a Gamma distribution
Reference: 
http://en.wikipedia.org/wiki/Dirichlet_distribution
http://stackoverflow.com/questions/3028571/non-uniform-distributed-random-array
"""

def Dirichlet(alpha):
    sample = [gammavariate(a, 1) for a in alpha]
    sample = [v / sum(sample) for v in sample]
    return sample

"""
Normalize a vector to be a probablistic representation
"""

def normalize(vec):
    vec_sum = sum(vec)
    assert (abs(vec_sum) != 0.0)  # the sum must not be 0
    """
    if abs(s) < 1e-6:
        print "Sum of vectors sums almost to 0. Stop here."
        print "Vec: " + str(vec) + " Sum: " + str(s)
        assert(0) # assertion fails
    """

    for i in range(len(vec)):
        assert (vec[i] >= 0)  # element must be >= 0
        vec[i] /= vec_sum

def choose(vec, pr):
    assert (len(vec) == len(pr))
    # normalize the distributions
    normalize(pr)
    r = random()
    index = 0
    while (r > 0):
        r = r - pr[index]
        index = index + 1
    return vec[index - 1]


if __name__ == "__main__":
    # This is a test
    print(Dirichlet([1, 1, 1]))
