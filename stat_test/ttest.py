from math import sqrt

from scipy.stats import t
from numpy import mean

def degrees_of_freedom(s1, s2, n1, n2):
    """
    Compute the number of degrees of freedom using the Satterhwaite Formula
    
    @param s1 The unbiased sample variance of the first sample
    @param s2 The unbiased sample variance of the second sample
    @param n1 Thu number of observations in the first sample
    @param n2 The number of observations in the second sample
    """
    #degreeoffreedom= numberofobservation-1
    
    a=s1 /n1
    b=s2 /n2
    c=1/(n1-1)
    d=1/(n2-1)

    degreeoffreedom=(a+b)**2 /( c * a**2 + d * b**2)

    return degreeoffreedom 

def unbiased_sample_variance(observations, mean):
    """
    Compute the unbiased sample variance

    @param observations Iterable set of observations
    @param mean The estimated mean
    """
    n=1/ (len(observations) - 1)
    sample_=0
    for i in observations:
        sum0=(i-mean)**2
        sample_=sample_+sum0

    sample_var= n*sample_

    return sample_var

def t_statistic(mean1, mean2, n1, n2, svar1, svar2):
    """
    Compute the t-statistic for the given estimates
    """
    # Complete this funtion
    a=svar1
    b=svar2
    c=a/n1
    d=b/n2
    e= 1/sqrt(c+d)
    t_stat = e*(mean1-mean2) 
    return t_stat

def t_test(sample1, sample2):
    """
    Return the two-tailed p-value of a t test with unequal variance for two samples.

    @param sample1 An iterable of the first sample
    @param sample2 An iterable of the second sample
    """
    m1=mean(sample1)
    m2=mean(sample2)
    n1=len(sample1)
    n2=len(sample2)
    s1=unbiased_sample_variance(sample1,m1)
    s2=unbiased_sample_variance(sample2,m2)

    degree=degrees_of_freedom(s1,s2,n1,n2)
    tsta=t_statistic(m1,m2,n1,n2,s1,s2)
    p=2*(1.0-t.cdf(abs(tsta),degree))
    return p

if __name__ == "__main__":
    v1 = [5, 7, 5, 3, 5, 3, 3, 9]
    v2 = [8, 1, 4, 6, 6, 4, 1, 2]

    print("p-value is %f" % t_test(v1, v2))
    
