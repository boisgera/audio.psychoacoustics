#!/usr/bin/env python
# coding: utf-8

__author__ = u"Sébastien Boisgérault <Sebastien.Boisgerault@mines-paristech.fr>"
__version__ = "trunk"

# Python 2.x Standard Library
# (no dependency)

# Third-Party Librairies
from numpy import *
from scipy.optimize import fsolve
from audio.fourtier import *

#
# Metadata
# ------------------------------------------------------------------------------
#
__main__ = (__name__ == "__main__")

from audio.about_psychoacoustics import *


#-------------------------------------------------------------------------------
# Bark Scale
#-------------------------------------------------------------------------------
def bark(f):
    "Convert a frequency from Hertz to Bark"
    f = array(f, copy=False)
    return 13.0 * arctan(0.76 * f / 1000.0) + 3.5 * arctan((f / 7500.0) ** 2)

def hertz(b):
    "Convert a frequency from Bark to Hertz"
    b0 = 10.0
    b = array(b, copy=False)
    f = zeros(shape(b))
    for i in ndindex(shape(b)):
        f[i] = fsolve(lambda f_: bark(f_) - b[i], b0)
    return f

def critical_bandwidth(f, solve=False):
    "Critical Bandwidth (input and output in Hz)"
    f = array(f, copy=False)
    if solve:
        df0 = 100.0
        df = zeros(shape(f))
        for i in ndindex(shape(f)):
            f_ = f[i]
            def width_in_bark_minus_one(df_):
                return bark(f_ + 0.5 * df_) - bark(f_ - 0.5 * df_) - 1.0
            df[i] = fsolve(width_in_bark_minus_one, df0)
    else:
        f = f / 1000.0
        df = 25.0 + 75.0 * (1.0 + 1.4 * f**2) ** 0.69
    return df

#-------------------------------------------------------------------------------
# Masks
#-------------------------------------------------------------------------------
class Mask(object):
    """
    Mask Functions Compositor

    Masks are frequency [Hz] to SPL [dB] functions whose sum computed as sums 
    of intensity levels.
    """
    def __init__(self, *masks):
        self._masks = []
        for mask in masks:
            if isinstance(mask, Mask):
                self._masks.extend(mask._masks)
            else:
                if callable(mask):
                    self._masks.append(mask)
                else:
                    error = "argument {0:r} is not a function"
                    raise TypeError(error.format(mask))

    def __call__(self, f):
        f = array(f, copy=False)
        I = zeros(shape(f))
        for mask in self._masks:
            I += 10 ** (mask(f) / 10.0)
        return 10 * log10(I) 

    def __add__(self, other):
        if isinstance(other, Mask):
            return Mask(*(self._masks + other._masks))
        elif callable(other):
            return self + Mask(other)
        else:
            return NotImplemented
    
    # Rk: MUCH MUCH less spread artifacts with hanning windows. 
    # Plus as it artificially decreases the energy of the masker, it
    # lowers the mask which compensation for the absence of gain ratio
    # In Fletcher's model.
    @staticmethod # frame normalized to [-1,+1] assumed
    def from_frame(frame, dt=1.0, window=hanning, floor=None):
        N = len(frame)
        df = 1.0 / dt
        f = arange(N) * df / N
        xf = F(frame, dt=dt, n=N, window=window)(f)
        T = N * dt
        I = ((1.0 / T) * abs(xf)) ** 2 # Parseval for FFT: the sum of all coeffs
        # of this I is <x**2>. Check in an assert ?
        I = 2 * I[:N/2+1]  # sum(I) = <x**2> does NOT hold anymore : the first
        # and last value have been doubled when there is no reason too (we
        # have removed no alias for them. But 0.5 * first + 0.5*last + sum(others)
        # is <x**2> now

        fk = arange(0, N/2 + 1) * df / N
        low = maximum(fk - 0.5 * df / N, 0.0) # avoid neg. freqs 
        # (halve the first bandwidth)
        high = minimum(fk + 0.5 * df / N, 0.5 * df) # avoid freqs > nyquist
        # (halves the last bandwidth)
        l = 10 * log10(I/(high - low)) + 96

        masks = [mask(l=l_, low=low_, high=high_) 
                 for (l_, low_, high_) in zip(l, low, high)]
        if floor:
            masks = [floor] + masks
        return sum(masks)

    __radd__ = __add__

# TODO: create a function and use Mask as a decorator instead ?

class ATH_Type(Mask):
    "Absolute Threshold of Hearing (input in Hz, output in dB)"
    def __init__(self):
        self._masks = [ATH_Type.ATH]
        
    @staticmethod
    def ATH(f):
        f = array(f, copy=False)
        f = f / 1000.0
        return 3.64 * f ** (-0.8) - 6.5 * exp(-0.6 * (f-3.3) ** 2) + 1e-3 * f ** 4

ATH = ATH_Type()

def mask(L=None, l=None, low=None, high=None, fc=None, bandwidth=None):
    """
    Compute the mask of band-limited noise or a pure tone masker

    Parameters
    ----------
    L or l : floats
        masker pressure level or power density [dB]
    low, high or fc, bandwidth: floats
        masker lowest/highest frequency or center frequency/bandwidth [Hz]

    Returns
    -------
    mask: function 
        Computes the mask level [dB] for a given frequency [Hz].
    """

    lh = (low is not None) and (high is not None) and \
         (fc is None) and (bandwidth is None)
    fb = (low is None) and (high is None) and \
         (fc is not None) and (bandwidth is not None)

    if not lh != fb:
        error = "define exactly one of (low, high) or (fc, bandwidth)."
        raise ValueError(error)
    if not (L is None) != (l is None):
        error = "define L or L (not both)." 
        raise ValueError(error)

    if low is None:
        low  = fc - 0.5 * bandwidth 
        high = fc + 0.5 * bandwidth
    elif fc is None:
        fc = 0.5 * (high + low)
        bandwidth = high - low

    if L is None:
        L = l + 10 * log10(bandwidth)

    CB = critical_bandwidth
    def mask_(f):
        f = array(f, copy=False)
        level = zeros(shape(f))
        if not bandwidth: # tonal masker
            near = abs(f-fc) <= 0.5 * CB(f)
            level[near] = L
            far = logical_not(near)
            level[far] = -inf
        else:
            effective_width = minimum(fc + 0.5 * bandwidth, f + 0.5 * CB(f)) - \
                              maximum(fc - 0.5 * bandwidth, f - 0.5 * CB(f))
            l = L - 10 * log10(bandwidth)
            level = 10 * log10((10 ** (l/10.0)) * effective_width)
            level[effective_width <= 0] = - inf # BUG: WONT WORK IF THE ARGUMENT
            # IS A SCALAR !
        return level
    return Mask(mask_)

#-------------------------------------------------------------------------------
# Mask Computation
#-------------------------------------------------------------------------------

# TODO: add 'mask' and 'mask_from_frame' as static methods of Mask, under the
#       name 'from_masker' and 'from_frame' ?


# TODO: review / plots / tests (with LP and sin signals for example), AWARE 
# appears to be borked now ...

# MAKE SURE THE FRAMES HAVE BEEN NORMALIZED to [-1, 1] !!!

# TODO: Add a tonal detection (or handle ALL frequs as pure sines ?) option ?
# TODO: migrate as a static method of Mask ?
def mask_from_frame(frame, df=44100.0, window=hanning, floor=None):
    N = len(frame)
    dt = 1.0 / df
    f = arange(N) * df / N
    xf = F(frame, dt=dt, n=N, window=window)(f)
    T = N * dt
    I = ((1.0 / T) * abs(xf)) ** 2 # Parseval for FFT: the sum of all coeffs
    # of this I is <x**2>. Check in an assert ?
    I = 2 * I[:N/2+1]  # sum(I) = <x**2> does NOT hold anymore : the first
    # and last value have been doubled when there is no reason too (we
    # have removed no alias for them. But 0.5 * first + 0.5*last + sum(others)
    # is <x**2> now

    fk = arange(0, N/2 + 1) * df / N
    low = maximum(fk - 0.5 * df / N, 0.0) # avoid neg. freqs (halve the first
    # bandwidth)
    high = minimum(fk + 0.5 * df / N, 0.5 * df) # avoid freqs > nyquist
    # (halves the last bandwidth)
    l = 10 * log10(I/(high - low)) + 96

    masks = [mask(l=l_, low=low_, high=high_) 
             for (l_, low_, high_) in zip(l, low, high)]
    if floor:
        masks = [floor] + masks
    return sum(masks)

# Make a test with a 44100 / 51.2 (k=10) pure sine, check that the mask at
# center frequency has the proper value (given that means(frame*frame)=0.5, 
# that is SPL of 96 - 3 = 93.0 dB and very low mask values outside of the 
# neighbourghood. NOTES: USE ones AS A WINDOW FOR THIS TEST !


# Use mask direcly as a parameter instead of frame ? And give as inputs the
# subbands (boundaries ?) and number of points to use for maxing ? Mmmm
# OVERALL number of points or by subband number of points ?
# (max could also be an argument, mean is for example another option).

def get_subband_mask(frame, dt=None, M=32):
    df = 1.0 / dt
    N = len(frame)
    mask_ = mask_from_frame(frame, floor=ATH)
    f = (arange(N)*df/N)[:N/2+1] # the pick max value for every 9 points, overlap, 1.
    mask__ = mask_(f)
    max_frame = (N/2) / M + 1
    maxs = zeros(M)
    for k in range(M):
        k_shift = k * (max_frame - 1)
        #print "k:", (0 + k_shift), (max_frame + k_shift)
        #print "  ", mask_[(0+k_shift):(max_frame+k_shift)]
        #print "  ->", min(mask_[(0+k_shift):(max_frame+k_shift)])
        maxs[k] = min(mask__[(0+k_shift):(max_frame+k_shift)])
    return maxs

#-------------------------------------------------------------------------------
# Quantization
#-------------------------------------------------------------------------------

def step_size(level): # level: mask level in dB
    level = array(level, copy=False)
    return 10 ** ((level- 96.0 + 10*log10(12.0)) / 20.0)


# OK. Seen 2.35 x reduction on on example frame, that would do some 600 Mb/sec.
def num_bits(level, max_=1.7):
    n = maximum(0, log2(2*max_/step_size(level)))
    return ceil(n).astype(int)

def num_quantizer_values(level, max_=1.7):
    num_val = 2*max_/step_size(level)
    return ceil(num_val)

# BUG: review the formulas and take into account the range (-1.7, 1.7). We may
#      also return a number of value instead of bits ? No reason to. It would
#      even make sense to reduce the number of options for the number of bits
#      (such as 4, 8, 16, 32 ?). Think of very low values ... Affect to 0
#      does it make sense ? For a signal that is supposed to be the BIGGEST
#      in this region ? Have a look at the Moreau ... Group the data and use
#      scale factors in a table ?




#-------------------------------------------------------------------------------
# Unit Tests
#-------------------------------------------------------------------------------
def test_bark():
    """
    >>> bark(0.0)
    0.0
    >>> 0.95 <= bark(100.0) <= 1.05
    True
    >>> 24.0 <= bark(20000.0) <= 25.0
    True
    >>> shape(bark([0.0, 500.0, 1000.0]))
    (3,)
    >>> shape(bark([[0.0, 500.0], [1000.0, 2000.0]]))
    (2, 2)
    """

def test_hertz():
    """
    >>> f = arange(0.0, 20000.0, 100.0)
    >>> max(abs(f - hertz(bark(f)))) <= 1e-7
    True
    """

def test_critical_bandwidth():
    """
    >>> abs(critical_bandwidth(200.0) - 100.0) <= 5
    True
    >>> f = arange(0.0, 500.0, 100.0)
    >>> all(abs(critical_bandwidth(f) - 100.0) <= 20.0)
    True
    >>> f = arange(500.0, 20000.0, 200.0)
    >>> all(abs(critical_bandwidth(f) - 0.2 * f) / (0.2 * f) <= 0.50)
    True
    """

def test():
    import doctest
    doctest.testmod() 

if __main__:
    test()
















# ------------------------------------------------------------------------------
#N = 512
#n = arange(N)
#t = n * dt
#fr1 = 500.0 # 516.796875
#fr2 = 10000.0 # 10335.9375
##f_ = logspace(log10(20.0), log10(18000.0), 1000)
## Incredible: the experimental data is CLEAN when the frequencies found in
## the signal are EXACT multiples of df/N. Otherwise, it gets ugly; when
## the frequency is high (say 10 khz), their is a residual of maybe L - 40 dB
## but when it's low, that'so ugly it is not usable at all: the residual is
## wideband and hardly allows to distinguish the peak -- UPDATE, ok, I made
## that up ... 25-30 dB is something we can distinguish ... but that's soooo
## over the ATH threshold !
##
## Windowing improves the situation somehow ... but articifically lowers the
## maximal L detected ... sometimes by a factor say 4-5 dB ...
#frame = 1.0 * (sin(2*pi*fr1*t) + sin(2*pi*fr2*t))

#def plot_():
#    c = lambda f: f # or bark
#    f_ = linspace(0.0, 0.5*df, 10000)
#    plot(c(f_), add_masks(masks_from_frame(frame))(f_), 'k') 
#    plot(c(f_), ATH(f_), 'k:')
#    sbm = get_subband_mask(frame)
#    fk = (arange(M) * 0.5 + 0.25) * (df / M)
#    plot(c(fk), sbm, 'k+') 
#    return fk, sbm

## ------------------------------------------------------------------------------



#def v(df, p): # Got to make it support -inf value for p
#    # check the sign of df wrt the shape of the masks in the Fastl.
#    # (graph w.r.t. to the test tone: fast slope at low freq, slow (long)
#    # slope at high freqs.
#    df = ravel(df)
#    if p == -inf:
#        return ones(shape(df)) * (-inf)
#    r1 = (-8 < df) * (df <= -1)
#    r2 = (-1 < df) * (df <=  0)
#    r3 = ( 0 < df) * (df <=  1)
#    r4 = ( 1 < df) * (df <=  3)
#    out = (df <= -8) + (3 < df)
#    out_inf = 0.0 * df
#    out_inf[out==True] = -inf
#    v = r1 * ( - (-df -  1) * (17 - 0.15 * p) - 17) + \
#        r2 * 17 * df +\
#        r3 * ((0.4 * p + 6) * (-df)) +\
#        r4 * (17 * (-df +1) -(0.4 * p + 6)) + \
#        out_inf
#    return v

#def PT(f, p):
#    def _PT(_f):
#        return p - 1.525 - 0.275 * f - 4.5 + v(_f - f, p)
#    return _PT

#def PNT(f, p):
#    def _PT(_f):
#        return p - 1.525 - 0.175 * f - 0.5 + v(_f - f, p)
#    return _PT


#def T(f): # need to return inf, not raise an error, when f=0
#    # Uhu ? Works in array form
#    "Threshold in quiet"
#    f = ravel(f)
#    f = f/1000.0
#    return 3.64 * f ** (-0.8) - 6.5 * exp(-0.6 * (f-3.3) ** 2) + 1e-3 * f ** 4

#def S(x):
#    N = 512
#    assert(len(x) == N)
#    assert(max(abs(x)) <= 1) 
#    w = hanning(N)
#    x = w * x
#    xk = (fft(x)/N)[:(N/2+1)]
#    PN = 96 # Error in the Painter/Spanias: the normalization has to be
#    # 96, not 90 to get a max Pk of 84 for a perfectly resolved pure tone.
#    Pk = PN + 20*log10(abs(xk))
#    return Pk

#def tonal(S):
#    k_tonal = []
#    for k in range(2, 250): # Arf, N=512 is hard-coded ...
#        if S[k] > S[k-1] and S[k] > S[k+1]:
#            if k <= 63:
#                kr = [-2, +2]
#            if k <= 127:
#                kr = [-3,-2, 2, 3]
#            else:
#                kr = [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]
#            Sr = array([S[k_] for k_ in k + ravel(kr)])
#            tonal = all(S[k] >= Sr + 7)
#            if tonal:
#                k_tonal.append(k)
#    return k_tonal

#def Ptonal(S):
#    k_tonal = tonal(S)
#    kS = []
#    for k in k_tonal:
#        P = 10.0 * log10(10**(S[k-1]/10.0) + 10**(S[k]/10.0) + 10**(S[k+1]/10.0))
#        kS.append((k, P))
#    return kS

#def non_tonal(S):
#    ksc = arange(0, 257)
#    k_tonal = tonal(S)
#    ks = []
#    for k in ksc:
#        if (k in k_tonal) or (k-1 in k_tonal) or (k+1 in k_tonal):
#            pass
#        else:
#            ks.append(k)
#    return ks
#    
#def Pnon_tonal(S):
#    ks = non_tonal(S)
#    ek = []
#    for i in range(25):
#        ek.append([])
#    eS = zeros(25)
#    for k, Sk in enumerate(S):
#        fk = k * (44100.0 / 512)
#        bk = int(floor(bark(fk))) # find the bark range
#        ek[bk].append(k)
#        if k in non_tonal(S):
#            eS[bk] += 10**(S[k]/10.0) # add the power contrib.
#    P = 10*log10(eS)
#    #return ek, P
#    mean_k = []
#    for bk in range(25):
#        mean = gmean(ek[bk])
#        mean_k.append(mean)
#    res = []
#    for bk in range(25):
#        res.append((mean_k[bk], P[bk]))
#    return res



#N = 512 # Apparently, in the "real deal", there is some overlap between frames ?
## Is it the explanation of the 384 frame for filter banks and 512 for fft ?
## Dunno, this unclear ... 
#b = 16

#fs = 44100.0
#f = (N/4)*(fs/N) # Any multiple of (fs/N) will be resolved exactly.
## We select here a frequency in the middle of the frequency range.
#Ts = 1.0/fs
#n = arange(0, N)

#x = sin(2*pi*f*n*Ts)

## OK. In The Painter/Spaniass, the 2^(b-1) is here to put the signal into
## [-1, 1], and the N to compensate for the fft definition. We compensate for
## the fft later and we already have a signal in [-1, 1], so normalization
## can be skipped. RK: EXPLAIN more here why the normalization make sense
## and why N is the right factor (combination of we want the mean energy /sample
## and Parseval equality). Mmmmm factor of 2 issue when we drop the negative
## frequencies ? Need to compensate ?

## x = x / N / 2**(b - 1)

#assert(83<max(S(x))<85)
## We should try with low amplitude with the assumption that we have 16 bit
## uniform and CHECK that the Pk is -15. Anyway, we end up with an amplitude
## range of around 100 db SPL. 

#f2 = 440 # la
#x = (0.1*sin(2*pi*f*n*Ts) + sin(2*pi*f2*n*Ts))/ 1.1

#k2f = lambda k: k * 44100/512.0

#k = arange(0, 257)
#Sk = S(x)
#fk = k2f(k)
#Tk = T(fk)



#def Sm(x):
#        
#    Sk = S(x)
#    t = Ptonal(Sk)
#    nt = Pnon_tonal(Sk)
#    def _Sm_(f):
#        f = ravel(f)
#        b = bark(f)
#        Sm_ = 10 ** (T(f) / 10)
#        for k_, P in t:
#            mask = PT(bark(k2f(k_)), P)
#            a = 10 ** (mask(b) / 10.0)
#            Sm_ += a
#        print 80*"-"
#        for k_, P in nt:
#            mask = PNT(bark(k2f(k_)), P)
#            a = 10 ** (mask(b) / 10.0)
#            if any(isnan(Sm_ + a)):
#                v = ValueError()
#                v.Sm_ = Sm_
#                v.a = a
#                v.mb = mask(b)
#                v.k_ = k_
#                v.P = P
#                raise v
#            Sm_ += a
#            #print a, Sm_
#        return ravel(10*log10(Sm_))
#    return _Sm_

#def Sm32(x):
#    Sm_ = Sm(x)(k2f(arange(0, 257)))
#    Sm32_ = []
#    for i in range(32):
#        Sm32_.append(min(Sm_[8*i:(8*i+9)]))
#    return ravel(Sm32_)

#def pplot():
#  k = arange(0, 257)
#  fk = k2f(k)
#  bk = bark(fk)
#  clf(); plot(fk, S(x), "k", fk, T(fk), ":r"); 
#  kP = Ptonal(S(x))
#  for k_, P in kP:
#      plot(k2f(k_), P, "ko")
#  for k_, P in Pnon_tonal(S(x)):
#      plot(k2f(k_), P, "k+")
#  plot(fk, Sm(x)(fk), "g--")
#  axis([0,20000, -20, 100]); grid(True); show()
#  
## Issue: Sm is always inf for k = 0 ??? Yeah, not an issue: the mask is going
## to get computed pessimistically (min) over the 32 subbands, so that should
## solve the issue.
#    
#def pplot2():
#  k = arange(0, 257)
#  fk = k2f(k)
#  bk = bark(fk)
#  clf(); plot(bk, S(x), "k", bk, T(fk), ":r"); 
#  kP = Ptonal(S(x))
#  for k_, P in kP:
#      plot(bark(k2f(k_)), P, "ko")
#  for k_, P in Pnon_tonal(S(x)):
#      plot(bark(k2f(k_)), P, "k+")
#  plot(bk, Sm(x)(fk), "g--")
#  axis([0,25, -20, 100]); grid(True); show()

