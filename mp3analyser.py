import sys
from matplotlib import pyplot
from matplotlib.pylab import plot
from matplotlib.pylab import xlabel
from matplotlib.pylab import ylabel
from matplotlib.pylab import show
from matplotlib.pylab import xscale
from matplotlib.pylab import arange
from matplotlib.pylab import grid
from matplotlib.pylab import axvline
from scipy.io import wavfile
from scipy import signal
import numpy

def load_wav(path):
    sampFreq, snd = wavfile.read(path)
    return [sampFreq, snd]

def draw_snd(data, freq):

    timA = arange(0,len(data),1)
    timA = timA / freq

    #plot
    #print(timA.shape)#

    pyplot.plot(timA, data, color='k', alpha=0.5)
    ylabel=('Amp')
    xlabel('Time (ms)')
    show()

def bass_filter(data, freq, cutoff=200, order=7):
    nyq = freq/2
    norm_cutoff = cutoff/nyq
    b, a = signal.butter(N=order, Wn=norm_cutoff, btype='low', analog=False)
    print(a,b)
    #w, h = signal.freqs(b,a)
    #plot(w, 20 * numpy.log10(abs(h)))
    #xscale('log')
    #xlabel('Freqs')
    #ylabel('Amp db')
    #grid(which='both', axis='both')
    #axvline(200, color='green')
    #show()
    # t = arange(0,len(data),1)

    output = signal.filtfilt(b,a, data)


    # pyplot.plot(t, output, color='g', alpha=0.5)

    return output

def high_filter(data, freq, cutoff=5500, order=7):
    nyq = freq/2
    norm_cutoff = cutoff/nyq
    b, a = signal.butter(N=order, Wn=norm_cutoff, btype='high', analog=False)
    print(a,b)

    # t = arange(0,len(data),1)

    output = signal.filtfilt(b,a, data)

    # pyplot.plot(t, output, color='r', alpha=0.5)

    #matplotlib.use('Agg')


    #fig.savefig('testeps.eps', format='eps', dpi=1000)
    #fig.savefig(sys.stdout)
    return output

def band_filter(data, freq, cutoff=(1000,5000), order=7):
    nyq = freq/2
    norm_cutoff_min, norm_cutoff_max = cutoff[0]/nyq, cutoff[1]/nyq
    b, a = signal.butter(N=order, Wn=(norm_cutoff_min, norm_cutoff_max) , btype='bandpass', analog=False)
    print(a,b)



    output = signal.filtfilt(b,a, data)

    return output

def draw_polar(data, color, offset=60000):
    phi = []
    rho = []
    print(max(abs(data)))

    #offset = 3*max(abs(data))
    for i in list(range(len(data))):
    	phi.append(float(i)/float(len(data))*2*numpy.pi)
    	rho.append(offset+data[i])

    phi = list(reversed(phi))
    # print(rho)
    # print(phi)

    ax = pyplot.subplot(111, projection='polar')
    ax.plot(phi, rho, color=color, alpha=0.5)
    #ax.set_rmax(5*(max(data) + offset))
    ax.set_rticks([])  # less radial ticks
    #ax.get_raxis.set_visible(False)
    #ax.get_paxis.set_visible(False)
    #ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(False)

    ax.set_title("A line plot on a polar axis", va='bottom')


if __name__ == '__main__':
    data = load_wav('C:/Users/NervousKiwi/MusicStuff/Exports/Apache Crew/Ricks Advice.wav')
    snd = data[1]
    # print(snd.dtype)
    freq = data[0]
    print(freq)
    #snd = snd / (2.**15)
    s1 = snd[:,0]
    s1 = [float(s1[i]) for i in list(range(len(s1)))]
    duration = snd.shape[0]/data[0]
    print(snd.shape[0])
    print(len(s1))
    # print(snd.shape)
    # print(snd.shape[0]/data[0])
    #draw_snd(s1, freq)
    # for i in list(range(12000,12500)):
    #     print(s1[i])


    # fig = pyplot.figure()
    bass_output = bass_filter(s1,freq)
    band_output = band_filter(s1,freq)
    high_output = high_filter(s1,freq)

    # t = arange(0,len(data),1)
    # pyplot.plot(t, bass_output, color='g', alpha=0.5)
    # pyplot.show()

    pol = pyplot.figure()
    u = numpy.ones(len(s1))
    draw_polar(u, 'w',offset=20)
    draw_polar(bass_output, 'g')
    draw_polar(band_output, 'b')
    draw_polar(high_output, 'r')
    pyplot.show()
    #pol.savefig('IrisdB_20.eps', format='eps', dpi=1)
