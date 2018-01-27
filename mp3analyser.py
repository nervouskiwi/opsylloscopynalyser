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

def draw_polar(data, color, alpha, offset=60000, fill=True):
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
    ax.plot(phi, rho, color=color, alpha=alpha)

    #Corlorize filling between offset (0 point of radial representation) and amplitude value
    if fill == True:
        ax.fill_between(phi, numpy.ones(len(rho))*offset, rho, facecolor=color, interpolate=True)
    #ax.set_rmax(5*(max(data) + offset))
    ax.set_rticks([])  # less radial ticks
    #ax.get_raxis.set_visible(False)
    #ax.get_paxis.set_visible(False)
    #ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(False)

    ax.set_title("NervousApache", va='bottom')

def triband_iris(sound, freq, filt_args={'b':[],'m':[],'h':[]}, draw_polar_args={'colors':{'l':None,'b':None,'m':None,'h':None},'alphas':{'l':None,'b':None,'m':None,'h':None},'offsets':{'l':None,'b':None,'m':None,'h':None}, 'fill':{'b':True,'m':True,'h':True}}):

    pyplot.figure()


    for band in ['b', 'm', 'h']:
        fargs = [sound,freq]
        if filt_args[band] != []:
            if len(filt_args[band]) == 2:
                fargs.extend(filt_args[band])
        if band == 'b':
            band_output = bass_filter(*fargs)
        elif band == 'm':
            band_output = band_filter(*fargs)
        elif band == 'h':
            band_output = high_filter(*fargs)

        if draw_polar_args['colors'][band] is not None:
            col = draw_polar_args['colors'][band]
        else:
            if band == 'b':
                col = 'g'
            elif band == 'm':
                col = 'b'
            elif band == 'h':
                col = 'r'
        if draw_polar_args['alphas'][band] is not None:
            al = draw_polar_args['alphas'][band]
        else:
            al = 0.5
        if draw_polar_args['offsets'][band] is not None:
            of = draw_polar_args['offsets'][band]
        else:
            of = 60000
        fi = draw_polar_args['fill'][band]
        draw_polar(band_output, col, al, of, fi)

    # if filt_args['b'] != []:
    #     if len(filt_args['b']) == 2:
    #         fargs = [sound,freq]
    #         fargs.extend(filt_args['b'])
    #         bass_output = bass_filter(*fargs)
    # else:
    #     bass_output = bass_filter(sound,freq)
    # if filt_args['m'] != []:
    #     if len(filt_args['m']) == 2:
    #         fargs = [sound,freq]
    #         fargs.extend(filt_args['m'])
    #         bass_output = bass_filter(*fargs)
    # else:
    #     band_output = band_filter(sound,freq)
    # if filt_args['h'] != []:
    #     if len(filt_args['h']) == 2:
    #         fargs = [sound,freq]
    #         fargs.extend(filt_args['h'])
    #         bass_output = bass_filter(*fargs)
    # else:
    #     high_output = high_filter(sound,freq)


    u = numpy.ones(len(sound)//chuck_down_factor)
    # u = numpy.ones(len(s1))

    if draw_polar_args['colors']['l'] is not None:
        col = draw_polar_args['colors']['l']
    else:
        col = 'w'
    if draw_polar_args['alphas']['l'] is not None:
        al = draw_polar_args['alphas']['l']
    else:
        al = 0.5
    if draw_polar_args['offsets']['l'] is not None:
        of = draw_polar_args['offsets']['l']
    else:
        of = 20
    draw_polar(u, col, al, of)

    # if draw_polar_args['colors']['b'] is not None:
    #     col = draw_polar_args['colors']['b']
    # else:
    #     col = 'g'
    # if draw_polar_args['alphas']['b'] is not None:
    #     al = draw_polar_args['alphas']['b']
    # else:
    #     al = 0.5
    # if draw_polar_args['offsets']['b'] is not None:
    #     of = draw_polar_args['offsets']['b']
    # else:
    #     of = 60000
    # fi = draw_polar_args['fill']['b']
    # draw_polar(bass_output, col, al, of, fi)
    #
    #
    # if draw_polar_args['colors']['m'] is not None:
    #     col = draw_polar_args['colors']['m']
    # else:
    #     col = 'b'
    # if draw_polar_args['alphas']['m'] is not None:
    #     al = draw_polar_args['alphas']['m']
    # else:
    #     al = 0.5
    # if draw_polar_args['offsets']['m'] is not None:
    #     of = draw_polar_args['offsets']['m']
    # else:
    #     of = 60000
    # fi = draw_polar_args['fill']['m']
    #
    # draw_polar(band_output, col, al, of, fi)
    #
    #
    # if draw_polar_args['colors']['h'] is not None:
    #     col = draw_polar_args['colors']['h']
    # else:
    #     col = 'r'
    # if draw_polar_args['alphas']['h'] is not None:
    #     al = draw_polar_args['alphas']['h']
    # else:
    #     al = 0.5
    # if draw_polar_args['offsets']['h'] is not None:
    #     of = draw_polar_args['offsets']['h']
    # else:
    #     of = 60000
    # fi = draw_polar_args['fill']['h']
    # draw_polar(high_output, col, al, of, fi)


    pyplot.show()

def spectrum(sound, freq):
    pyplot.figure()
    k = numpy.arange(len(sound))
    T = len(sound)/freq
    frq = k/T
    Y = numpy.fft.fft(sound)/len(sound)
    plot(frq,abs(Y),'r')
    pyplot.show()

def crush_sound_data(sound, framerate=30, freq=44100, data_percentage=100, first_data_mode=True):
    if data_percentage >100:
        data_percentage=100

    elif data_percentage < 0:
        data_percentage = 0

    chunk_size = freq//framerate
    print(chunk_size)
    chunks=[]
    for i in list(range(len(sound)//chunk_size)):

        if first_data_mode:
            chunk_data = sound[i*chunk_size:(i+1)*chunk_size*data_percentage//100]
        else:
            chunk_data = sound[i*chunk_size:chunk_size*data_percentage//100:(i+1)*chunk_size]


        chunks.append(chunk_data)

    return chunks


if __name__ == '__main__':
    #/home/apache/Musique/No more Lord.wav
    #C:/Users/NervousKiwi/MusicStuff/Battrey/Battrey 4/Battery 4 Factory Library/Samples/One Shots/SFX/SFX Autopsy 2 V2.wav
    data = load_wav('/home/apache/Musique/No more Lord.wav')
    chuck_down_factor = 60
    snd = data[1]
    # print(snd.dtype)
    freq = data[0]
    # print(freq)
    #snd = snd / (2.**15)
    snd = snd[:,0]
    # s1 = [float(s1[i]) for i in list(range(len(s1)))]
    sound = [float(snd[i]) for i in list(range(len(snd)))]
    sound = sound[:len(sound)//chuck_down_factor]
    # duration = snd.shape[0]/data[0]
    # print(snd.shape[0])
    # print(len(sound))
    chunks = crush_sound_data(sound,framerate=60, data_percentage=50,first_data_mode=False)
    triband_iris(sound, freq)
    spectrum(sound, freq)

    #pol.savefig('IrisdB_20.eps', format='eps', dpi=1)
