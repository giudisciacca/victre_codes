import matplotlib.pyplot as plt
import numpy
import h5py
import matplotlib.patches as mpatches
import seaborn as sns
def load_v(name):
    fol = '/cs/research/medim/gdisciac/SOLUS/example/VICTRE_PARADIGM/';

    #fData = h5py.File(fol + 'violinplotdat4.mat', 'r')
    fData = h5py.File(fol + name, 'r')
    log = 0;
    if log == 1:
        bAt =numpy.log( 10+numpy.array(fData.get('btmpAt')))
        bAr = numpy.log(10+numpy.array(fData.get('btmpAr')))
        mAt = numpy.log(10+numpy.array(fData.get('mtmpAt')))
        mAr = numpy.log(10+numpy.array(fData.get('mtmpAr')))
    else:
        bAt =  numpy.array(fData.get('btmpAt'))
        bAr =  numpy.array(fData.get('btmpAr'))
        mAt =  numpy.array(fData.get('mtmpAt'))
        mAr =  numpy.array(fData.get('mtmpAr'))

    featt= list()
    featr = list()
    for i in range(0,8):
        featt.append(bAt[i, :])
        featr.append(bAr[i, :])
        featt.append(mAt[i, :])
        featr.append(mAr[i, :])
    return featt,featr

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

l = [635 ,670 ,685,785 ,905,930 ,975 ,1060]
#l = [635 ,635,670,670 ,685,685,785,785 ,905,905,930,930 ,975,975 ,1060,1060]

pos = [1,1.70,3,3.70,5,5.7,7,7.70,9,9.70,11,11.70,13,13.70,15,15.7]
featt,featr=load_v('violinplotdat.mat')
plt.figure()
ax1 = plt.gca()
ax1.set_xticks([1.35,3.35,5.35,7.35,9.35,11.35,13.35,15.35])
ax1.set_xticklabels(l)
ax1.set_xlabel('Wavelength (nm)',fontsize=20)#%, fontdict=15)
ax1.set_ylabel('$\mu_{a}$ (mm$^{-1}$)',fontsize=20)#, fontdict=15)
ax1.tick_params(axis='both', which='major', labelsize=15)
part1t = ax1.violinplot(featt,pos,quantiles=[[]]*16,showextrema=False);

ax2 = ax1.twiny()
#ax2.set_xticklabels(l)
part1r = ax2.violinplot(featr,pos,quantiles=[[0.05,0.25,0.75,0.95]]*16,showextrema=False);
ax2.set_xticks([])
i=0;
for pc in part1t['bodies']:
    if i == 0:
        pc.set_facecolor('red')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        i=1
    elif i == 1:
        pc.set_facecolor('green')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        i=0
for pc in part1r['bodies']:
    if i == 0:
        pc.set_facecolor('red')
        pc.set_edgecolor('black')
        pc.set_alpha(0.4)
        i = 1
    elif i == 1:
        pc.set_facecolor('green')
        pc.set_edgecolor('black')
        pc.set_alpha(0.4)
        i = 0
plt.show()




