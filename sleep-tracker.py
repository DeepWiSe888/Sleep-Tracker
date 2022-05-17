import os
from enum import Enum
import socket
from time import ctime
import threading
import time
import struct
# import numpy.random.common
# import numpy.random.bounded_integers
# import numpy.random.entropy
import numpy as np
from queue import Queue
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
#from scipy.interpolate import spline

from scipy.interpolate import make_interp_spline
import csv


fps = 40
FPS = 20 * fps
OFFSET = 0
MAX_BIN = 93

brpm,arpm = signal.butter(3,[0.1,0.5],'bandpass',analog=False,output='ba',fs=fps)
bbpm,abpm = signal.butter(5,[1,2],'bandpass',analog=False,output='ba',fs=fps)

class PhaseCorrection(object):
    def __init__(self, bbframes, bin_ref):
        self.bin_ref = bin_ref
        self.bin_angle = np.angle(bbframes[:,bin_ref]).mean()

    def filter(self, frame):
        phase_correction = self.bin_angle - np.angle(frame[self.bin_ref])
        return frame * np.exp(1j*phase_correction)

#还需调整
class PhaseCorrection1(object):
    def __init__(self, bbframes, bin_ref):
        self.bin_ref = bin_ref
        self.bin_angle = np.angle(bbframes[:,bin_ref]).mean()

    def filter(self, frames):
        phase_correction = self.bin_angle - np.angle(frames[:,self.bin_ref])
        phc = np.exp(1j*phase_correction)
        phc = phc.reshape(phc.shape[0],1)
        return frames* phc

class MyThread(threading.Thread):
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

global toshow
toshow = Queue()
def recvdata():
    recvBuffer = bytes()
    headerSize = 16
    packSize = 1128
    while True:
        recvBuffer = bytes()
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("",15647))  # ip 137.0.0.1
        s.listen(1)
        conn,address = s.accept()
        print("be connected successfully.")
        while True:
            try:
                data = conn.recv(1024)
                # print("from client:{},recv len:{}".format(s.getpeername(),len(data)))
                if not data:
                    print('disconnect')
                    s.close()
                    recvBuffer = ''
                    break
                
                recvBuffer += data
                index = recvBuffer.find(b'xglx4dataframe')
                if(index == -1 ):
                    continue
                else:
                    pack = recvBuffer[index:index+packSize]
                    if len(pack) == packSize:
                        z = np.zeros((138,2))
                        # frameno = struct.unpack('i',pack[16:20])
                        # print(frameno)
                        orgI = pack[24:24+552]
                        orgQ = pack[576:576+552]
                        i = struct.unpack('138f',orgI)
                        q = struct.unpack('138f',orgQ)
                        i = np.array(i)
                        q = np.array(q)
                        z[:,0] = i
                        z[:,1] = q
                        toshow.put(z)
                        # qlsit.put(z)
                        recvBuffer = recvBuffer[index+packSize:]
                    else:
                        continue
                
            except Exception as e:
                print(e)
                pass

def calcuPer(data,thre):
    row = data.shape[0]
    col = data.shape[1]
    perdata = (data > thre).sum(0)/row*100
    return perdata

def calcuSigQ(snr_dB):
    SigQ = 0
    if snr_dB < 23:
        SigQ=0
    elif snr_dB < 26:
        SigQ=1
    elif snr_dB < 29:
        SigQ=2
    elif snr_dB < 32:
        SigQ=3
    elif snr_dB < 35:
        SigQ=4
    elif snr_dB < 38:
        SigQ=5
    elif snr_dB < 41:
        SigQ=6
    elif snr_dB < 44:
        SigQ=7
    elif ( snr_dB < 47 ):
        SigQ=8
    elif ( snr_dB < 50 ):
        SigQ=9
    else:
        SigQ=10
    return SigQ

def findpeak(in_arr):
    peaks = 0
    peakmax = 0
    for i in range(1,len(in_arr) - 1):
        if in_arr[i - 1] < in_arr[i] and in_arr[i] < in_arr[i + 1] and in_arr[i] > peakmax:
            peaks = i
            peakmax  = in_arr[i]
    
    return peaks

def calcuRPM(x,fastper,lastpeak):
    filter_data = signal.filtfilt(brpm,arpm,x,axis=0)
    filter_sum = np.sum(np.abs(filter_data),axis=0)
    newpeak = findpeak(filter_sum)
    peak = newpeak
    if abs(newpeak - lastpeak) > 5:
        if fastper < 10:
            peak = lastpeak
    breathe = x[:,peak]
    breathe_filter = signal.filtfilt(brpm,arpm,breathe,axis=0)
    angle = np.angle(breathe_filter[-1])
    amp = np.abs(breathe_filter[-1])
    breathe_fft = abs(np.fft.fft(breathe_filter))
    breathe_fft_max = np.max(breathe_fft[:int(len(breathe_fft)/2)])
    breathe_fft_max_index = (np.where(breathe_fft[:int(len(breathe_fft)/2)] == breathe_fft_max))[0][0]
    rpm = breathe_fft_max_index * fps/FPS * 60
    return rpm,peak,newpeak,angle,amp

def calcuBPM(x):
    filter_data = signal.lfilter(bbpm,abpm,x,axis=0)
    mean_data = filter_data - np.mean(filter_data)
    bpm_fft = np.abs(np.fft.fft(mean_data))
    bpm_fft_max = np.max(bpm_fft[:int(len(bpm_fft)/2)])
    bpm_fft_max_index = (np.where(bpm_fft[:int(len(bpm_fft)/2)] == bpm_fft_max))[0][0]
    bpm = bpm_fft_max_index * fps/FPS * 60
    return bpm

def calcumove(iq,thre1,thre2):
    iq = iq - np.mean(iq,axis=0)
    win = iq.shape[0]
    handata = np.hanning(win)[:, None] * np.array(iq)
    fft = np.fft.fft(handata, axis=0)
    fftabs = np.abs(fft)
    log = 20*np.log10(fftabs)
    log[log < thre1] = thre1
    movelist = calcuPer(log,thre1)
    moveper = len(np.where(movelist >= thre2)[0]) /MAX_BIN *100

    return fft,movelist,moveper

def handle():
    global toshow

    datas = np.zeros((1,138,2))
    frecv = True
    lastpeak = 0
    peakchange = 0

    rpmlist = np.zeros((120,))
    rpmlist_s = np.zeros((120,))
    bpmlist = np.zeros((120,))
    anglelist = np.zeros((120,))
    amplist = np.zeros((120,))
    distancelist = np.ones((120,))*0.4
    slowhistory = np.zeros((120,))
    fasthistory = np.zeros((120,))

    


    NoMovement = 1
    Movement = 0
    Movementtacking = 0
    breathing = 0
    state = 3

    FMN = 0
    FNMN = 0

    SMN = 0
    SNMN = 0

    exists = 0
    exist = 0
    noexist = 0


    # csvfile = 'xethru_sleep_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
    # f =  open(csvfile,'w',newline='')
    # writer = csv.writer(f,delimiter=';')
    # writer.writerow(['TimeStamp','FrameCounter','State','RPM','BPM','ObjectDistance','SignalQuality','MovementSlow','MovementFast'])
    # f.close()

    while True:
        if toshow.empty():
            time.sleep(0.01)
            continue
        else:
            data = toshow.get()
            data = data.reshape(1,138,2)
            if frecv:
                datas = data
                frecv = False
            else:
                datas = np.vstack((datas,data))
            l = datas.shape[0]
            if(l >=FPS):
                t1 = time.time()
                iqdata1 = datas[-FPS:,:,:]

                org = iqdata1[:,:,0]+1j*iqdata1[:,:,1]
                x = org[:,OFFSET:MAX_BIN]

                ###不做相位校正，会影响心率的计算
                pc = PhaseCorrection(x,1)
                for i in range(FPS):
                    x[i,:] = pc.filter(x[i,:])

                ###slow move
                slowwin = 20*fps
                slowiq = x
                slowfft,slowlist,slowper = calcumove(slowiq,-47,10)

                ###fast move
                fastwin = 6*fps
                fastiq = x[FPS-fastwin:FPS,:]
                fastfft,fastlist,fastper= calcumove(fastiq,-45,10)

                ###calcu rpm 
                rpm,peak,newpeak,angle,amp = calcuRPM(slowiq,fastper,lastpeak)
                amp = amp * 1000
                bpm = 0
                if breathing == 1:
                    bpm = calcuBPM(slowiq[:,peak])
                

                ####Singal Quality amplitude peak value/Noise floor
                SigQPeak = np.max(np.abs(slowfft))
                Nf = np.mean(np.abs(slowfft[:,-10:]))
                snr_dB = 20*np.log10(SigQPeak/Nf)
                SigQ = calcuSigQ(snr_dB) 


                ###exist
                fastlistsum = np.sum(fastlist)
                slowlistsum = np.sum(slowlist)
                if fastlistsum > 0 or slowlistsum > 5:
                    exist += 1
                    noexist = 0
                else:
                    exist = 0
                    noexist += 1
                if exist > 3 and exists == 0:
                    exists = 1
                elif noexist > 10 and exists == 1:
                    exists = 0
                    

                # if abs(newpeak - lastpeak) > 5:
                #     peakchange += 1
                # if peakchange >= 5:
                #     lastpeak = newpeak
                #     peakchange = 0
                # else:
                #     lastpeak = peak

                if breathing == 1 or Movementtacking == 1:
                    lastpeak = peak
                else:
                    lastpeak = newpeak

                distance = (peak) * 0.0514 + 0.4

                if slowper > 5 or exists == 1:
                    SMN += 1
                    SNMN = 0
                elif slowper <= 5 and fastper <= 5 and SigQ <= 2:
                    SMN  = 0
                    SNMN += 1

                if fastper > 10:
                    if breathing == 1:
                        if np.where(fastlist >10 )[0][0] < peak:
                            FMN +=1
                            FNMN = 0
                        else:
                            FMN = 0
                            FNMN += 1
                    else:
                        FMN +=1
                        FNMN = 0
                else:
                    FMN = 0
                    FNMN += 1


                if NoMovement == 1 and slowper > 5:
                    Movement = 1
                    Movementtacking = 0
                    NoMovement = 0
                    breathing = 0
                    state = 1
                elif Movement == 1:
                    if SNMN > 25:
                        Movement = 0
                        Movementtacking = 0
                        NoMovement = 1
                        breathing = 0
                        state = 3
                    elif SMN >= 10  and FNMN >= 5:
                        Movement = 0
                        Movementtacking = 1
                        NoMovement = 0
                        breathing = 0
                        state = 2
                elif Movementtacking == 1:
                    if SMN >= 15  and FNMN >= 10:
                        Movement = 0
                        Movementtacking = 0
                        NoMovement = 0
                        breathing = 1
                        state = 0
                    elif FMN > 5: # or  np.std(distancelist[-6:]) > 0.2
                        Movement = 1
                        Movementtacking = 0
                        NoMovement = 0
                        breathing = 0
                        state = 1
                elif breathing == 1:
                    if FMN > 3: # or  np.std(distancelist[-6:]) > 0.2
                        Movement = 1
                        Movementtacking = 0
                        NoMovement = 0
                        breathing = 0
                        state = 1

                if breathing == 0:
                    rpm = 0
                    bpm = 0
                    angle = 0
                    amp = 0
                elif exists == 0:
                    rpm = 0
                    bpm = 0
                    angle = 0
                    amp = 0
                    distance = 0
                elif distance >= 2:
                    bpm = 0

                # if breathing == 1 or Movementtacking == 1 :
                #     distance += (2 * 0.0514)
                rpm_s = rpm
                if rpm > 0:
                    nozerosrpm = [r for r in rpmlist[-5:] if r > 0]
                    nozerosrpm.append(rpm)
                    rpm_s = np.mean(nozerosrpm)
                rpmlist = np.append(rpmlist,rpm)
                rpmlist = rpmlist[1:]

                rpmlist_s = np.append(rpmlist_s,rpm_s)
                rpmlist_s = rpmlist_s[1:]

                if bpm > 0:
                    nozerosbpm = [r for r in bpmlist[-10:] if r > 0]
                    nozerosbpm.append(bpm)
                    bpm = np.mean(nozerosbpm)
                bpmlist = np.append(bpmlist,bpm)
                bpmlist = bpmlist[1:]


                if angle > 0:
                    nozerosangle = [r for r in anglelist[-5:] if r > 0]
                    nozerosangle.append(angle)
                    angle = np.mean(nozerosangle)
                anglelist = np.append(anglelist,angle)
                anglelist = anglelist[1:]

                if amp > 0:
                    nozerosamp = [r for r in amplist[-5:] if r > 0]
                    nozerosamp.append(amp)
                    amp = np.mean(nozerosamp)
                amplist = np.append(amplist,amp)
                amplist = amplist[1:]

                distancelist = np.append(distancelist,distance)
                distancelist = distancelist[1:]
                slowhistory = np.append(slowhistory,slowper)
                slowhistory = slowhistory[1:]
                fasthistory = np.append(fasthistory,fastper)
                fasthistory = fasthistory[1:]
                
                showrpm = 0
                showrpm_s = 0
                showbpm = 0
                showamp = 0
                showangle = 0
                if breathing == 1:
                    showrpm = rpmlist[-1]
                    showrpm_s = rpmlist_s[-1]
                    showbpm = bpmlist[-1]
                    showamp = amplist[-1]
                    showangle = anglelist[-1]
                


                #2018-08-20T21:49:31.153+08:00
                #writer.writerow(['TimeStamp','FrameCounter','State','RPM','BPM','ObjectDistance','SignalQuality','MovementSlow','MovementFast'])
                
                # f =  open(csvfile,'a+',newline='')
                # writer = csv.writer(f,delimiter=';')
                # timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[0:23] + '+08:00'
                # writer.writerow([timestamp,0,state,'{:.1f}'.format(showrpm_s),'{:.1f}'.format(showbpm),'{:.4f}'.format(distance),SigQ,'{:.4f}'.format(slowper),'{:.4f}'.format(fastper)])
                # f.close()

                t2 = time.time()
                result = 'State:{};RPM:{:.1f};BPM:{:.1f},ObjectDistance:{:.2f};SignalQuality:{};MovementSlow:{:.2f};MovementFast:{:.2f},Time:{:.3f}'.format(state,showrpm_s,showbpm,distance,SigQ,slowper,fastper,t2-t1)
                print(result)
                

                title = 'NoMovement'
                if state == 0:
                    title = 'Breathing'
                elif state == 1:
                    title = 'Movement'
                elif state == 2:
                    title = 'MovementTacking'

                fig = plt.figure('Wirush',figsize=(8,9))
                plt.clf()
                fig.suptitle(title)
                plt.subplots_adjust(left=0.075, bottom=0.050, right=0.925, top=0.950,
                                    wspace=0.145, hspace=0.290)
                sns.set_style('darkgrid')
                sns.axes_style()

                plt.subplot(711)
                plt.plot(rpmlist)
                plt.ion()
                plt.plot(rpmlist_s)
                plt.ylim(8,30)
                # plt.legend(['RPM:{:.1f}'.format(showrpm)],loc='upper right')
                plt.legend(['RPM:{:.2f}'.format(showrpm),
                            'AVG RPM:{:.2f}'.format(showrpm_s)],loc='upper right')
                plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])

                plt.subplot(712)
                plt.plot(bpmlist)
                plt.ylim(60,120)
                plt.legend(['BPM:{:.1f}'.format(showbpm)],loc='upper right')
                plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])

                plt.subplot(713)
                plt.plot(amplist)
                # plt.ylim(0.005,0.05)
                plt.legend(['RPM AMP:{:.3f}'.format(showamp)],loc='upper right')
                plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])

                plt.subplot(714)
                plt.plot(anglelist)
                plt.ylim(-5,5)
                plt.legend(['RPM PATTERN:{:.1f}'.format(showangle)],loc='upper right')
                plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])
                

                plt.subplot(715)
                plt.plot(distancelist)
                plt.ylim(0.39,5.0)
                plt.legend(['DISTANCE:{:.2f}'.format(distancelist[-1])],loc='upper right')
                plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])

                plt.subplot(716)
                plt.plot(slowhistory)
                plt.ion()
                plt.plot(fasthistory)
                plt.ylim(0,100)
                # plt.title('MOVEMENT HISTORY')
                plt.legend(['SLOW:{:.2f}'.format(slowhistory[-1]),'FAST:{:.2f}'.format(fasthistory[-1])],loc='upper right')
                plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])


                plt.subplot(7,2,13)
                # plt.hist(slowlist,bins=MAX_BIN)
                plt.bar(x=[i for i in range(MAX_BIN)], height=slowlist)
                plt.ylim(0,100)
                plt.xlim(0,MAX_BIN)
                plt.legend(['MOVEMENT SLOW'],loc='upper right')
                plt.xticks([0,20,40,60,80],[round(d*0.0514 + 0.4,2) for d in [0,20,40,60,80]])

                plt.subplot(7,2,14)
                # plt.hist(fastlist,bins=MAX_BIN)
                plt.bar(x=[i for i in range(MAX_BIN)], height=fastlist,label='MOVEMENT FAST')
                plt.ylim(0,100)
                plt.xlim(0,MAX_BIN)
                plt.legend(['MOVEMENT FAST'],loc='upper right')
                plt.xticks([0,20,40,60,80],[round(d*0.0514 + 0.4,2) for d in [0,20,40,60,80]])
                

                
                plt.ion()
                plt.show()
                plt.pause(0.01)


                datas = datas[fps:,:,:]

def main():
    print('start,waiting for the connection.')
    collect = threading.Thread(target=recvdata)
    collect.setDaemon(True)
    collect.start()
    handle()
    print("all over.")

# if __name__ == "__main__":
main()
        