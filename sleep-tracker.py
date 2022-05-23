import os
from enum import Enum
import socket
from time import ctime
import threading
import time
import struct
from cv2 import BFMatcher_create
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

class PhaseCorrection(object):
    def __init__(self, bbframes, bin_ref):
        self.bin_ref = bin_ref
        self.bin_angle = np.angle(bbframes[:,bin_ref]).mean()

    def filter(self, frame):
        phase_correction = self.bin_angle - np.angle(frame[self.bin_ref])
        return frame * np.exp(1j*phase_correction)

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
        s.bind(("",13456))  # ip 137.0.0.1
        s.listen(1)
        print("listening")
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

def calcuRPM(x,fastper,lastpeak):
    x_sum = np.sum(np.abs(x),axis=0)
    newpeak = findpeak(x_sum)
    peak = newpeak
    if abs(newpeak - lastpeak) > 5:
        if fastper < 10:
            peak = lastpeak
    breathe = x[:,peak]
    breathe = signal.filtfilt(brpm,arpm,breathe,axis=0)
    breathe_fft = abs(np.fft.fft(breathe))
    breathe_fft_max = np.max(breathe_fft[:int(len(breathe_fft)/2)])
    breathe_fft_max_index = (np.where(breathe_fft[:int(len(breathe_fft)/2)] == breathe_fft_max))[0][0]
    rpm = breathe_fft_max_index * fps/FPS * 60
    return rpm,peak,newpeak



def findpeak(in_arr):
    peaks = 0
    peakmax = 0
    for i in range(1,len(in_arr) - 1):
        if in_arr[i - 1] < in_arr[i] and in_arr[i] < in_arr[i + 1] and in_arr[i] > peakmax:
            peaks = i
            peakmax  = in_arr[i]
    
    return peaks



    



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
    

    distancelist = np.ones((120,))*0.4
    movehistory = np.zeros((120,))
 
    
    
    sleeplist = np.ones((120,))*3
    breathlist = np.zeros((120,))
    


    NoMovement = 1
    Movement = 0
    Movementtracking = 0
    sleeping = 0
    state = 3

    MN = 0
    NMN = 0



    exists = 0
    exist = 0
    noexist = 0


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

                pc = PhaseCorrection(x,1)
                for i in range(FPS):
                    x[i,:] = pc.filter(x[i,:])

                ### movement
              
                iq = x
                _,movelist,moveper = calcumove(iq,-47,10)

  
                ###calcu rpm 
                rpm,peak,newpeak= calcuRPM(iq,moveper,lastpeak)

                ###exist
                movelistsum = np.sum(movelist)
                if  movelistsum > 5:
                    exist += 1
                    noexist = 0
                else:
                    exist = 0
                    noexist += 1
                if exist > 3 and exists == 0:
                    exists = 1
                elif noexist > 10 and exists == 1:
                    exists = 0
                    
                distance = (peak) * 0.0514 + 0.4
                    
                    
                if exists == 1:
                    
                    if moveper > 15 :
                        MN += 1
                        NMN = 0
                    elif moveper <= 10  :
                        MN  = 0
                        NMN += 1


                    if NMN > 10:

                        state = 1
                        sleeping = 1
                        if NMN > 25:
                            state = 0
                    elif MN > 10:
                        sleeping = 0
                        state = 1

                    elif sleeping == 1 :
                        if MN > 10 :

                            sleeping = 0
                            state = 2  
                    else :
                        state = 2
                            
                else:
                    distance = 0
                    state = 3
            
                r1 = 'MN:{};NMN:{:.2f};moveper:{:.2f};sleeping:{:.2f}'.format(MN,NMN,moveper,sleeping)   
                print(r1)

                distancelist = np.append(distancelist,distance)
                distancelist = distancelist[1:]
                movehistory = np.append(movehistory,moveper)
                movehistory = movehistory[1:]
          
                sleeplist = np.append(sleeplist,state)
                sleeplist = sleeplist[1:]
                
                
                if 6 < rpm < 30:
                    breathlist = np.append(breathlist,rpm)
                    breathlist = breathlist[1:]
                
               



                t2 = time.time()
                result = 'State:{};Distance:{:.2f};Movement:{:.2f};RPM:{},Time:{:.3f}'.format(state,distance,moveper,rpm,t2-t1)
                print(result)
                
                showrpm = 0
                title = 'Waiting...'
                if state == 0:
                    title = 'Deep-Sleep'
                    showrpm = breathlist[-1]
                elif state == 1:
                    title = 'Light-Sleep'
                    showrpm = breathlist[-1]
                elif state == 2:
                    title = 'Awake'
                    showrpm = 0
                    
                

                fig = plt.figure('Sleep-Tracker',figsize=(8,9))
                plt.clf()
                fig.suptitle(title)
                plt.subplots_adjust(left=0.15, bottom=0.050, right=0.925, top=0.950,
                                    wspace=0.145, hspace=0.290)
                sns.set_style('darkgrid')
                sns.axes_style()

                plt.subplot(411)
                plt.plot(sleeplist)
                # plt.ylim(0.39,5.0)
                sleepstate = ['Deep-Sleep','Light-Sleep','Awake','Waiting']
                plt.legend(['Sleep-Traccker:{:.2f}'.format(sleeplist[-1])],loc='upper right')
                # plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])
                plt.xticks([0,20,40,60,80,100,120])
                plt.yticks([0,1,2,3],sleepstate)
                
            
                plt.subplot(413)
                plt.plot(distancelist)
                plt.ylim(0.39,5.0)
                plt.legend(['DISTANCE:{:.2f}'.format(distancelist[-1])],loc='upper right')
                # plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])
                plt.xticks([0,20,40,60,80,100,120])


                plt.subplot(412)
                plt.plot(movehistory)
                plt.ylim(0,100)
                # plt.title('MOVEMENT HISTORY')
                plt.legend(['Slow-Movement:{:.2f}'.format(movehistory[-1])],loc='upper right')
                plt.xticks([0,20,40,60,80,100,120])
                # plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])

                plt.subplot(414)
                plt.plot(breathlist)
                plt.ylim(8,30)
                # plt.legend(['RPM:{:.1f}'.format(showrpm)],loc='upper right')
                plt.legend(['RPM:{:.2f}'.format(showrpm)],loc='upper right')
                # plt.xticks([0,20,40,60,80,100,120],[120,100,80,60,40,20,0])
                plt.xticks([0,20,40,60,80,100,120])

                
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
        