""" Denoised First & Then Deblurred """

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import pandas as pd
import math

""" Reading Data """
Data = pd.read_csv('DATA.csv')

""" Declaring Impulse response h[n] as an array h"""
h = np.array([1/16 , 4/16 , 6/16 , 4/16 , 1/16])

""" Recieved Signal y[n]"""
Received_Signal_Y = Data['y'].tolist()

"""True Signal x[n]"""
True_Signal_X = Data['x'].tolist()

""" X axis for Graph Plotting """
OPP = [(2*math.pi*i)/193 for i in range(193)]
XAXIS = np.array(OPP)

"""Mean Squared Error of Recieved Signal and Original Signal"""
MSE1 = np.subtract(True_Signal_X,Received_Signal_Y)
M1summ=0
for i in MSE1:
    M1summ = M1summ + i*i



"""
Denoising The Recieved Signal
Method Used: Local Mean Denoising
Explanation: Replacing y[n] = ( y[n-2]+y[n-1]+y[n]+y[n+1]+y[n+2] )/5.
             First & last values are taken care off by appending them 2 times at beginning and end of array respectively.
"""

Last_value = Received_Signal_Y[len(Received_Signal_Y)-1]

Received_Signal_Y.append(Last_value)
Received_Signal_Y.append(Last_value)

First_value = Received_Signal_Y[0]

Received_Signal_Y.insert(0,First_value)
Received_Signal_Y.insert(0,First_value)


Y = [] # Array Corresponding to Denoised Signal
Nb=[]

# Denoising
for t in range(2,len(Received_Signal_Y)-2):
    sumi = (Received_Signal_Y[t]+Received_Signal_Y[t-1]+Received_Signal_Y[t+1]+Received_Signal_Y[t-2]+Received_Signal_Y[t+2])/5
    Y.append(sumi)
    Nb.append(sumi)

"""Mean Squared Error of Recieved Signal+ Denoised and Original Signal"""
MSE2 = np.subtract(True_Signal_X,Y)
M2summ=0
for i in MSE2:
    M2summ = M2summ + i*i



"""
Deblurring the Denoised Signal
"""

"""
 Computing DTFT(Discrete Time Fourier Transform) of Denoised signal 
 w(omega) = 2*pi*k/193
 Here k is varied from (0,193)
"""

yw = [] # List for storing DTFT
YW=[] # List for Storing absolute values of DTFT

for k in range(0,193):
    summ = 0
    for n in range(0,193):
        x = math.cos((2*math.pi*k*n)/193)
        y = -math.sin((2*math.pi*k*n)/193)
        z = complex(x,y)
        summ = summ +  Y[n]*z
    yw.append(summ)
    YW.append(abs(summ))

"""
 Computing DTFT(Discrete Time Fourier Transform) of Impulse Response h[n] 
 w(omega) = 2*pi*k/193
 Here k is varied from (0,193)
"""

hw = [] # List for storing DTFT
HW=[] # List for Storing absolut values of DTFT of h[n]

for k in range(0,193):
    pi=[]
    for n in range(-2,3):
        x = math.cos((2*math.pi*k*n)/193)
        y = -math.sin((2*math.pi*k*n)/193)
        z = complex(x,y)
        pi.append(z)
    expon = np.array(pi)
    PO = np.multiply(h,expon)
    hw.append(np.sum(PO))

for i in range(193):
    HW.append(abs(hw[i]))

"""
 Computing DTFT(Discrete Time Fourier Transform) of x[n]
 DTFT of x[n] = DTFT of y[n]/DTFT of h[n] 
 w(omega) = 2*pi*k/193
 Here k is varied from (0,193)
"""


xw = [] # List for Storing DTFT of x[n]
XW = [] # List for Storing Absolute values of DTFT of x[n]

for i in range(0,193):
    """ A cap of 0.5 is applied as value of hw can be infinitesimally small and can lead to very large variations"""
    if hw[i].real<0.35:
        xw.append(yw[i]/0.35)
    else:
        xw.append(yw[i]/hw[i])

for i in range(193):
    XW.append(abs(xw[i]))

"""
 Computing Inverse Fourier Transform of DTFT of x[n]
"""

finaloutput = []
for n in range(0,193):
    summ = 0
    for k in range(0,193):
        x = math.cos((2*math.pi*k*n)/193)
        y = math.sin((2*math.pi*k*n)/193)
        z = complex(x,y)
        summ = summ + xw[k]*z
    finaloutput.append(abs(summ/193))
print("Final Output X1[n] is \n")
for H1 in finaloutput:
    print(round(H1,3))

plt.figure(figsize=(11,11))
plt.plot(XAXIS,np.array(YW),label='DTFT Of Y[n]')
plt.xlabel('Ω-Omega',fontsize=13)
plt.ylabel('|Y(e^jΩ)|',fontsize = 13)
plt.legend()
plt.show()

plt.figure(figsize=(11,11))
plt.plot(XAXIS,np.array(HW),label='DTFT Of h[n]')
plt.xlabel('Ω-Omega',fontsize = 13)
plt.ylabel('|H(e^jΩ)|',fontsize = 13)
plt.legend()
plt.show()

plt.figure(figsize=(11,11))
plt.plot(XAXIS,np.array(XW),label='DTFT Of x[n]')
plt.xlabel('Ω-Omega',fontsize = 13)
plt.ylabel('|X(e^jΩ)|',fontsize =13)
plt.legend()
plt.show()

xaxis=[i for i in range(0,193)]

plt.figure(figsize=(11,11))
plt.plot(np.array(xaxis),np.array(finaloutput),label='Recovered Signal x1[n]')
plt.xlabel('n',fontsize=13)
plt.ylabel('x1[n]',fontsize = 13)
plt.legend()
plt.show()

#Comparing with Original Signal
plt.figure(figsize=(11,11))
plt.plot(np.array(xaxis),np.array(True_Signal_X),label='Original Signal x[n]')
plt.plot(np.array(xaxis),np.array(finaloutput),label='Recovered Signal x1[n]')
plt.xlabel('n',fontsize = 13)
plt.legend()
plt.show()

print("Mean Squared Error between Recieved Signal+Denoised and Original Signal",round(M2summ/193,3))

print("Mean Squared Error between Recieved Signal and Original Signal",round(M1summ/193,3))

""" Calculating Mean Squared Error for Final Output"""
MSE=np.subtract(True_Signal_X,finaloutput)
Msum=0
for i in MSE:
    Msum = Msum + i*i
print(Msum)
print("Mean Squared Error between x[n] and x1[n]",round(Msum/193,3)) 