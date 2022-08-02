""" Deblurred First & Then Denoised """

import numpy as np
import matplotlib.pyplot as plt
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
Deblurring the Signal y[n]
"""

"""
 Computing DTFT(Discrete Time Fourier Transform) of the signal y[n] 
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
        summ = summ +  Received_Signal_Y[n]*z
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
    expon1 = np.array(pi)
    PO = np.multiply(h,expon1)
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
    if hw[i].real<0.35:
        xw.append(yw[i]/0.35)
    else:
        xw.append(yw[i]/hw[i])

for i in range(193):
    XW.append(abs(xw[i]))

"""
 Computing Inverse Fourier Transform of DTFT of x[n]
"""


finaloutput1 = []
for n in range(0,193):
    summ = 0
    for k in range(0,193):
        x = math.cos((2*math.pi*k*n)/193)
        y = math.sin((2*math.pi*k*n)/193)
        z = complex(x,y)
        summ = summ + xw[k]*z
    finaloutput1.append(abs(summ/193))

"""
Denoising The Recieved Signal
Method Used: Local Mean Denoising
Explanation: Replacing y[n] = ( y[n-2]+y[n-1]+y[n]+y[n+1]+y[n+2] )/5.
             First & last values are taken care off by appending them 2 times at beginning and end of array respectively.
"""

"""Mean Squared Error of Recieved Signal and Original Signal"""
MSE2=np.subtract(True_Signal_X,finaloutput1)
M2summ = 0
for i in MSE2:
    M2summ = M2summ + i*i


First_value = finaloutput1[0]

finaloutput1.insert(0,First_value)
finaloutput1.insert(0,First_value)

Last_value = finaloutput1[len(finaloutput1)-1]

finaloutput1.append(Last_value )
finaloutput1.append(Last_value )

finaloutput = []
for t in range(2,len(finaloutput1)-2):
    sumi = (finaloutput1[t]+finaloutput1[t-1]+finaloutput1[t+1]+finaloutput1[t-2]+finaloutput1[t+2])/5
    finaloutput.append(abs(sumi))

print("Final output X2[n] \n")
for H1 in finaloutput:
    print(round(H1,3))

plt.figure(figsize=(11,11))
plt.plot(XAXIS,np.array(YW),label='DTFT Of Y[n]')
plt.xlabel('Ω-Omega',fontsize = 13)
plt.ylabel('|Y(e^jΩ)|',fontsize = 13)
plt.legend()
plt.show()

plt.figure(figsize=(11,11))
plt.plot(XAXIS,np.array(HW),label='DTFT Of h[n]')
plt.xlabel('Ω-Omega',fontsize = 13)
plt.ylabel('|H(e^jΩ)|',fontsize =13)
plt.legend()
plt.show()

plt.figure(figsize=(11,11))
plt.plot(XAXIS,np.array(XW),label='DTFT Of x[n]')
plt.xlabel('Ω-Omega',fontsize =13)
plt.ylabel('|X(e^jΩ)|',fontsize =13)
plt.legend()
plt.show()

xaxis=[i for i in range(0,193)]

plt.figure(figsize=(11,11))
plt.plot(np.array(xaxis),np.array(finaloutput),label='Recovered Signal x2[n]')
plt.xlabel('n',fontsize =13)
plt.ylabel('x2[n]',fontsize =13)
plt.legend()
plt.show()

#Comparing with Original Signal
plt.figure(figsize=(11,11))
plt.plot(np.array(xaxis),np.array(True_Signal_X),label='Original Signal x[n]')
plt.plot(np.array(xaxis),np.array(finaloutput),label='Recovered Signal x2[n]')
plt.xlabel('n',fontsize =13)
plt.legend()
plt.show()

print("Mean Squared Error between Recieved Signal and Original Signal",round(M1summ/193,3))

print("Mean Squared Error between Recieved Signal + Deblurred and Original Signal",round(M2summ/193,3))

""" Mean Squared Error of FinalOutput and Original Signal """
MSE=np.subtract(True_Signal_X,finaloutput)
Msum=0
for i in MSE:
    Msum = Msum + i*i
print("Mean Squared Error between x[n] and x2[n] ",round(Msum/193,3)) 