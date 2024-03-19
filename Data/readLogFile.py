import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import struct


lines = []
splitLines = []


with open('Log', 'r') as filehandle:
    lines = [current_place.rstrip() for current_place in filehandle.readlines()]

splitLines = lines[3650:3700]

size = len(splitLines)
data_count = np.zeros(size)
data_temp = np.zeros(size)
data_press = np.zeros(size)
data_quat_1 = np.zeros(size)
data_quat_2 = np.zeros(size)
data_quat_3 = np.zeros(size)
data_quat_4 = np.zeros(size)
data_accX = np.zeros(size)
data_accY = np.zeros(size)
data_accZ = np.zeros(size)
data_gyroX = np.zeros(size)
data_gyroY = np.zeros(size)
data_gyroZ = np.zeros(size)
data_magX = np.zeros(size)
data_magY = np.zeros(size)
data_magZ = np.zeros(size)
data_linAccX = np.zeros(size)
data_linAccY = np.zeros(size)
data_linAccZ = np.zeros(size)
data_velX = np.zeros(size)
data_velY = np.zeros(size)
data_velZ = np.zeros(size)
data_velNorm = np.zeros(size)
data_adc_1 = np.zeros(size)
data_adc_2 = np.zeros(size)
data_adc_3 = np.zeros(size)
data_adc_4 = np.zeros(size)
data_motor_1 = np.zeros(size)
data_motor_2 = np.zeros(size)
data_motor_3 = np.zeros(size)
data_motor_4 = np.zeros(size)

for l in range(len(splitLines)):
    raw_data = splitLines[l].split(",")
    revers = [i[::-1] for i in raw_data]
    int_data = int(revers[0], 16)
    float_data = [struct.unpack('f', struct.pack('I', int(i, 16)))[0] for i in revers]

    data_count[l] = int_data
    data_temp[l] = float_data[1]
    data_press[l] = float_data[2]
    data_quat_1[l] = float_data[3]
    data_quat_2[l] = float_data[4]
    data_quat_3[l] = float_data[5]
    data_quat_4[l] = float_data[6]
    data_accX[l] = float_data[7]
    data_accY[l] = float_data[8]
    data_accZ[l] = float_data[9]
    data_gyroX[l] = float_data[10]
    data_gyroY[l] = float_data[11]
    data_gyroZ[l] = float_data[12]
    data_magX[l] = float_data[13]
    data_magY[l] = float_data[14]
    data_magZ[l] = float_data[15]
    data_linAccX[l] = float_data[16]
    data_linAccY[l] = float_data[17]
    data_linAccZ[l] = float_data[18]
    data_velX[l] = float_data[19]
    data_velY[l] = float_data[20]
    data_velZ[l] = float_data[21]
    data_velNorm[l] = float_data[22]
    data_adc_1[l] = float_data[23]
    data_adc_2[l] = float_data[24]
    data_adc_3[l] = float_data[25]
    data_adc_4[l] = float_data[26]

    data_motor_1[l] = float_data[27]
    data_motor_2[l] = float_data[28]
    data_motor_3[l] = float_data[29]
    data_motor_4[l] = float_data[30]



plt.figure(1)
plt.title('Load')
plt.xlabel('Time')
plt.ylabel('N/A')
plt.plot(data_count,'b')

plt.figure(2)
plt.title('Temperature')
plt.xlabel('Time')
plt.ylabel('Celcius')
plt.plot(data_temp,'b')

plt.figure(3)
plt.title('Pressure')
plt.xlabel('Time')
plt.ylabel('Bar')
plt.plot(data_press,'b')

plt.figure(4)
plt.title('Acceleration')
plt.xlabel('Time')
plt.ylabel('G')
plt.plot(data_accX,'r')
plt.plot(data_accY,'g')
plt.plot(data_accZ,'b')
plt.axvline(x=10)
#plt.plot([],'black')

plt.figure(5)
plt.title('Gyro')
plt.xlabel('Time')
plt.ylabel('degrees/s')
plt.plot(data_gyroX,'r')
plt.plot(data_gyroY,'g')
plt.plot(data_gyroZ,'b')
plt.axvline(x=10)

plt.figure(6)
plt.title('Magnetometer')
plt.xlabel('Time')
plt.ylabel('Gauss')
plt.plot(data_magX,'r')
plt.plot(data_magY,'g')
plt.plot(data_magZ,'b')
plt.axvline(x=10)

plt.figure(7)
plt.title('Linear Acceleration')
plt.xlabel('Time')
plt.ylabel('G')
plt.plot(data_linAccX,'r')
plt.plot(data_linAccY,'g')
plt.plot(data_linAccZ,'b')

plt.figure(8)
plt.title('Velocity')
plt.xlabel('Time')
plt.ylabel('m/s')
plt.plot(data_velX,'r')
plt.plot(data_velY,'g')
plt.plot(data_velZ,'b')
plt.plot(data_velNorm,'p')

plt.figure(9)
plt.title('ADC')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.plot(data_adc_1,'r')
plt.plot(data_adc_2,'g')
plt.plot(data_adc_3,'b')
plt.plot(data_adc_4,'p')


plt.figure(10)
plt.title('Motor')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.plot(data_motor_1,'r')
plt.plot(data_motor_2,'g')
plt.plot(data_motor_3,'b')
plt.plot(data_motor_4,'p')

plt.figure(11)
plt.title('Quaternions')
plt.xlabel('Time')
plt.ylabel('Quat')
plt.plot(data_quat_1,'r')
plt.plot(data_quat_2,'g')
plt.plot(data_quat_3,'b')
plt.plot(data_quat_4,'p')



plt.show()