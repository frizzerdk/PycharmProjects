import pyqtgraph as pg
import pyqtgraph.opengl as gl
from stl import mesh
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import struct


projectileMesh = mesh.Mesh.from_file('Projectile_assem2.STL')
data = np.array(projectileMesh)
data_reshaped = np.reshape(data, (171676, 3, 3))
meshData = gl.MeshData(vertexes=data_reshaped)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLMeshItem')
w.setCameraPosition(distance=500)


g = gl.GLGridItem()
g.scale(20,20,1)
w.addItem(g)


md = gl.MeshData.sphere(rows=4, cols=8)
m4 = gl.GLMeshItem(meshdata=meshData, smooth=True, shader='shaded')
m4.translate(0,10,0)
w.addItem(m4)


with open('logs/FINALMOSTIMPORTANT.TXT', 'r') as filehandle:
    lines = [current_place.rstrip() for current_place in filehandle.readlines()]

splitLines = lines[3500:-1]

win = pg.GraphicsWindow()
win.setWindowTitle('pyqtgraph example: Scrolling Plots')

size = 100
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
# 1) Simplest approach -- update data in the array such that plot appears to scroll
#    In these examples, the array size is fixed.
p_acc = win.addPlot(0,0)
p_acc.disableAutoRange()
p_acc.showGrid(x=True, y=True)
p_acc.setYRange(-2, 2, padding=0)
p_acc.setXRange(0, 100, padding=0)
p_acc.setTitle(title='Acceleration')
p_acc.addLegend()
curve_acc_X = p_acc.plot(data_accX, pen=pg.mkPen('r'), name="X")
curve_acc_Y = p_acc.plot(data_accY, pen=pg.mkPen('g'), name="Y")
curve_acc_Z = p_acc.plot(data_accZ, pen=pg.mkPen('b'), name="Z")


p_lin = win.addPlot(1,0)
p_lin.disableAutoRange()
p_lin.showGrid(x=True, y=True)
p_lin.setYRange(-2, 2, padding=0)
p_lin.setXRange(0, 100, padding=0)
p_lin.setTitle(title='Linear Acceleration')
p_lin.addLegend()
curve_lin_X = p_lin.plot(data_linAccX, pen=pg.mkPen('r'), name="X")
curve_lin_Y = p_lin.plot(data_linAccY, pen=pg.mkPen('g'), name="Y")
curve_lin_Z = p_lin.plot(data_linAccZ, pen=pg.mkPen('b'), name="Z")

p_vel = win.addPlot(2,0)
p_vel.disableAutoRange()
p_vel.showGrid(x=True, y=True)
p_vel.setYRange(-20, 20, padding=0)
p_vel.setXRange(0, 100, padding=0)
p_vel.setTitle(title='Velocity')
p_vel.addLegend()
curve_vel_X = p_vel.plot(data_velX, pen=pg.mkPen('r'), name="X")
curve_vel_Y = p_vel.plot(data_velY, pen=pg.mkPen('g'), name="Y")
curve_vel_Z = p_vel.plot(data_velZ, pen=pg.mkPen('b'), name="Z")
curve_vel_N = p_vel.plot(data_velNorm, pen=pg.mkPen('w'), name="Norm")


p_adc = win.addPlot(0,1)
p_adc.showGrid(x=True, y=True)
p_adc.setTitle(title="ADC's")
p_adc.addLegend()
curve_adc1 = p_adc.plot(data_adc_1,pen=pg.mkPen('r'),name="1")
curve_adc2 = p_adc.plot(data_adc_2,pen=pg.mkPen('g'),name="2")
#curve_adc3 = p_adc.plot(data_adc_3,pen=pg.mkPen('b'),name="3")
curve_adc4 = p_adc.plot(data_adc_4,pen=pg.mkPen('w'),name="4")

p_gyro = win.addPlot(1,1)
p_gyro.disableAutoRange()
p_gyro.showGrid(x=True, y=True)
p_gyro.setYRange(-250, 250, padding=0)
p_gyro.setXRange(0, 100, padding=0)
p_gyro.setTitle(title='Gyroscope')
p_gyro.addLegend()
curve_gyro_X = p_gyro.plot(data_gyroX, pen=pg.mkPen('r'),name="X")
curve_gyro_Y = p_gyro.plot(data_gyroY, pen=pg.mkPen('g'),name="Y")
curve_gyro_Z = p_gyro.plot(data_gyroZ, pen=pg.mkPen('b'),name="Z")


p_mag = win.addPlot(2,1)
p_mag.disableAutoRange()
p_mag.showGrid(x=True, y=True)
p_mag.setYRange(-1000, 1000, padding=0)
p_mag.setXRange(0, 100, padding=0)
p_mag.setTitle(title='Magnetometer')
p_mag.addLegend()
curve_mag_X = p_mag.plot(data_magX, pen=pg.mkPen('r'),name="X")
curve_mag_Y = p_mag.plot(data_magY, pen=pg.mkPen('g'),name="Y")
curve_mag_Z = p_mag.plot(data_magZ, pen=pg.mkPen('b'),name="Z")

"""
p_motor = win.addPlot(0,2)
p_motor.showGrid(x=True, y=True)
p_motor.setTitle(title="Motor positions")
p_motor.addLegend()
curve_motor1 = p_motor.plot(data_motor_1,pen=pg.mkPen('r'),name="1")
curve_motor2 = p_motor.plot(data_motor_2,pen=pg.mkPen('g'),name="2")
curve_motor3 = p_motor.plot(data_motor_3,pen=pg.mkPen('b'),name="3")
curve_motor4 = p_motor.plot(data_motor_4,pen=pg.mkPen('w'),name="4")
"""

p_temp = win.addPlot(0,3)
p_temp.showGrid(x=True, y=True)
p_temp.setYRange(-5, 35, padding=0)
p_temp.setTitle(title='Temperature')
p_temp.addLegend()
curve_temp = p_temp.plot(data_temp,pen=pg.mkPen('w'), name="Celcius")


p_press = win.addPlot(1,3)
p_press.showGrid(x=True, y=True)
p_press.setTitle(title='Pressure')
p_press.addLegend()
curve_press = p_press.plot(data_press,pen=pg.mkPen('w'),name="Bar")
p_count = win.addPlot(2,3)
p_count.showGrid(x=True, y=True)
p_count.setYRange(0, 300, padding=0)
p_count.setTitle(title='Load')
p_count.addLegend()
curve_count = p_count.plot(data_count,pen=pg.mkPen('w'),name="perceived Load")


p_quat = win.addPlot(0,4)
p_quat.showGrid(x=True, y=True)
p_quat.setTitle(title='Quaternion')
p_quat.addLegend()
curve_quat1 = p_quat.plot(data_quat_1,pen=pg.mkPen('r'),name="1")
curve_quat2 = p_quat.plot(data_quat_2,pen=pg.mkPen('g'),name="2")
curve_quat3 = p_quat.plot(data_quat_3,pen=pg.mkPen('b'),name="3")
curve_quat4 = p_quat.plot(data_quat_4,pen=pg.mkPen('w'),name="4")

p_motor = win.addPlot(1,4)
p_motor.showGrid(x=True, y=True)
p_motor.setTitle(title='Motor Degrees')
p_motor.addLegend()
curve_motor1 = p_motor.plot(data_motor_1,pen=pg.mkPen('r'),name="1")
curve_motor2 = p_motor.plot(data_motor_2,pen=pg.mkPen('g'),name="2")
curve_motor3 = p_motor.plot(data_motor_3,pen=pg.mkPen('b'),name="3")
curve_motor4 = p_motor.plot(data_motor_4,pen=pg.mkPen('w'),name="4")

rotation_values_new = np.zeros(4)
rotation_values_old = [0,0,0,0]

count = 0;
def shift():
    data_count[:-1] = data_count[1:]

    data_temp[:-1] = data_temp[1:]
    data_press[:-1] = data_press[1:]

    data_quat_1[:-1] = data_quat_1[1:]
    data_quat_2[:-1] = data_quat_2[1:]
    data_quat_3[:-1] = data_quat_3[1:]
    data_quat_4[:-1] = data_quat_4[1:]

    data_accX[:-1] = data_accX[1:]
    data_accY[:-1] = data_accY[1:]
    data_accZ[:-1] = data_accZ[1:]

    data_gyroX[:-1] = data_gyroX[1:]
    data_gyroY[:-1] = data_gyroY[1:]
    data_gyroZ[:-1] = data_gyroZ[1:]

    data_magX[:-1] = data_magX[1:]
    data_magY[:-1] = data_magY[1:]
    data_magZ[:-1] = data_magZ[1:]

    data_linAccX[:-1] = data_linAccX[1:]
    data_linAccY[:-1] = data_linAccY[1:]
    data_linAccZ[:-1] = data_linAccZ[1:]

    data_velX[:-1] = data_velX[1:]
    data_velY[:-1] = data_velY[1:]
    data_velZ[:-1] = data_velZ[1:]
    data_velNorm[:-1] = data_velNorm[1:]

    data_adc_1[:-1] = data_adc_1[1:]
    data_adc_2[:-1] = data_adc_2[1:]
    data_adc_3[:-1] = data_adc_3[1:]
    data_adc_4[:-1] = data_adc_4[1:]

    data_motor_1[:-1] = data_motor_1[1:]
    data_motor_2[:-1] = data_motor_2[1:]
    data_motor_3[:-1] = data_motor_3[1:]
    data_motor_4[:-1] = data_motor_4[1:]

def populate(data,intData):
    data_count[-1] = intData

    data_temp[-1] = data[1]
    data_press[-1] = data[2]

    data_quat_1[-1] = data[3]
    data_quat_2[-1] = data[4]
    data_quat_3[-1] = data[5]
    data_quat_4[-1] = data[6]

    data_accX[-1] = data[7]
    data_accY[-1] = data[8]
    data_accZ[-1] = data[9]

    data_gyroX[-1] = data[10]
    data_gyroY[-1] = data[11]
    data_gyroZ[-1] = data[12]

    data_magX[-1] = data[13]
    data_magY[-1] = data[14]
    data_magZ[-1] = data[15]


    data_linAccX[-1] = data[16]
    data_linAccY[-1] = data[17]
    data_linAccZ[-1] = data[18]

    data_velX[-1] = data[19]
    data_velY[-1] = data[20]
    data_velZ[-1] = data[21]
    data_velNorm[-1] = data[22]

    data_adc_1[-1] = data[23]
    data_adc_2[-1] = data[24]
    data_adc_3[-1] = data[25]
    data_adc_4[-1] = data[26]

    data_motor_1[-1] = data[27]
    data_motor_2[-1] = data[28]
    data_motor_3[-1] = data[29]
    data_motor_4[-1] = data[30]

def update1():
    global curve_acc_X, curve_acc_Y, curve_acc_Z, curve_gyro_X, curve_gyro_Y, curve_gyro_Z, curve_mag_X, curve_mag_Y, curve_mag_Z, curve_temp, curve_press, curve_lin_X, curve_lin_Y, curve_lin_Z, curve_vel_X, curve_vel_Y, curve_vel_Z, curve_vel_N,a,rotation_values_new, rotation_values_old,count

    raw_data = splitLines[count].split(",")
    revers = [i[::-1] for i in raw_data]
    int_data = int(revers[0], 16)
    float_data = [struct.unpack('f', struct.pack('I', int(i, 16)))[0] for i in revers]

    shift()
    populate(float_data,int_data)

    curve_acc_X.setData(data_accX)
    curve_acc_Y.setData(data_accY)
    curve_acc_Z.setData(data_accZ)

    curve_gyro_X.setData(data_gyroX)
    curve_gyro_Y.setData(data_gyroY)
    curve_gyro_Z.setData(data_gyroZ)

    curve_mag_X.setData(data_magX)
    curve_mag_Y.setData(data_magY)
    curve_mag_Z.setData(data_magZ)

    curve_temp.setData(data_temp)
    curve_press.setData(data_press)
    curve_count.setData(data_count)


    curve_lin_X.setData(data_linAccX)
    curve_lin_Y.setData(data_linAccY)
    curve_lin_Z.setData(data_linAccZ)

    curve_vel_X.setData(data_velX)
    curve_vel_Y.setData(data_velY)
    curve_vel_Z.setData(data_velZ)
    curve_vel_N.setData(data_velNorm)

    curve_adc1.setData(data_adc_1)
    curve_adc2.setData(data_adc_2)
    #curve_adc3.setData(data_adc_3)
    curve_adc4.setData(data_adc_4)


    curve_quat1.setData(data_quat_1)
    curve_quat2.setData(data_quat_2)
    curve_quat3.setData(data_quat_3)
    curve_quat4.setData(data_quat_4)

    curve_motor1.setData(data_motor_1)
    curve_motor2.setData(data_motor_2)
    curve_motor3.setData(data_motor_3)
    curve_motor4.setData(data_motor_4)


    k = np.arccos(float_data[3])
    s = np.sin(k)
    m4.resetTransform();
    m4.rotate(k*57.2958, float_data[4]/s, float_data[5]/s, float_data[6]/s)

    count += 1

# update all plots
def update():
    update1()


timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()