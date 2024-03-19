import pyautogui as pag
import sys
import win32api
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():

    size = pag.size()
    print("size: "+str(size))
    pos = pag.position()
    print("pos: "+str(pos))
    x_min=0
    y_min=0
    if sys.platform == 'win32':
        # get the lowest x and y coordinate of the monitor setup
        monitors = win32api.EnumDisplayMonitors()
        x_min = min([mon[2][0] for mon in monitors])
        y_min = min([mon[2][1] for mon in monitors])
        # add negative offset due to multi monitor
        print(x_min)



    windows=(pag.getWindowsWithTitle("Path of Exile"))
    window=windows[0]
    print("region:" + str(window.left)+" "+ str(window.top)+" "+ str(window.width/2)+" "+str(window.height*0.9))
    # region=(window.left+window.width*0.01, window.top+window.height*0.05, window.width*0.5,window.height*0.9)
    # pag.screenshot("last.png",region=region)
    # find_grid("last.png")

    region=(window.left+window.width*0.01+window.width*0.5, window.top+window.height*0.05, window.width*0.5,window.height*0.9)
    pag.screenshot("last.png",region=region)
    find_grid("last.png")
    ex_pos=pag.locateAllOnScreen('ex.png',confidence=0.4)
    for p in ex_pos:
        print(p)
    ex_cen = pag.center(ex_pos[0])
    pag.moveTo(ex_cen.x,ex_cen.y)
    print("ex at"+str(ex_pos))
    #pag.click('ex.png')
    print(str(window))
    for i in range(100):
        time.sleep(1)
        pos = pag.position()
        print("pos: " + str(pos))

def find_grid(file):

    images=[]
    titles=[]
    def imshow(name,img):
        images.append(img)
        titles.append(name)
        cv2.imshow(name,img)

    image = cv2.imread(file)
    #cv2.imshow("Image", image)
    #imshow("Image", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    #imshow("gray", gray)

    ksize=(0,0)
    sigmaX=1
    blur = cv2.GaussianBlur(gray, ksize, sigmaX,sigmaX)
    blur = np.copy(gray)
    #cv2.imshow("blur", blur)
    #imshow("blur", blur)

    bs=15
    c=-0
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, c)
    #cv2.
    imshow("thresh", thresh)

    hor_kernel = np.ones((1,100),np.uint8)
    it=4
    hor_eroded = cv2.erode(thresh,hor_kernel,iterations=it)
    imshow("hor_eroded",hor_eroded)
    hor_dialated = cv2.dilate(hor_eroded,hor_kernel,iterations=it+1)
    imshow("hor_dilated",hor_dialated)

    #def remove_largest_cont(con)

    ver_kernel = np.ones((50,1),np.uint8)
    it=4
    ver_eroded = cv2.erode(thresh,ver_kernel,iterations=it)
    imshow("ver_eroded",ver_eroded)
    ver_dialated = cv2.dilate(ver_eroded,ver_kernel,iterations=it+1)
    imshow("ver_dilated",ver_dialated)

    combined=ver_dialated +hor_dialated
    imshow("combined",combined)

    #lines = cv2.erode()


    contours,_ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
                image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    contours1=np.copy(image)
    imshow("conturs", contours1)
    cv2.waitKey(1000)


    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)
    imshow("mask", mask)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]
    imshow("New image", out)

    blur = cv2.GaussianBlur(out, (5, 5), 0)
    cv2.imshow("blur1", blur)

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cv2.imshow("thresh1", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000 / 2:
            cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    #imshow("Final Image", image)

    n=len(images)
    for i in range(n):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
    plt.show()

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



main()
