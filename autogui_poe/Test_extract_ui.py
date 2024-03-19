import pyautogui as pag
import sys
import win32api
import time
import cv2 as cv
import numpy as np
import pyperclip as pc
import random as ra
from matplotlib import pyplot as plt


def main():
    size = pag.size()
    print("size: " + str(size))
    pos = pag.position()
    print("pos: " + str(pos))
    x_min = 0
    y_min = 0

    grid_region = get_stash_grid()
    st=StashTab(region=grid_region)
    print(st)
    last_file=snap_region(grid_region)
   # cv.namedWindow("2", cv.WINDOW_NORMAL)
    cv.imshow("last", last_file)
    cv.waitKey()
    ran = ra.Random()
    items = [ ["" for r in range(int(st.rows/2))]for c in range(int(st.columns/2))]
    for r in range(int(st.rows/2)):
        for c in range(int(st.columns/2)):
            cell_pos = st.get_cel_pos(r+(ran.random()-0.5)*0.8,c+(ran.random()-0.5)*0.8)
            pag.moveTo(cell_pos[0],cell_pos[1],duration=0.1*(ran.random()*0.4+0.8))
            pag.rightClick()
            pc.copy("empty")
            pag.hotkey('ctrl','alt','c')
            data=pc.paste()
            print(data)
            items[r][c]=data
    print(items)
    

    # for i in range(100):
    #     time.sleep(1)
    #     pos = get_pos_on_window("Path of exile")
    #     print("pos: " + str(pos))

class UIRegion():

    def __init__(self,region,*args,**kwargs):
        self.left=region[0]
        self.top=region[1]
        self.width=region[2]
        self.height=region[3]
        self.right=self.left+self.width
        self.bottom=self.top+self.height

    def __str__(self):
        return  "left: "+str(self.left)+\
                " top: "+str(self.top)+\
                " width: "+str(self.width)+\
                " height: "+str(self.height)+\
                " right: "+str(self.right)+\
                " bottom: "+str(self.bottom)
        # self.top=region[1]
        # self.width=region[2]
        # self.height=region[3]
        # self.right=self.left+self.width
        # self.bottom=self.top+self.height

class StashTab(UIRegion):

    def __init__(self,rows=12,columns=12,*args,**kwargs):
        UIRegion.__init__(self,*args,**kwargs)
        self.rows=rows
        self.columns=columns
        self.cell_height = self.height / self.rows
        self.cell_width = self.width / self.columns

    def get_cel_pos(self,row,column):
        return (self.left+self.cell_width*(column+0.5),self.top+self.cell_height*(row+0.5))



def snap_region(region):
    grid = pag.screenshot("last.png", region=region)
    open_cv_image = np.array(grid)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return  open_cv_image

def get_stash_grid():
    windows = (pag.getWindowsWithTitle("Path of Exile"))
    window = windows[0]
    print("region:" + str(window.left) + " " + str(window.top) + " " + str(window.width ) + " " + str(window.height ))

    margin_x = 8
    margin_y = 31
    grid = {"left": 18 / 1080, "top": 132 / 1080, "width": 630 / 1080, "height": 632 / 1080}
    window_height=window.height- margin_x - margin_y

    region_grid=(window.left + margin_x +grid["left"]*window_height,
                 window.top + 32+grid["top"]*window_height,
                 grid["width"]*window_height,
                 grid["height"]*window_height)

    region = (window.left + margin_x, window.top + 32, window.width - margin_x * 2, window.height - margin_x - margin_y)

    return region_grid#, region_grid,grid

def get_grid_content():

    pc.copy("abc")  # now the clipboard content will be string "abc"
    text = pc.paste()  # text will have the content of clipboard


def get_pos_on_window(name):
    margin_x = 8
    margin_y = 32

    windows = (pag.getWindowsWithTitle(name))
    window = windows[0]
    pos = pag.position()
    print("region:" + str(window.left) + " " + str(window.top) + " " + str(window.width / 2) + " " + str(
        window.height * 0.9))
    # region=(window.left+window.width*0.01, window.top+window.height*0.05, window.width*0.5,window.height*0.9)
    # pag.screenshot("last.png",region=region)
    # find_grid("last.png")
    region = (window.left, window.top + window.height, window.width, window.height)

    return pos.x - window.left - margin_x, pos.y - window.top - margin_y


def get_screen_dim():
    if sys.platform == 'win32':
        # get the lowest x and y coordinate of the monitor setup
        monitors = win32api.EnumDisplayMonitors()
        x_min = min([mon[2][0] for mon in monitors])
        y_min = min([mon[2][1] for mon in monitors])
        # add negative offset due to multi monitor
        print(x_min)


if __name__ == "__main__":
    main()
