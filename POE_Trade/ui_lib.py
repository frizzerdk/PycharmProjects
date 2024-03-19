from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import tkinter as tk
import tkinter.ttk as ttk
import json

'''
class Combobox(ttk.Combobox):

    def _tk(self, cls, parent):
        obj = cls(parent)
        obj.destroy()
        if cls is tk.Toplevel:
            obj._w = self.tk.call('ttk::combobox::PopdownWindow', self)
        else:
            obj._w = '{}.{}'.format(parent._w, 'f.l')
        return obj

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.popdown = self._tk(tk.Toplevel, parent)
        self.listbox = self._tk(tk.Listbox, self.popdown)

        self.bind("<KeyPress>", self.on_keypress, '+')
        self.listbox.bind("<Up>", self.on_keypress)

    def on_keypress(self, event):
        if event.widget == self:
            state = self.popdown.state()

            if state == 'withdrawn' \
                    and event.keysym not in ['BackSpace', 'Up']:
                self.event_generate('<Button-1>')
               # self.after(0, self.focus_set)

            if event.keysym == 'Down':
                print("nothing")
               # self.after(0, self.listbox.focus_set)
               # self.after(0, self.focus_set)

        else:  # self.listbox
            curselection = self.listbox.curselection()

            if event.keysym == 'Up' and curselection[0] == 0:
                self.popdown.withdraw()

'''


def clone_widget(widget, master=None):
    """
    Create a cloned version o a widget

    Parameters
    ----------
    widget : tkinter widget
        tkinter widget that shall be cloned.
    master : tkinter widget, optional
        Master widget onto which cloned widget shall be placed. If None, same master of input widget will be used. The
        default is None.

    Returns
    -------
    cloned : tkinter widget
        Clone of input widget onto master widget.

    """
    # Get main info
    parent = master if master else widget.master
    cls = widget.__class__

    # Clone the widget configuration
    cfg = {key: widget.cget(key) for key in widget.configure()}
    cloned = cls(parent, **cfg)

    # Clone the widget's children
    for child in widget.winfo_children():
        child_cloned = clone_widget(child, master=cloned)
        if child.grid_info():
            grid_info = {k: v for k, v in child.grid_info().items() if k not in {'in'}}
            child_cloned.grid(**grid_info)
        elif child.place_info():
            place_info = {k: v for k, v in child.place_info().items() if k not in {'in'}}
            child_cloned.place(**place_info)
        else:
            pack_info = {k: v for k, v in child.pack_info().items() if k not in {'in'}}
            child_cloned.pack(**pack_info)

    return cloned


def mlamda(x, value):
    it = json.dumps(x["text"]).lower()
    iv = value.lower()
    print(it)
    print(iv)
    val = iv in it
    print(val)
    return val


def check_input(event, data):
    value = event.widget.get()
    box = event.widget
    if value == '':
        box['values'] = data
    else:
        box['values'] = list(filter(lambda x: mlamda(x, value), data))
        print(list(filter(lambda x: mlamda(x, value), data)))
    '''
    else:
        data = []
        for item in data:
            if value.lower() in json.dumps(item).lower():
                data.append(item)

        box['values'] = data
'''
