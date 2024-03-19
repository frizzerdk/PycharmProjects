from ui_lib import *
from classes_trade import *
from poe_trade_interface import *


class ItemSearchUi:
    

    def __init__(self, pt):
        self.root = Tk()
        self.root.geometry("1500x1000")
        self.pt = pt
        self.root.mainloop()

    def make_ui(pt):
        root = Tk()
        root.geometry("1500x1000")
        # ttk.Entry(root).grid(sticky='wens')

        '''
        root.filename = filedialog.askopenfilename(initialdir="/mfiles/", title="select config file",
                                                   filetypes=(("items object", ".json"), ("all", ".*")))
        '''

        # frame
        # tempstats = [" one ", " two ", " three "]
        item_frame = LabelFrame(root, text="item", padx=10, pady=10)
        options = pt.stats["result"][0]["entries"]
        options2 = pt.stats["result"][0]["entries"]
        # search bar
        ci = lambda event: check_input(event, options)
        combo_box = ttk.Combobox(item_frame, width=50)
        combo_box.bind('<Configure>', ci)
        combo_box.bind('<KeyRelease>', ci)
        combo_box.grid(row=1, column=0, columnspan=5)
        label_widget1 = Label(item_frame, text="Pseudo stat search")
        label_widget1.grid(row=0, column=0)
        item_frame.grid(row=0, column=0)

        itemframes = []
        # combo_box_temp=[]
        count = 0

        for i in item_class_names:
            itemframes.append(clone_widget(item_frame))
            itemframes[count].configure(text=item_class_names[count])
            itemframes[count].grid(row=count + 1, column=0, columnspan=3)
            combo_box_temp = ttk.Combobox(itemframes[count], width=50)
            combo_box_temp.bind('<Configure>', ci)
            combo_box_temp.bind('<KeyRelease>', ci)
            combo_box_temp.grid(row=1, column=0, columnspan=5)
            print(item_class_names[count])
            count += 1


def open_item():
    item_window = LabelFrame(root, text="item", padx=10, pady=10)
    options = pt.stats["result"][0]["entries"]
    options2 = pt.stats["result"][0]["entries"]
    # search bar
    ci = lambda event: check_input(event, options)
    combo_box = ttk.Combobox(item_frame, width=50)
    combo_box.bind('<Configure>', ci)
    combo_box.bind('<KeyRelease>', ci)
    combo_box.grid(row=1, column=0, columnspan=5)
    label_widget1 = Label(item_frame, text="Pseudo stat search")
    label_widget1.grid(row=0, column=0)
    item_frame.grid(row=0, column=0)

    itemframes[1].destroy()

    root.mainloop()
