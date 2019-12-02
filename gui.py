import predict
import stk
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3agg import (
    FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure


class MainWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Stock_Predict")

        a, self.Tindex, b = stk.csv_to_df(2330)
        self.Tindex = np.array(self.Tindex)
        self.box1 = Gtk.VBox(False, spacing=6)
        self.add(self.box1)
        self.box2 = Gtk.HBox(spacing=6)
        self.box3 = Gtk.HBox(spacing=6)
        self.box4 = Gtk.HBox(spacing=6)
        self.box1.pack_start(self.box2, False, False, 2)
        self.box1.pack_start(self.box3, True, True, 0)
        self.box1.pack_start(self.box4, False, False, 0)

        self.button1 = Gtk.Button(label="2337 : Macronix International")

        self.button1.connect("clicked", self.on_button1_clicked)
        self.box2.pack_start(self.button1, True, True, 2)

        self.button2 = Gtk.Button(
            label="2330 : Taiwan Semiconductor Manufacturing")

        self.button2.connect("clicked", self.on_button2_clicked)
        self.box2.pack_start(self.button2, True, True, 2)

        self.button3 = Gtk.Button(label="6223 : MPI Corp")

        self.button3.connect("clicked", self.on_button3_clicked)
        self.box2.pack_start(self.button3, True, True, 2)

        self.button4 = Gtk.Button(label="2867 : BonEagle Electric Co Ltd")

        self.button4.connect("clicked", self.on_button4_clicked)
        self.box2.pack_start(self.button4, True, True, 2)

        # f2 block INDEX
        self.f2 = Figure(figsize=(5, 4), dpi=100)
        self.a2 = self.f2.add_subplot(111)
        self.t2 = np.arange(0.0, 255, 1)
        self.s2 = self.Tindex
        self.a2.plot(self.t2, self.s2, zorder=3)
        self.a2.set_title("TAIEX(Taipei)")
        self.a2.set_ylabel("Score")
        self.a2.set_xlabel("Days")
        self.a2.grid()

        # f1 block predict
        self.f1 = Figure(figsize=(5, 4), dpi=100)
        self.a1 = self.f1.add_subplot(111)
        #self.t1 = np.arange(0.0, 3.0, 0.01)
        #self.s1 = np.sin(2 * np.pi * self.t1)
        x, y = predict.fit_company_change(2337)
        #print(x, y)
        predict.plotchange(self, list_all[0], x, y)
        #self.a1.plot(self.t1, self.s1, zorder=3)
        self.a1.grid()

        self.sw1 = Gtk.ScrolledWindow()
        self.sw1.set_border_width(10)
        self.box3.pack_start(self.sw1, True, True, 0)

        self.canvas1 = FigureCanvas(self.f1)  # a Gtk.DrawingArea
        self.canvas1.set_size_request(400, 400)
        self.sw1.add_with_viewport(self.canvas1)

        self.sw2 = Gtk.ScrolledWindow()
        self.sw2.set_border_width(10)
        self.box3.pack_start(self.sw2, True, True, 0)

        self.canvas2 = FigureCanvas(self.f2)  # a Gtk.DrawingArea
        self.canvas2.set_size_request(400, 400)
        self.sw2.add_with_viewport(self.canvas2)
        # A scrolled window border goes outside the scrollbars and viewport
        pass

    def on_button1_clicked(self, data):
        # print("Button1", data,)

        self.f1 = Figure(figsize=(5, 4), dpi=100)
        self.a1 = self.f1.add_subplot(111)
        # self.t = np.arange(0.0, 3.0, 0.01)
        # self.s = np.sin(2 * self.t)
        # self.a1.plot(self.t, self.s)
        x, y = predict.fit_company_change(2337)
        #print(x, y)
        predict.plotchange(self, list_all[0], x, y)
        self.box3.remove(self.sw1)
        self.canvas2 = FigureCanvas(self.f1)  # a Gtk.DrawingArea
        self.canvas2.set_size_request(400, 400)
        self.sw1 = Gtk.ScrolledWindow()
        self.sw1.add_with_viewport(self.canvas2)
        self.sw1.set_border_width(10)
        self.box3.pack_end(self.sw1, True, True, 0)
        win.show_all()
        pass

    def on_button2_clicked(self, data):
        #print("Button2", data,)

        self.f1 = Figure(figsize=(5, 4), dpi=100)
        self.a1 = self.f1.add_subplot(111)

        x, y = predict.fit_company_change(2330)
        #print(x, y)
        predict.plotchange(self, list_all[1], x, y)
        self.box3.remove(self.sw1)
        self.canvas2 = FigureCanvas(self.f1)  # a Gtk.DrawingArea
        self.canvas2.set_size_request(400, 400)
        self.sw1 = Gtk.ScrolledWindow()
        self.sw1.add_with_viewport(self.canvas2)
        self.sw1.set_border_width(10)
        self.box3.pack_end(self.sw1, True, True, 0)
        win.show_all()
        pass

    def on_button3_clicked(self, data):
        #print("Button3", data,)

        self.f1 = Figure(figsize=(5, 4), dpi=100)
        self.a1 = self.f1.add_subplot(111)
        x, y = predict.fit_company_change(6223)
        #print(x, y)
        predict.plotchange(self, list_all[2], x, y)
        self.box3.remove(self.sw1)
        self.canvas2 = FigureCanvas(self.f1)  # a Gtk.DrawingArea
        self.canvas2.set_size_request(400, 400)
        self.sw1 = Gtk.ScrolledWindow()
        self.sw1.add_with_viewport(self.canvas2)
        self.sw1.set_border_width(10)
        self.box3.pack_end(self.sw1, True, True, 0)
        win.show_all()
        pass

    def on_button4_clicked(self, data):
        #print("Button4", data,)

        self.f1 = Figure(figsize=(5, 4), dpi=100)
        self.a1 = self.f1.add_subplot(111)
        x, y = predict.fit_company_change(2867)
        #print(x, y)
        predict.plotchange(self, list_all[3], x, y)
        self.box3.remove(self.sw1)
        self.canvas2 = FigureCanvas(self.f1)  # a Gtk.DrawingArea
        self.canvas2.set_size_request(400, 400)
        self.sw1 = Gtk.ScrolledWindow()
        self.sw1.add_with_viewport(self.canvas2)
        self.sw1.set_border_width(10)
        self.box3.pack_end(self.sw1, True, True, 0)
        win.show_all()
        pass


if __name__ == "__main__":
    list_all = predict.predict_all()
    win = MainWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
