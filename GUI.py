from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2

from main import *

class GUI():

    def __init__(self):

        # Set Tkinter Window  
        self.fenster = Tk()
        self.fenster.title("Summenfeld berechnung")
        self.fenster.configure(background='#49A')

        # Set Variables
        self.live = BooleanVar()
        self.homography = BooleanVar()
        self.threshold = StringVar()
        self.threshold.set("50")
        self.maxLineGap = StringVar()
        self.maxLineGap.set("100")
        self.minLineLength = StringVar()
        self.minLineLength.set("200")

        self.contours = BooleanVar()
        self.vectorised = BooleanVar()
        self.vectorised.set(True)
        self.monitor = StringVar()
        self.monitor.set("1")
        self.kamera = StringVar()
        self.kamera.set("0")
        self.processes = StringVar()
        self.processes.set("4")
        self.path_to_non_live_img = StringVar()
        self.density = StringVar()
        self.density.set("1")
        self.linewidth = StringVar()
        self.linewidth.set("1")
        self.charge_circle_radius = StringVar()
        self.charge_circle_radius.set("0.05")

        # Homografie
        self.centers = None
        self.H = None
        self.window_name = None
        self.beamer_img = None

        # Set Buttons
        self.exit_label = Label(self.fenster, text= "Der Beenden Button schliesst das Programm.")
        self.exit_button = Button(self.fenster, text="Beenden", command=self.exit)

        # Set Checkbutton
        self.live_label = Label(self.fenster, text= "Soll ein Live Bild aufgenommen werden?")
        self.live_button = Checkbutton(self.fenster, text="live", variable=self.live)
        self.do_homography_label = Label(self.fenster, text= "Soll die Homografie angewandt werden?")
        self.do_homography = Checkbutton(self.fenster, text="Homografie anwenden", variable=self.homography)
        self.kalibrierung_label = Label(self.fenster, text= "Führe Kalibrierung aus")
        self.kalibrierung = Button(text='Kalibriere', command=self.calibrate)
        self.path_to_non_live_img_label = Label(self.fenster, text= "Bild auswählen")
        self.path_to_non_live_img_button = Button(text='Datei auswählen (png oder jpg)', command=self.callback)
        #TODO Überschrift für Berechnung
        self.vectorised_button = Checkbutton(self.fenster, text="Summenfeldberechnung mit Vektorisierung", variable=self.vectorised)
        self.contours_button = Checkbutton(self.fenster, text="Summenfeld der Konturen", variable=self.contours)
        self.snapshot_button = Button(text='Snapshot', command=self.snapshot)
        self.start_button = Button(text='Start', command=self.start)

        # Set Entry
        self.monitor_label = Label(self.fenster, text= "Welcher Monitor soll verwendet werden:")
        self.monitor_entry = Entry(self.fenster, textvariable=self.monitor)
        self.kamera_label = Label(self.fenster, text= "Welche Kamera soll verwendet werden:")
        self.kamera_entry = Entry(self.fenster, textvariable=self.kamera)
        self.processes_label = Label(self.fenster, text= "Wie viele Prozesse")
        self.processes_entry = Entry(self.fenster, textvariable=self.processes)
        self.path_to_non_live_img_entry = Entry(self.fenster, textvariable=self.path_to_non_live_img)
        self.density_label = Label(self.fenster, text= "Pfeildichte anpassen")
        self.density_entry = Entry(self.fenster, textvariable=self.density)
        self.linewidth_label = Label(self.fenster, text= "Pfeildicke anpassen")
        self.linewidth_entry = Entry(self.fenster, textvariable=self.linewidth)
        self.charge_circle_radius_label = Label(self.fenster, text= "Radius der Ladungen anpassen")
        self.charge_circle_radius_entry = Entry(self.fenster, textvariable=self.charge_circle_radius)
        self.threshold_label = Label(self.fenster, text= "Threshold")
        self.threshold_entry = Entry(self.fenster, textvariable=self.threshold)
        self.maxLineGap_label = Label(self.fenster, text= "max Line Gap")
        self.maxLineGap_entry = Entry(self.fenster, textvariable=self.maxLineGap)
        self.minLineLength_label = Label(self.fenster, text= "min Line Length")
        self.minLineLength_entry = Entry(self.fenster, textvariable=self.minLineLength)

        # Label
        self.calibrate_label = Label(self.fenster, text= "Kalibrierung")
        self.main_label = Label(self.fenster, text= "Main")
        self.design_label = Label(self.fenster, text= "Darstellung der Feldlinien anpassen:")
        self.homografie_label = Label(self.fenster, text= "Homografie Parameter anpassen:")


        # Label und Buttons erstellen.
        row = 0
        self.calibrate_label.grid(row = row, column=0)

        row += 1
        self.live_label.grid(row=row, column=0)
        self.live_button.grid(row=row, column=1)
        self.path_to_non_live_img_entry.grid(row=row, column=2)
        self.path_to_non_live_img_button.grid(row=row, column=3)

        row +=1
        self.kamera_label.grid(row=row, column=0)
        self.kamera_entry.grid(row=row, column=1)

        row +=1
        self.monitor_label.grid(row=row, column=0)
        self.monitor_entry.grid(row=row, column=1)

        row +=1
        self.do_homography_label.grid(row=row, column=0)
        self.do_homography.grid(row=row, column=1)

        row +=1
        self.homografie_label.grid(row = row, column=0)

        row +=1
        #anpassung bei Homografie
        self.threshold_label.grid(row=row, column=0)
        self.threshold_entry.grid(row=row, column=1)
    
        row +=1
        self.maxLineGap_label.grid(row=row, column=0)
        self.maxLineGap_entry.grid(row=row, column=1)

        row +=1
        self.minLineLength_label.grid(row=row, column=0)
        self.minLineLength_entry.grid(row=row, column=1)

        row +=1
        #self.kalibrierung_label.grid(row=row, column=0)
        self.kalibrierung.grid(row=row, column=1)

        row += 1
        self.main_label.grid(row = row, column=0)

        row +=1
        self.vectorised_button.grid(row=row, column=0)
        self.contours_button.grid(row=row, column=1)

        row +=1
        self.processes_label.grid(row=row, column=0)
        self.processes_entry.grid(row=row, column=1)
        
        row += 1
        self.design_label.grid(row = row, column=0)

        row += 1
        #optische anpassungen am endbild
        self.density_label.grid(row=row, column=0)
        self.density_entry.grid(row=row, column=1)

        row += 1
        self.linewidth_label.grid(row=row, column=0)
        self.linewidth_entry.grid(row=row, column=1)

        row += 1
        self.charge_circle_radius_label.grid(row=row, column=0)
        self.charge_circle_radius_entry.grid(row=row, column=1)

        row += 1
        self.snapshot_button.grid(row=row, column=1)

        row +=1
        self.start_button.grid(row=row, column=0)
        self.exit_button.grid(row=row, column=1)

        # In der Ereignisschleife auf Eingabe des Benutzers warten.
        self.fenster.mainloop()

    def callback(self):
        self.path_to_non_live_img.set(askopenfilename())

    def calibrate(self):
        self.centers, self.H, self.window_name, self.beamer_img = calibrate(self.live.get(), self.homography.get(), self.path_to_non_live_img.get(), int(self.monitor.get()), int(self.threshold.get()), int(self.maxLineGap.get()),int(self.minLineLength.get()))

    def snapshot(self):
        if self.beamer_img is not None:
            main(centers=self.centers, H=self.H, window_name=self.window_name, beamer_img=self.beamer_img, processes=int(self.processes.get()), live=self.live.get(), do_homography=self.homography.get(), do_vectorized=self.vectorised.get(), contours=self.contours.get(), path_to_non_live_img=self.path_to_non_live_img.get(), density=float(self.density.get()), linewidth=float(self.linewidth.get()), charge_circle_radius=float(self.charge_circle_radius.get()))
        print("Please do calibration first!")

    def start(self):
        if self.beamer_img is not None:
            self.beamer_img = main(centers=self.centers, H=self.H, window_name=self.window_name, beamer_img=self.beamer_img, processes=int(self.processes.get()), live=self.live.get(), do_homography=self.homography.get(), do_vectorized=self.vectorised.get(), contours=self.contours.get(), path_to_non_live_img=self.path_to_non_live_img.get())
            self.fenster.after(1000, self.start)
        print("Please do calibration first!")

    def exit(self):
        cv2.destroyAllWindows()
        self.fenster.quit()


if __name__ == '__main__':
    GUI()
