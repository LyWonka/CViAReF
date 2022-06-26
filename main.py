import cv2 # Zur
import numpy as np # Vectorizierung,
import matplotlib.pyplot as plt
import time
import screeninfo
import numpy as np
from scipy.integrate import ode as ode
from matplotlib.patches import Circle
from numpy.linalg import inv
from skimage.color import *
import GeoTransformation as geoTrans
import homography as Homography

from multiprocessing import Pool

from ctypes import *
import ctypes

plt.rcParams["figure.figsize"] = [7.00, 3.50]   #legt die plt Parameter fest
plt.rcParams["figure.autolayout"] = True
plt.rcParams['toolbar'] = 'None'

# Nur für die Summenfeld Funktion ohne Vektorisierung
def E(q, q_x, q_y, x, y):
    if (q_x==x and q_y==y):
        return 0,0 # ansonsten bekommen wir Probleme mit einer Division durch Null
    else:
        den = np.hypot(x-q_x, y-q_y)**3 # np.hypoth(a,b) = np.sqrt(a,b)
        return q * (x - q_x) / den, q * (y - q_y) / den

def Summenfeld(x_coords, y_coords, nrows, ncols, charge_grid):
    Ex, Ey = np.zeros((nrows, ncols)), np.zeros((nrows, ncols)) # <-- Ex und Ex sind 2D-Arrays oder eben grids mit der Größe nrows x ncols (Zeilen mal Spalten)
    charge_idx = np.argwhere(charge_grid !=0) # <-- array der (x,y)-Koordinaten an denen sich Ladungen befinden
    print("index gesetzt")
    for pos in charge_idx: # <-- hier wird nur über die Position (x,y) iteriert, an denen sich tatsächlich eine von Null verschiedene Ladung befindet
        for i,x in enumerate(x_coords): # <-- hier wird über alle x-Positionen iteriert
            for j,y in enumerate(y_coords): # hier wird über alle y-Positionen iteriert
                ex,ey = E(charge_grid[pos[0],pos[1]], x_coords[pos[1]], y_coords[pos[0]], x, y)
                Ex[i,j] += ex # Bildung des Summenfeldes
                Ey[i,j] += ey # Bildung des Summenfeldes
    return(Ex, Ey)

def do_multiprocessing_summenfeld(x, y, x_coords, y_coords, charge_grid, Ex, Ey, charge_idx):
    for pos in charge_idx:
        q = charge_grid[pos[0], pos[1]]
        qx = x_coords[pos[1]]
        qy = y_coords[pos[0]]

        xqx = x - qx
        yqy = y - qy
        den = np.hypot((xqx), (yqy))**3

        # damit er nicht durch 0 teilt, wird der Wert in den = 1 und für xqx = yqy = 0 gesetzt, damit das Ergebnis 0 ist
        for test in np.argwhere(den == 0):
            den[test[0], test[1]] = 1
            xqx[test[0], test[1]] = 0
            yqy[test[0], test[1]] = 0
        Ex += q * np.divide(xqx, den)
        Ey += q * np.divide(yqy, den)
    return(Ex, Ey)


def Summenfeld_vectorized(x_coords, y_coords, nrows, ncols, charge_grid, processes):
    Ex, Ey = np.zeros((nrows, ncols)), np.zeros((nrows, ncols)) # <-- Ex und Ex sind 2D-Arrays oder eben grids mit der Größe nrows x ncols (Zeilen mal Spalten)
    charge_idx = np.argwhere(charge_grid !=0) # <-- array der (x,y)-Koordinaten an denen sich Ladungen befinden

    if len(x_coords) >= len(y_coords):
        x = x_coords + np.zeros((len(y_coords), len(x_coords)))
        y = np.transpose(y_coords + np.zeros((len(x_coords), len(y_coords))))
    else:
        x = np.transpose(x_coords + np.zeros((len(y_coords), len(x_coords))))
        y = y_coords + np.zeros((len(x_coords), len(y_coords)))

    # each multiprocess handle one part of the charge_idx
    charge_idx = np.array_split(charge_idx, processes)

    inputs = [(x, y, x_coords, y_coords, charge_grid, Ex, Ey, charge_idx[i]) for i in range(processes)]

    with Pool(processes) as p:
        results = p.starmap(do_multiprocessing_summenfeld, inputs)

    for r in results:
        Ex += r[0]
        Ey += r[1]

    return(Ex, Ey)


def Summenfeld_c(nrows, ncols, charge_grid, x_coords, y_coords):
    """
    Python Funktion die die Summenfeld Berechnung in eine C Funktion auslagert
    """
    fun = ctypes.CDLL("C\summenfeld.so")
    fun.Summenfeld.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),c_size_t, c_size_t, c_size_t, c_size_t]
    fun.Summenfeld.restype = None

    # charges müssen in ein eindimensionales array gespeichert werden (bspw. charges[n]=Ladung am Ort (row,col) mit n = Anzahl_Spalten * row + col
    charges = charge_grid.astype(ctypes.c_float)
    x_coords = x_coords.astype(ctypes.c_float)
    y_coords = y_coords.astype(ctypes.c_float)

    # diese werden im C-Programm gefüllt
    E_x = np.zeros(nrows*ncols, dtype=ctypes.c_float)
    E_y = np.zeros(nrows*ncols, dtype=ctypes.c_float)

    # hier wird die Zahl der Einträge im Array gespeichert, also size = Anzahl_Zeilen * Anzahl_Spalten
    rows = ctypes.c_size_t(nrows)

    # hier wird die Anzahl der Spalten gespeichert
    cols = ctypes.c_size_t(ncols)
    x_coords_size = ctypes.c_size_t(len(x_coords))
    y_coords_size = ctypes.c_size_t(len(y_coords))

    # Hier wird der Datentyp (Pointer auf float) definiert
    floatp = ctypes.POINTER(ctypes.c_float)

    # hier wird die eigentliche C-Funktion aufgerufen
    fun.Summenfeld(charges.ctypes.data_as(floatp), E_x.ctypes.data_as(floatp), E_y.ctypes.data_as(floatp), x_coords.ctypes.data_as(floatp), y_coords.ctypes.data_as(floatp), rows, cols, x_coords_size, y_coords_size)

    E_x = np.reshape(E_x, (nrows,ncols))
    E_y = np.reshape(E_y, (nrows,ncols))

    return(E_x, E_y)


#um Laufzeit zu verkürzen, Mittelpunkt der Formen finden
def get_contoures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    charges = []

    i = 0

    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 255, 0), 3)

        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
             x = int(M['m10']/M['m00'])
             y = int(M['m01']/M['m00'])
             charges.append([x,y])
    return(charges)

def calibrate(live, do_homography, path_to_non_live_img, monitor, camera, threshold=50, maxLineGap=100, minLineLength=200):
    # Anstelle die Homography seperat auszuführen, würde ich es abhängig von der Datei "centers.py" machen. ist diese vorhanden benutzte sie,
    # wenn nicht berechne sie neu. (eventuell auch mit zusätzlichem Parameter eine neu berechnung erzwingen lassen)
    #if (os.path.exists("Data/centers.npy") and os.path.exists("Data/Matrix.npy")):
    if(do_homography):
        Homography.makeHomography("Data/", path_to_non_live_img, live, camera, threshold, maxLineGap, minLineLength)
        centers = np.load("Data/centers.npy")
        H = np.load("Data/Matrix.npy")
    else:
        centers = np.load("Data/centers.npy")
        H = np.load("Data/Matrix.npy")

    screen = screeninfo.get_monitors()[monitor]
    width, height = screen.width, screen.height
    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    black_img = np.zeros((height, width))
    cv2.imwrite("Pictures/black.png", black_img)

    return(centers, H, window_name, black_img)

def take_a_picture(window_name, live, camera, path_to_non_live_img):
    #plottet am Anfang ein schwarzes Bild, um Regenbogeneffekt vom Beamer zu eliminieren !wichtig!
    black_img = cv2.imread("Pictures/black.png")
    cv2.imshow(window_name, black_img)
    cv2.waitKey(2)

    if(live):
        Kamera = cv2.VideoCapture(camera)
        _, img = Kamera.read()
        Kamera.release()
    else:
        img = cv2.imread(path_to_non_live_img)

    img = cv2.resize(img,(640,480))

    return(img)


def main(centers, H, window_name, beamer_img, processes=4, live=True, do_homography=True, camera=0, do_vectorized=True, contours=False, path_to_non_live_img="Pictures/Ladungen_ohneBeamer.png", density=1, linewidth=1, charge_circle_radius=0.05):

    # Nehme neues Foto auf
    img = take_a_picture(window_name, live, camera, path_to_non_live_img)

    # Da für die Aufnahme das Bild geschwärzt wird muss danach das beamer_img wieder ausgegeben werden
    cv2.imshow(window_name, beamer_img)
    cv2.waitKey(2)

    # Croppe das Bild falls gewünscht
    if do_homography:
        #Festlegen der koordinaten um die crop funktion auszuführen
        max = np.max(centers,0)
        min = np.min(centers,0)

        xc1 = int(min.item(0))
        yc1 = int(min.item(1))
        xc2 = int(max.item(0))
        yc2 = int(max.item(1))

        #mit den ausgegebenen koordinaten nun das bild croppen
        img = img[yc1:yc2, xc1:xc2]
        cv2.imwrite("Pictures/CroppedImg.png",img)

    #damit die Koordinaten von OpenCV "falschrum sind", umd bei Matplotlib das richtige ergebnis zu bekommen
    img = cv2.flip(img, 0)

    #Größe des Ergebnisbildes definiert
    nrows, ncols = len(img), len(img[0])
    charge_grid = np.zeros((ncols, nrows))

    #Bild in HSV konvertieren um rote und blaue Maske zu erstellen
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # rote Maske
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([8, 255, 255])
    mask0 = cv2.inRange(hsv, lower1, upper1)

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([170,50,50])
    upper2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower2, upper2)

    # combine both parts to red mask
    maskr = mask0+mask1

    # set all red pixels to 255 and all non red pixels to 0
    img_red = np.copy(img)
    img_red[np.where(maskr==0)] = 255
    img_red[np.where(maskr!=0)] = 0

    # cv2.waitKey(10)
    # cv2.imwrite("Pictures/red.png", img_red)

    # to save time use only contours
    if(contours):
        charges = get_contoures(img_red)
        for charge in charges:
            charge_grid[charge[0], charge[1]] = 1
    else:
        charge_grid = charge_grid + np.all(img_red!=0, axis=2).astype(int).transpose() * -1

    # blaue Maske
    lower3 = np.array([90, 50, 70])
    upper3 = np.array([128, 255, 255])
    maskb = cv2.inRange(hsv, lower3, upper3)
    img_blue = np.copy(img)

    img_blue[np.where(maskb==0)] = 255
    img_blue[np.where(maskb!=0)] = 0
    # cv2.waitKey(10)
    # cv2.imwrite('Pictures/blue.png', img_blue)
    if(contours):
        charges = get_contoures(img_blue)
        for charge in charges:
            charge_grid[charge[0], charge[1]] = -1
    else:
        charge_grid = charge_grid + np.all(img_blue!=0, axis=2).astype(int).transpose()

    # berechnen der Feldlinien
    # Für ungerade Zahlen überprüfen, ob es keinen Fehler gibt
    x_coords = np.linspace(0, nrows, nrows) # <-- Erzeuge ein array von Zahlen zwischen -2 und 2 mit insgesamt nx Elementen. Die 2 gibt hier eine Skalierung vor, die wir natürlich selber festlegen können. 1 Pixel = x m,cm,mm ???
    y_coords = np.linspace(0, ncols, ncols) # <-- dito mit ny Elementen

    # Wir berechnen das elektrische (Summen-)Feld aller Ladungen an allen Stellen x und y
    charge_idx = np.argwhere(charge_grid !=0) # <-- array der (x,y)-Koordinaten an denen sich Ladungen befinden

    if (do_vectorized):
        start = time.time()
        Ex, Ey = Summenfeld_vectorized(x_coords, y_coords, nrows, ncols, charge_grid, processes)
        time_needed = str(time.time() - start)
        print("Summenfeld_vectorized " + time_needed)

        #with open("time_needed_summenfeld_vectorized.txt","a") as out:
        #    out.write(time_needed + "\n")
    else:
        # #Fuer die Berechnung des Summenfeldes ohne Vektorisierung
        start = time.time()
        Ex, Ey = Summenfeld(x_coords, y_coords, nrows, ncols, charge_grid)
        time_needed = str(time.time() - start)
        print("Summenfeld " + time_needed)

        #with open("time_needed_summenfeld.txt","a") as out:
        #    out.write(time_needed + "\n")

#    start = time.time()
#    Ex, Ey = Summenfeld_c(nrows, ncols, charge_grid, x_coords, y_coords)
#    print("Summenfeld_c " + str(time.time() - start))

    fig = plt.figure()
    fig.set_size_inches(3.8,2.3)
    ax = fig.add_subplot(111)

    # Plot the streamlines with an appropriate colormap and arrow style
    color = 2 * np.log(np.hypot(Ex, Ey))
    ax.streamplot(y_coords, x_coords, Ey, Ex, color=color, linewidth=linewidth, cmap=plt.cm.inferno,
                  density=density, arrowstyle='->', arrowsize=1.0)

    # Zeichne ebenfalls die Punktladungen ein
    charge_colors = {False: '#0000aa', True: '#aa0000'}
    for pos in charge_idx:
        val = charge_grid[pos[0], pos[1]]
        ax.add_artist(Circle(pos, charge_circle_radius, color=charge_colors[val>0]))
    plt.axis("off")

    plt.savefig("Pictures/Beamer_Image.png", dpi=150, bbox_inches = "tight", pad_inches = 0)
    beamer_img = cv2.imread("Pictures/Beamer_Image.png")
    plt.close()

    #################################################################################
    dim = (img.shape[1], img.shape[0])
    beamer_img = cv2.resize(beamer_img, dim)

    #################################################################################
    if do_homography:
        b,g,r = cv2.split(beamer_img)
        H = inv(H)
        beamer_imgb, displacement = geoTrans.geoTransformation(b, H, "bilinear")
        beamer_imgg, displacement = geoTrans.geoTransformation(g, H, "bilinear")
        beamer_imgr, displacement = geoTrans.geoTransformation(r, H, "bilinear")

        beamer_img = cv2.merge([beamer_imgb, beamer_imgg, beamer_imgr])
        # cv2.imshow("merged, Homografie", beamer_img)
        ###############################################################################

        PB = np.load("Data/PB.npy")

        ub1 = xk1 = PB[0,0]                                                           #Hier könnte der Fehler liegen. Siehe Tewes: DLT und mehr.
        ub2 = xk2 = PB[1,0]                                                           #ich bin mir nicht sicher ob die Kamera v´ und u´ ist, oder der Beamer. Also was wird hier als Ausgang und was als Eingang angenommen?
        ub3 = xk3 = PB[2,0]                                                           #auch nicht sicher ob x und y richtig sind oder vertauscht werden müssen?!
        ub4 = xk4 = PB[3,0]                                                           # ist das falsch, dann müssen entweder die koordinaten oder die Matrix verändert werden

        vb1 = yk1 = PB[0,1]
        vb2 = yk2 = PB[1,1]
        vb3 = yk3 = PB[2,1]
        vb4 = yk4 = PB[3,1]

        vector1 = (np.matmul(H, np.array([[ub1],[vb1],[1]])))
        vector2 = (np.matmul(H, np.array([[ub4],[vb4],[1]])))
        (rr1,cc1) = (int(vector1[0]/vector1[2]+displacement[0]), int(vector1[1]/vector1[2]+displacement[1]))
        (rr2,cc2) = (int(vector2[0]/vector2[2]+displacement[0]), int(vector2[1]/vector2[2]+displacement[1]))

        #Koordinaten fürs croppen
        xc1 = rr1-100
        xc2 = rr2+100
        yc1 = cc1-100
        yc2 = cc2+100

        beamer_img = beamer_img[xc1:xc2, yc1:yc2]
    ###############################################################################

    # cv2.imwrite("Pictures/Kamera.png", beamer_img)


    cv2.imshow(window_name,beamer_img)
    cv2.waitKey(2)

    return(beamer_img)

if __name__ == "__main__":
    threshold=50
    maxLineGap=100
    minLineLength=200

    live = False
    monitor = 0
    do_homography = False
    camera = 0
    do_vectorized = True
    contours = False
    processes = 4
    path_to_non_live_img = "Pictures/LEIFI/quadropol_ohneFeldlinien.png"
    density = 1
    linewidth = 1
    charge_circle_radius = 0.05


    centers, H, window_name, beamer_img = calibrate(live, do_homography, path_to_non_live_img, monitor, threshold, maxLineGap, minLineLength)

    beamer_img = main(centers=centers, H=H, window_name=window_name, beamer_img=beamer_img, processes=processes, live=live, do_homography=do_homography, camera=camera, do_vectorized=do_vectorized, contours=contours, path_to_non_live_img=path_to_non_live_img, density=density, linewidth=linewidth, charge_circle_radius=charge_circle_radius)
