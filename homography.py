import cv2
import numpy as np
from numpy.linalg import svd
#from numpy.linalg import matmul
import matplotlib.pyplot as plt
import plotly.offline as py
import sys
from scipy.integrate import ode as ode
from matplotlib.patches import Circle
from numpy.linalg import svd
import time
import GeoTransformation as geoTrans
import screeninfo

plt.rcParams["figure.figsize"] = [7.00, 3.50]           #damit werden die Plots Vollbild
plt.rcParams["figure.autolayout"] = True
plt.rcParams['toolbar'] = 'None'


def find_intersection(line1, line2):                    #die defs sind alles Hough
    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py

def segment_lines(lines, delta):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < delta: # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2-y1) < delta: # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines

def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers

def make_1080p(Kamera):
    Kamera.set(3, 1920)
    Kamera.set(4, 1080)

def make_720p(Kamera):
    Kamera.set(3, 1280)
    Kamera.set(4, 720)

def make_480p(Kamera):
    Kamera.set(3, 640)
    Kamera.set(4, 480)

def change_res(Kamera, width, height):
    Kamera.set(3, width)
    Kamera.set(4, height)

def makeHomography(outfolder, path_to_non_live_img, live=True, camera=0, threshold=50, maxLineGap=100, minLineLength=200):
    plt.figure(facecolor='blue')                           #hier wird ein schwarzes Bild geplottet, für die Hough, damit kein Regenbogeneffekt entsteht
    fig = plt.scatter([1], [1], c='blue', s=1)
    plt.axis('off')
    plt.xlim(0, 1080)
    plt.ylim(0, 1920)
    vollbild1 = plt.get_current_fig_manager()
    vollbild1.full_screen_toggle()
    plt.pause(.5)

    Kamera = cv2.VideoCapture(camera)                            #das ist ein live-Bild das auf Knopfdruck das zu verarbeitende Bild liefert
    #make_1080p()
    if live:                                         #weil meine Kamera eine Art "Aufwachzeit hat" ist das notwendig bei mir
        while True:
            isTrue, img = Kamera.read()
            cv2.imshow('Video',img)
            if cv2.waitKey(20) & 0xFF==ord('d'):
                break
    else:
        img = cv2.imread(path_to_non_live_img)
    cv2.imshow("test",img)
    cv2.waitKey()
    Kamera.release()
    cv2.destroyAllWindows()

    img2 = np.copy(img)

    cv2.imwrite('Pictures/erstes Bild.png', img)
    img_output = cv2.resize(img, None, fx=1.0, fy=1.0)          #Bilverarbeitung
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3,3), dtype=np.uint8))

    #cv2.imshow("Dilated", dilated)
    #cv2.waitKey(0)
    #cv2.imwrite('dilated.png', dilated)

    # run the Hough transform
    lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi/180, threshold=threshold, maxLineGap=maxLineGap, minLineLength=minLineLength)    #Hier die Parameter für Hough

    # segment the lines
    delta = 60
    h_lines, v_lines = segment_lines(lines, delta)

    # draw the segmented lines
    houghimg = img.copy()                                        #Hough
    for line in h_lines:
        for x1, y1, x2, y2 in line:
            color = [0,0,255] # color hoz lines red
            cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)
        for line in v_lines:
            for x1, y1, x2, y2 in line:
                color = [255,0,0] # color vert lines blue
                cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)

    cv2.imshow("Segmented Hough Lines", houghimg)
    cv2.waitKey(0)
    #cv2.imwrite('hough.png', houghimg)

    # find the line intersection points
    Px = []
    Py = []
    for h_line in h_lines:
        for v_line in v_lines:
            px, py = find_intersection(h_line, v_line)
            Px.append(px)
            Py.append(py)

    # draw the intersection points
    intersectsimg = img.copy()
    for cx, cy in zip(Px, Py):
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)
        color = np.random.randint(0,255,3).tolist() # random colors
        cv2.circle(intersectsimg, (cx, cy), radius=2, color=color, thickness=-1) # -1: filled circle

    cv2.imshow("Data\Intersections", intersectsimg)
    cv2.waitKey(0)
    #cv2.imwrite('intersections.png', intersectsimg)

    # use clustering to find the centers of the data clusters
    P = np.float32(np.column_stack((Px, Py)))
    nclusters = 4
    centers = cluster_points(P, nclusters)
    np.save("Data\centers", centers)                             #hier werden die Cropp-Punkte für das andere Programm zwischengespeichert
    print(centers)

    # draw the center of the clusters
    for cx, cy in centers:                                  #eigentliche Vorgang fürs Croppen
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)
        cv2.circle(img, (cx, cy), radius=4, color=[0,0,255], thickness=-1) # -1: filled circle

    cv2.imshow("Center of intersection clusters", img)
    #cv2.imwrite('corners.png', img)
    cv2.waitKey(0)
    #Festlegen der koordinaten um die crop funktion auszuführen
    #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
    max = np.max(centers,0)
    #print(max)
    min = np.min(centers,0)
    #print(min)

    xc1 = int(min.item(0))                                       #das c, damit ich das später in der while Schleife nicht einfach überschreibe
    yc1 = int(min.item(1))
    xc2 = int(max.item(0))
    yc2 = int(max.item(1))
    print(x1, y1, x2, y2)
    list = [xc1, xc2, yc1, yc2]
    #mit den ausgegebenen koordinaten nun das bild croppen
    crop_img = img2[yc1:yc2, xc1:xc2]                            #fertige Bild
    cv2.imshow("cropped, start projector now", crop_img)
    cv2.waitKey()

    plt.close("all")
    #

    height = crop_img.shape[0]
    width = crop_img.shape[1]
    print(height, width)
    dimensions_original = img.shape
    print(dimensions_original)
    dimensions = crop_img.shape
    print(dimensions)                                           #das ist für mich um die Größe des Bildes zu sehen

    width1 = width-100
    height1 = height-100

    x = [100, height1, 100, height1]                               #hier mit den Bildgrenzen vom gecroppten Bild.
    y = [100, 100, width1, width1]                            #das sind die Koordinaten der Referenzpunkte

    print("x:", x)
    print("y", y)

    #x = [100, 980, 100, 980]                #hier mit 1080p Auflösung
    #y = [100, 100, 1820, 1820]


    i = 0
    PB = []                                             #leere Liste um die Mittelpunkte der aufgenommenen Punkte zu speichern
    while i <= 3:                                       #die vier durchläufe funzen aufjeden
        plt.figure(facecolor='blue')
        fig = plt.scatter(y[i], x[i], c='red', s=4000)
        plt.axis('off')
        plt.xlim(0, width)                          #hier noch die Bildgrenzen des gecroppten Bildes einfügen
        plt.ylim(height, 0)                         #y gespiegelt um koordinatensystem von oben anzufangen weil opencv von oben anfängt aber matplotlib die koordinaten von unten links anfängt
        vollbild = plt.get_current_fig_manager()
        vollbild.full_screen_toggle()
        plt.pause(.5)
        #plt.waitforbuttonpress(0)
        Kamera = cv2.VideoCapture(camera)                    # Kamera aktivieren (Kameraobjekt erstellen)
        #make_1080p()                                #Der Index kann auch -1, oder 1 sein, je nachdem welche Kamera sie verwenden
        if (Kamera.isOpened()== False):                 # prüfen ob die Kamera geöffnet ist
            print("Fehler")

        if (live):
            _, img = Kamera.read()
            Kamera.release()
            cv2.destroyAllWindows()
        else:
            img = cv2.imread(path_to_non_live_img)
        crop_img = img[yc1:yc2, xc1:xc2]                #jedes bild wird dann mit den vorher festgelgten Koordinaten gecroppt
        #cv2.imwrite("Punkte%d.png" %i, crop_img)
        img_red = np.copy(crop_img)
        # um die punkte zu finden rot rausfiltern
        hsv = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 50, 50])
        upper1 = np.array([8, 255, 255])
        mask0 = cv2.inRange(hsv, lower1, upper1)

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([170,50,50])
        upper2 = np.array([180,255,255])
        mask1 = cv2.inRange(hsv, lower2, upper2)

        maskr = mask0+mask1        #https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/
        img_red[np.where(maskr==0)] = 0
        XR,YR = np.where(np.all(img_red!=0, axis=2))
        zippedr = np.column_stack((XR,YR))
        cv2.imshow("rot",img_red)
        cv2.waitKey()
        cv2.destroyAllWindows()
        x_p = [p[0] for p in zippedr]
        y_p = [p[1] for p in zippedr]
        centroid = (sum(x_p) / len(zippedr), sum(y_p) / len(zippedr))       #Formel zur Ermittlung der Mittelpunke, aus den aufgenommenen Punkten
        #cent = int(centroid)
        #print(cent)

        PB.append(centroid)                                                  #Hier sind die Mittelpunkte der von der Kamera aufgenommenen Punkte gespeichert
        plt.pause(.5)
        plt.close("all")
        i += 1
    i=0
    #print(PB)
    PB = np.asarray(PB, dtype = int)                                                     #das hats für mich vereinfacht die Koordinatenpaare zu finden
                                                                        #xk und yk sind hier die von der Kamera aufgenommenen Punkte
    ub1 = xk1 = PB[0,0]                                                           #Hier könnte der Fehler liegen. Siehe Tewes: DLT und mehr.
    ub2 = xk2 = PB[1,0]                                                           #ich bin mir nicht sicher ob die Kamera v´ und u´ ist, oder der Beamer. Also was wird hier als Ausgang und was als Eingang angenommen?
    ub3 = xk3 = PB[2,0]                                                           #auch nicht sicher ob x und y richtig sind oder vertauscht werden müssen?!
    ub4 = xk4 = PB[3,0]                                                           # ist das falsch, dann müssen entweder die koordinaten oder die Matrix verändert werden
    #print("ub: ", ub1,ub2,ub3,ub4)
    vb1 = yk1 = PB[0,1]
    vb2 = yk2 = PB[1,1]
    vb3 = yk3 = PB[2,1]
    vb4 = yk4 = PB[3,1]
    #print("vb: ", vb1,vb2,vb3,vb4)
    u1 = xb1 = x[0]                                                              #xb und yb sind hier die dargestellten Punkte des Beamers
    u2 = xb2 = x[1]
    u3 = xb3 = x[2]
    u4 = xb4 = x[3]
    #print("u: ", u1,u2,u3,u4)
    v1 = yb1 = y[0]
    v2 = yb2 = y[1]
    v3 = yb3 = y[2]
    v4 = yb4 = y[3]
    #print("v: ", v1,v2,v3,v4)

    np.save("Data\PB", PB)

    #hier wird die matrix als Array nach Tewes erstellt
    #und hier ist halt die frage was ist was. Also ist Beamer oder kamera der Ausgang (u und v) oder ist Kamera das worauf es angewendet wird (u´und v´)
    A = np.array([[0,            0,      0,     u1,        v1,          1,       -u1*vb1,  -v1*vb1,     -vb1],
                [-u1,        -v1,     -1,     0,          0,          0,        u1*ub1,   v1*ub1,      ub1],
                [0,            0,      0,     u2,        v2,          1,       -u2*vb2,  -v2*vb2,     -vb2],
                [-u2,        -v2,     -1,     0,          0,          0,        u2*ub2,   v2*ub2,      ub2],
                [0,            0,      0,     u3,        v3,          1,       -u3*vb3,  -v3*vb3,     -vb3],
                [-u3,        -v3,     -1,     0,          0,          0,        u3*ub3,   v3*ub3,      ub3],
                [0,            0,      0,     u4,        v4,          1,       -u4*vb4,  -v4*vb4,     -vb4],
                [-u4,        -v4,     -1,     0,          0,          0,        u4*ub4,   v4*ub4,      ub4]

                ])
    u, s, vh = svd(A)
    res = vh[8,:]
    H = res.reshape((3,3))
    print(H)
    # A = H**(-1)
    # print(A)
    np.save("Data/Matrix", H)                                                    #Matrix wird für das Hauptprogramm als .npy Datei gespeichert und kann dann im Haupt eingelesen werden
    Kamera.release()

def name():
    print(__name__)

if __name__ == "__main__":
    makeHomography("Data/") # Fuer aufruf mit Kamera
    #makeHomography("Data/", False)

###########################################################################
#Hier ist das nicht-Tewes-Ding

# plt.figure(facecolor='black')                   #plottet am Anfang ein schwarzes Bild, um Regenbogeneffekt vom Beamer zu eliminieren !wichtig!
# fig = plt.scatter([1], [1], c='black', s=1)
# plt.axis('off')
# plt.xlim(0, 1080)
# plt.ylim(0, 1920)
# vollbild1 = plt.get_current_fig_manager()
# vollbild1.full_screen_toggle()
# plt.pause(1)
#
#
# Kamera = cv2.VideoCapture(camera)
# make_1080p()
# while True:
#     isTrue, img_test = Kamera.read()
#     cv2.imshow('Video',img_test)
#     if cv2.waitKey(20) & 0xFF==ord('d'):
#         break
# cv2.imshow("test",img_test)
# cv2.waitKey()
# Kamera.release()
# cv2.destroyAllWindows()
# img2 = np.copy(img)
#
# crop_img = img_test[yc1:yc2, xc1:xc2]
#
# x = [100, width1, 100, width1]                               #hier mit den Bildgrenzen vom gecroppten Bild.
# y = [100, 100, height1, height1]
#
# screen = screeninfo.get_monitors()[0]
# width, height = screen.width, screen.height
# pts_src = np.array([[100,100],[100,width1],[height1,100],[height1,width1]])
# pts_dst = np.array([[u1,v1],[u2,v2],[u3,v3],[u4,v4]])
# h, status = cv2.findHomography(pts_src, pts_dst)
# im_out = cv2.warpPerspective(img_test, h, (width,height))
#
#
#
# window_name = 'projector'
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
#                           cv2.WINDOW_FULLSCREEN)
# cv2.imshow(window_name, im_out)
# cv2.imshow("fertsch", im_out)
# cv2.waitKey()
############################################################################################################################################
#Am Ende soll das Programm die koordinaten fürs Croppen erzeugen und diese und die Matrix H speichern (np.save())
#Diese können dann im Hauptprogramm eingelesen werden (np.load())

#Dazu werden nacheinander die Koordinaten vom Beamer ausgegeben durch die while Schleife.
#Diese werden von der Kamera aufgenommen und durch "centroid" wird der Mittelpunkt der einzelnen Punkte bestimmt
#diese Punkte werden dann in der Liste PB gespeichert (Zeile 228)
#aus diesen mittelpunkten und den Koordinaten an denen der Beamer die Punkte projiziert wird die Matrix erstellt

#Also das Programm hier soll ohne das andere ablaufen aus mehreren Gründen:
#1. Das ist dann allgemein anwendbar und ist unabhängig von der Fragestellung der Arbeit
#2. Ist das hier durchgelaufen und der Aufbau ändert sich nicht, dann muss das hier nichtmehr beachtet werden
#3. außerdem kann nachdem das hier fertig ist, schon aufs whiteboard gezichnet werden (sonst machen die gezeichneten Punkte hier Probleme)
#4. Es ist somit ein klarer Cut zwischen den Programmen

#Probleme:
#liest man die Cropp-Koordinaten und die Matrix H im Hauptprogramm ein, dann
#geht bei der Homographie um Hauptprogramm irgendwas schief. das Croppen geht aber einwandfrei
#die Meldung ist: out of bounds
#Also da läuft irgendwas mit den Bildgrenzen falsch
#ich denke es liegt an den in diesem programm festgelegten Koordinaten (Zeile 236 bis 254) und die daraus entstehende matrix
#Ich seh allerdings den Fehler nicht und habe auch schon xb und yb sowie yk und xk durchgetauscht, ohne erfolg.
#Ich habe den Porgrammvorschlag mit Implementierung auch ausprobiert, was am Ende den selben Fehler erzeugt.
#aus diesem Grund denke ich, dass die Koordinaten und die daraus folgende Matrix falsch ist

#wenn es nicht die matrix ist, dann kann es daran liegen, dass die Kamera die Koordinaten von unten links andängt
#aber matplotlib von oben links
#ich hab die die grenzen von y in der Schleife umgedreht um das zu lösen, kann aber sein, dass das nicht so geht
