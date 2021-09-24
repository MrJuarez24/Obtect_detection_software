import argparse
import os
import cv2
import time
import numpy as np
import math
from glob import glob
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import sys
import flask

p = os.path.abspath('..')
if p not in sys.path:
    sys.path.append(p)


import print_utils
from object_detectors.torch.yolo_object_detector import YOLO_Object_Detector
from coco_format_utils import Coco_Annotation_Object, Coco_Annotation_Set

id = 1
alarma = 0
areas = [[]]

nombres_fotos = []
historial_alarmas = []

def generator_from_video(video_path):
    vidcap = cv2.VideoCapture(video_path)    #cv2.CAP_FFMPEG        -DWITH_FFMPEG=ON
    success, image = vidcap.read()           # We try to read the next image
    while success:
        yield image
        success, image = vidcap.read()           # We try to read the next image

def printbien(trazas):
    i = 0
    hay = 1
    while i < len(trazas) and hay:
        if trazas[i][6] == 0:
            hay = 0
        else:
            print(trazas[i])
        i += 1



def arreglarMatrix(trazas):
    for i in range(len(trazas)):
        if i+1 < len(trazas) and trazas[i][0] == 0:
            if trazas[i+1][0] != 0:
                j = 0
                it = 0
                while j <= i and it == 0:
                    if trazas[j][0] == 0:
                        it = j
                    j += 1

                for t in range(len(trazas[0])):
                    trazas[it][t] = trazas[i+1][t]
                    trazas[i+1][t] = 0


def tratarproblemas(areas, id, trazas_sin_salir):
    global nombres_fotos
    global historial_alarmas

    global alarma

    #Primero nos quedamos con las variables y, x, además de hacer la correspondiente interpolación
    if len(areas) > 5:
        y = areas[-5:]
    else:
        y = areas

    x = np.arange(len(y))

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = np.nan
    
    y = pd.Series(y)
    y = y.interpolate()

    #Tras esto vamos a hacer la regresión lineal entre los últimos 5 frames para ver la pendiente y ver si hay problemas (> umbral)
    #Variables por lo que cambia esto: frame rate(del video) y cuantos analizamos por segundo

    slope, intercept, r, p, se = stats.linregress(x, y)
    print("Pendiente regresion para el objeto " + str(id + 1) + ": " + str(slope))

    valorpendientemaxima = 25000
    if len(y) <= 2:
        valorpendientemaxima = 150000
    elif len(y) <= 3:
        valorpendientemaxima = 90000
    elif len(y) <= 4:
        valorpendientemaxima = 35000

    lastAreaDif = 0
    if len(areas) > 2:
        lastAreaDif = areas[len(areas) - 1] - areas[len(areas) - (2+int(trazas_sin_salir))]


    if slope > valorpendientemaxima or (lastAreaDif > 1.5 * areas[len(areas) - (2+int(trazas_sin_salir))] and lastAreaDif > 25000):
        print("OBJETO CULPABLE DE LA ALARMA: " + str(id + 1))
        if slope > valorpendientemaxima:
            print("Pendiente: " + str(slope))
        
        if lastAreaDif > areas[len(areas) - (2+int(trazas_sin_salir))]:
            print("¿Subida brusca?, último áreas: " + str(areas[len(areas) - (2+int(trazas_sin_salir))]) + ". Diferencia con el nuevo área" + str(lastAreaDif))

        alarma = 1

    
    #Tras esto vamos a hacer vamos a hacer las gráficas de los últimos 5 frames
    
    #Hacemos las gráficas completas de los objetos

    



def procesarTrazabilidad(trazas, trazasnuevas, iterador):
    global historial_alarmas
    global nombres_fotos

    global id
    global alarma
    global areas
    frames_hasta_eliminar = 4
    frames_muy_lejano = 200

    '''
    Matriz de costes:  trazasnuevas x trazas
[                    Traza 1, antigua        Traza 2, antigua       Traza 3, antigua
Traza a, nueva        a1 = Coste(a,1)        a2 = Coste(a,2)         a3 = Coste(a,3)
Traza b, nueva        b1 = Coste(b,1)        b2 = Coste(b,2)         b3 = Coste(b,3)
Traza c, nueva        c1 = Coste(c,1)        c2 = Coste(c,2)         c3 = Coste(c,3)
Traza d, nueva        d1 = Coste(d,1)        d2 = Coste(d,2)         d3 = Coste(d,3)
]

    trazas:         [CentroideX       CentroideY      Clase       Frame_analizado      Frames_sin_aparecerw     id]

    trazasnuevas:   [CentroideX       CentroideY      Clase       id]
    '''

    matriz_costes = np.zeros((len(trazasnuevas),len(trazas)))


    for i in range(len(trazasnuevas)):
        for j in range(len(trazas)):
            if trazas[j][6] == 0:
                matriz_costes[i][j] = 50000
            else:
                coste = abs(trazasnuevas[i][0]- trazas[j][0]) + abs(trazasnuevas[i][1] - trazas[j][1])
                if coste > frames_muy_lejano:            #Está muy lejos: no es el mismo objeto o es uno nuevo
                    matriz_costes[i][j] = 50000
                else:                                    #No está en ningún caso por lo que es posible
                    matriz_costes[i][j] = coste

    print("Matriz de costes: ")
    print(matriz_costes)

    row_ind, col_ind = linear_sum_assignment(matriz_costes)

    '''
    Si el máximo es a1, b3, c2 y d3:
    row_ind = [0    1   2   3]
    col_ind = [0    2   1   2]
    '''

    cantidad_objetos = 0
    for i in range(len(trazas)):
        if trazas[i][6] != 0:
            cantidad_objetos += 1

    #Ahora actualizamos las trazas que son el mismo objeto y añadimos como objetos nuevos los que no entren


    for i in row_ind:
        columna = col_ind[i]
        if matriz_costes[i][columna] < frames_muy_lejano:

            trazas_sin_salir = trazas[columna][4]
            
            for j in range(int(trazas[columna][4])):                           #Rellenamos de 0 las áreas para los frames que no ha salido
                areas[int(trazas[columna][6]) - 1].append(0)

            trazas[columna][0] = trazasnuevas[i][0]                            #CentroideX
            trazas[columna][1] = trazasnuevas[i][1]                            #CentroideY
            trazas[columna][2] = trazasnuevas[i][2]                            #Clase
            trazas[columna][3] = iterador                                      #Frame_analizado
            trazas[columna][4] = 0                                             #Frames sin salir
            trazas[columna][5] = trazasnuevas[i][3]                            #Area del objeto
            #trazas[columna][6] = trazasnuevas[i][4]                           #id del objeto

            areas[int(trazas[columna][6]) - 1].append(int(trazas[columna][5]))

            tratarproblemas(areas[int(trazas[columna][6]) - 1], int(trazas[columna][6]) - 1, trazas_sin_salir)
        else:    
            if id == 1:
                areas[0].append(int(trazasnuevas[i][3]))
            else:
                areas.append([int(trazasnuevas[i][3])])

            trazas[cantidad_objetos][0] = trazasnuevas[i][0]                   #CentroideX
            trazas[cantidad_objetos][1] = trazasnuevas[i][1]                   #CentroideY
            trazas[cantidad_objetos][2] = trazasnuevas[i][2]                   #Clase
            trazas[cantidad_objetos][3] = iterador                             #Frame_analizado
            trazas[cantidad_objetos][4] = 0                                    #Frames sin salir
            trazas[cantidad_objetos][5] = trazasnuevas[i][3]                   #Area del objeto
            trazas[cantidad_objetos][6] = id                                   #id del objeto
            id += 1
            cantidad_objetos += 1

            
            
    
    #Ahora recorremos y los que no han sido añadidos en esta iteración le añadimos 1 a los frames sin salir
    #Tras esto eliminamos los que no han salido en más de X frames y reajustamos el array para no tener problemas

    for i in range(len(trazas)):
        if trazas[i][3] != iterador and trazas[i][6] != 0:
            trazas[i][4] += 1
        
        if trazas[i][4] >= frames_hasta_eliminar:
            for t in range(len(trazas[0])):
                    trazas[i][t] = 0
    
    #Arreglamos la matriz
    arreglarMatrix(trazas)

def main(path_video, path_output_json, path_output_images):
    global alarma
    global areas
    global nombres_fotos
    global historial_alarmas
    frames_analizados = 20      #si =20, analizaremos uno de cada 20 (3/s, si es de 60fps)

    print("Loading YOLO torch object detector")
    object_detector = YOLO_Object_Detector(model="default")

    print(f"Input video: {path_video}")
    print(f"Output results json will be written in {path_output_json}")
    print(f"Output images will be saved in {path_output_images}")

    
    if not os.path.isfile(path_video):
        print(f"{path_video} is not file.")
        print("ERROR: There is no input.")
        quit()
    
    
    if not os.path.isdir(os.path.dirname(path_output_json)):                                         #Creamos si no existe la dirección de salida
        os.makedirs(os.path.dirname(path_output_json))
    
    coco_annotations = Coco_Annotation_Set()

    # We need to initialize the image input generator.
    image_generator = generator_from_video(path_video)

    # Main loop
    image_index = 1
    obj_id = 1
    process_times_list = []
    initial_process_time = time.time()
    
    print("Start video process.")

    trazas = np.zeros((15,7))            #[CentroideX       CentroideY      Clase       Frame_analizado      Frames_sin_aparecer       Area     id]
    
    #trazasnuevas = []                  #[CentroideX       CentroideY      Clase       Area                 id]
    
    iterador = 1

    for image in image_generator:                                   # While there is a next image.

        if image_index % frames_analizados == 0:
            print(image_index)

            frame_filename = "frame_{:0>6}.png".format(image_index)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            initial_frame_time = time.time()

            preprocessed_images = object_detector.preprocess(rgb_image)                  # We preprocess the batch.
            outputs = object_detector.process(preprocessed_images)                          # We apply the model.
            outputs = object_detector.filter_output_by_confidence_treshold(outputs, treshold = 0.5)         # We filter output using confidence.
            
            #print(outputs)

            img_output = outputs[0]
            
            it=0
            # Now we create the coco format annotation for this image.
            for bbox, _class, confidence in zip(img_output[0], img_output[1], img_output[2]):
                coco_object = Coco_Annotation_Object(bbox=bbox, category_id=_class, id=obj_id, image_id=image_index, score=confidence)
                coco_annotations.insert_coco_annotation_object(coco_object)
                obj_id+=1
                if any(_class == x for x in [2,3,4,6,8]):
                    it+=1


            trazasnuevas = np.zeros((it,4))

            it=0
            for bbox, _class, confidence in zip(img_output[0], img_output[1], img_output[2]):
                if any(_class == x for x in [2,3,4,6,8]):               #Comprobamos que sea algún vehículo el objeto
                    trazasnuevas[it][0] = (bbox[0] + bbox[2])/2
                    trazasnuevas[it][1] = (bbox[1] + bbox[3])/2
                    trazasnuevas[it][2] = _class
                    trazasnuevas[it][3] = bbox[2]*bbox[3]/2
                    it += 1
                obj_id += 1


            
            print("Trazabilidad en el frame: " + str(image_index))
            
            print("Objetos a añadir: ")
            print(trazasnuevas)
            procesarTrazabilidad(trazas, trazasnuevas, iterador)
            print("Final: ")
            printbien(trazas)

            '''
            print("Áreas:")
            print(areas)
            '''

            if path_output_images is not None:
                drawn_image = print_utils.print_detections_on_image(img_output, rgb_image[:,:,[2,1,0]], trazas)
                cv2.imwrite(os.path.join(path_output_images, frame_filename), drawn_image)

                nombres_fotos.append(os.path.join("/static/images/", frame_filename))
                historial_alarmas.append(alarma)

            
            if alarma == 1:
                print("ALARMA")
                alarma = 0
                #os.system("pause")
            
            

            process_time = time.time()-initial_frame_time
            process_times_list.append(process_time)

            iterador += 1
        
        image_index += 1
    

    print(f"Total process time {time.time()-initial_process_time}, average time per frame {np.array(process_times_list).mean()}")

    coco_annotations.to_json(path_output_json)

    return nombres_fotos, historial_alarmas
    
