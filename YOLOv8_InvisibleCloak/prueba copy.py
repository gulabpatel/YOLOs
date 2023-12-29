
from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np


threshold = 0.1

kernel = np.ones((5,5),np.uint8)


#model = YOLO('best.pt')        #modelo entrenado con roboflow
model = YOLO('yolov8l-seg.pt')  #modelo base Large segmentacion Yolo
model.to('cuda')



#asigna la camara a openCV
cap = cv2.VideoCapture(0)

#lee el primer frame la camara y lo guarda como fondo (backgorund)
_,bg=cap.read()

time.sleep(2) #espera 2 segundos para entrar en la escena


print("Backgroun Capturado")
# Loop a traves de los frames capturados
while cap.isOpened():
    # lee los frames del stream de video y evalua el estado
    
    success, frame = cap.read()
    
    if success:
        # corre el modelo sobre los frames, con una confidencia de 0.5 y solo las clases tipo Persona
        result = model(frame,conf=0.5,classes=0)
        
        #consultael boundig box de los resultados y plotea sobre los frame sla prediccion
        bbox = result[0].plot()
        #Imprime en pantalla la imagen de la camara con la prediccion y la re-escala        
        cv2.imshow('imagenoriginal',cv2.resize(bbox, (720, 480)))
        #guarda los boundig boxes
        boxes=result[0].boxes
        #extra la clase del boundig box
        classID=boxes.cls.cpu().numpy()
        
        #evalua si la clase pertenece a humano para evitar mascaras en otros objetos 
        if classID.all()==0:               
            
            
            #obtiene las mascaras
            mask=result[0].masks
            #obtiene los boundig boxes en formato xmin,ymin,xmax,ymax              
            boxes=result[0].boxes.xyxy.cpu().numpy().astype(int)
            #consulta las dimensiones del frame
            (H, W) = frame.shape[:2]
              
                           
            #convierte el tensor de mascara en arreglo              
            mask=mask.masks[0].cpu().numpy()
            #re escala la mascara al tamano del frame
            mask = cv2.resize(mask, (W,H),interpolation=cv2.INTER_LINEAR)
            #se genera una mascara en ceros del tama;o del frame
            full_mask=np.zeros_like(frame[:,:,0],dtype=np.uint8)
            #compara los pixeles de la mascara original y los de la mascara negra y los mayores al treshold pone en 255    
            full_mask = (mask > threshold).astype(np.uint8)*255
            #aplica un filtro(kernel) para difunir la imagen    
            full_mask = cv2.dilate(full_mask,kernel,iterations=20)
            #asigna los valores del fondo en la mascara cuando son iguales a 255
            frame[full_mask==255] = bg[full_mask==255]
            
        #imprime en pantalle el frame editado "invisibilidad"
        cv2.imshow('imagen',cv2.resize(frame, (720, 480)))
        
        
        # hace un break a la rutina si se presiona la letra Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Hace un break si el stream de video ha terminado
        break

# libera la captura de video y cierra las ventanas

cap.release()
cv2.destroyAllWindows()

torch.cuda.is_available()