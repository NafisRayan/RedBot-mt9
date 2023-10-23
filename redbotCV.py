#redbotCV

def callCV(loc=0):

  import cv2

  vid = cv2.VideoCapture(loc)

  box = cv2.CascadeClassifier('fist.xml')

  while True:
      ret, img = vid.read()

      # Detect Objects in the img
      Objects = box.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)#, minSize=(30, 30))

      for (x, y, w, h) in Objects:

          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 155, 255), 2)

          # center coordinates of the detected Object
          Object_center_x = x + w // 2
          Object_center_y = y + h // 2

          # center coordinates of the whole screen
          screen_center_x = img.shape[1] // 2
          screen_center_y = img.shape[0] // 2


          (a,b)=(Object_center_x-screen_center_x),-(Object_center_y-screen_center_y)

          # line from the screen center to the Object center
          cv2.line(img, (screen_center_x, screen_center_y), (Object_center_x, Object_center_y), (0, 0, 255), 2)

          # THESE ARE HERE JUST TO MARK THE CENTER
          c1 = cv2.circle(img, (Object_center_x,Object_center_y), 1, (255, 0, 0), 2)
          c2 = cv2.circle(img, (screen_center_x,screen_center_y), 1, (255, 0, 0), 2)

          cv2.putText(img,f"(x,y)={a,b}",(20,img.shape[0]-50),cv2.FONT_ITALIC,1,(0,250,0),4)



      cv2.imshow('Detection', img)

      # Exit the loop if the 'q' key is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  vid.release()
  cv2.destroyAllWindows()
  destination_coordinate = a,b
  return destination_coordinate

print(callCV())

