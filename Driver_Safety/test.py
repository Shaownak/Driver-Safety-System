import cv2


cap = cv2.VideoCapture(r"G:\Driver Safety\Driver_Safety\test_video\test.mp4")



while True:
    
    _,img = cap.read()
    
    cv2.imshow("Image", img)
    
    
    x = img.shape[0]
    y = img.shape[1]
    
    print(x,y)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()