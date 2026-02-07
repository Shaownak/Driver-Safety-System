import threading

def noob():
    while True:
     print("Fuck You")
     
     
def gg():
    while True:
        print("Fucking on progress")
        
        
def lol():
    while True:
        print("Are you okay?")

def fuckk():
    while True:
        print("Bitch")
        
def damnboy():
    while True:
        print("Bitch 2")
        
def yeahboy():
    while True:
        print("Why are you Gay")
        
def kamon():
    while True:
        print("chole")
        
        
        
noob_thread = threading.Thread(target=noob)
gg_thread = threading.Thread(target=gg)
lol_thread = threading.Thread(target=lol)
fuckk_thread = threading.Thread(target=fuckk)
damnboy_thread = threading.Thread(target=damnboy)
yeahboy_thread = threading.Thread(target=yeahboy)
kamon_thread = threading.Thread(target=kamon)



noob_thread.start()
gg_thread.start()
lol_thread.start()
fuckk_thread.start()
damnboy_thread.start()
yeahboy_thread.start()
kamon_thread.start()


noob_thread.join()
gg_thread.join()
lol_thread.join()
fuckk_thread.join()
damnboy_thread.join()
yeahboy_thread.join()
kamon_thread.join()

        