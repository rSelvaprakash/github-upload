from tkinter import *
import tkinter
import tkinter.messagebox
def Knn():
    from Algorithms import knn
def Svm():
    from Algorithms import svm
def Naivebayes():
    from Algorithms import naivebayes   
def Decision():
    
    from Algorithms import Decision
    
root = Tk()
root.title('Evaluation')
L1=Label(root,text="Comparission of Classification Algorithms on Intrusion Detection")
L1.pack()
L2=Label(root,text="Dataset : NSL KDD Loaded").pack()
button = Button(root, text="Decision", command=Decision) 
button.pack()
button1=Button(root,text="NaiveBayes",command=Naivebayes)
button1.pack()
button2=Button(root,text="K-NN",command=Knn)
button2.pack()
button3=Button(root,text="Support Vector Machine",command=Svm)
button3.pack()
bu2 = Button(root, text="CLEAR", command=lambda :textbox.delete(1.0,END))
bu2.pack()

textbox=Text(root)
textbox.pack()

def redirector(inputStr):
    textbox.insert(INSERT, inputStr)

sys.stdout.write = redirector #whenever sys.stdout.write is called, redirector is called.

root.mainloop()
