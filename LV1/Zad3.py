a=True
L=[]

while a==True:
    try:
        x=input("Unesi broj: ")
        if x=="Done":
            break
        if int(x):
            L.append(int(x))
        else:
            break
        
    except ValueError:
        print("Pogresan unos")
    print(L)

print(L)
print("Brojeva unutar liste: "+str(len(L)))
print("Projecna vrijednost brojeva u listi: "+ str(float(sum(L)/len(L))) )
print("Maks: " + str(max(L)))
print("Min: " + str(min(L)))
L.sort
print(L)