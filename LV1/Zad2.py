x=float(input("Unesi broj od 0 do 1: "))
try:
    if x<0.0 or x>1.0:
        raise Exception("Broj nije unutar intervala")
    if x>=0.9:
        print("Ocjena pripada kategoriji A")
    elif x>=0.8 and x<0.9:
        print("Ocjena pripada kategoriji B")
    elif x>=0.7 and x<0.8:
        print("Ocjena pripada kategoriji C")
    elif x>=0.6 and x<0.7:
        print("Ocjena pripada kategoriji D")
    elif x<0.6:
        print("Ocjena pripada kategoriji F")
except ValueError:
    print("Uneseni tekst nije broj")
except Exception as e:
    print(e)