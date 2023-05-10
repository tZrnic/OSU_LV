def total_euro(a):
    sat=8.5
    return a*sat

try:
    a=int(input("Broj radnih sati: "))
except ValueError:
    print("Uneseni tekst nije broj")
print("zarada: "+ str(total_euro(a)))