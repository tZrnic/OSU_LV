text = open('SMSSpamCollection.txt')

hamC = 0
spamC = 0
hamW = 0
spamW = 0
usklicnik = 0


for line in text:
    line = line.rstrip()
    words = line.split()
    if words[0] == 'ham':
        hamC += 1
        hamW += len(words[1:])
    else:
        if line.endswith("!"):
            usklicnik += 1
        spamW += len(words[1:])  
        spamC += 1
        
        
print(f'Spam: {spamW/spamC}')
print(f'Ham: {hamW/hamC}')
print(f'usklicnici: {usklicnik}')