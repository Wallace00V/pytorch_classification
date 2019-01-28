import os,sys

cat_train = "/Users/zhang/Downloads/catdog_project/cat_train"
dog_train = "/Users/zhang/Downloads/catdog_project/dog_train"

cat_val = "/Users/zhang/Downloads/catdog_project/cat_val"
dog_val = "/Users/zhang/Downloads/catdog_project/dog_val"

with open("/Users/zhang/Downloads/catdog_project/label_train.txt",'w') as f:
    C = os.listdir(cat_train)
    for c in C:
        f.write(os.path.join(cat_train,c)+" "+'0'+"\n")

    D = os.listdir(dog_train)
    for d in D:
        f.write(os.path.join(dog_train,d)+" "+'1'+"\n")

with open("/Users/zhang/Downloads/catdog_project/label_val.txt", 'w') as f:
    C = os.listdir(cat_val)
    for c in C:
        f.write(os.path.join(cat_val, c) + " " + '0' + "\n")

    D = os.listdir(dog_val)
    for d in D:
        f.write(os.path.join(dog_val, d) + " " + '1' + "\n")