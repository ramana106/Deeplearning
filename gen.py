import os

path = os.path.abspath('./Fashion_mnist')
lis = os.listdir(path)
lis = sorted(lis)
print "Classes:"
print lis
st = ''
for i, n in enumerate(lis):
    for m in os.listdir(os.path.join(path, n)):
        st += os.path.join(path, n, m)
        st += ' '+str(i)+'\n'
fd = open('data.txt', 'w')
fd.write(st)
fd.close()
