import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--data', '-d', help='extracted data directory path')
args = parser.parse_args()
if not args.data:
    sys.exit('Enter data directory')

for data in os.listdir(args.data):
    if not os.path.isdir(os.path.join(args.data, data)):
        continue
    path = os.path.abspath(os.path.join(args.data, data))
    lis = os.listdir(path)
    lis = sorted(lis)
    print "Classes:"
    print lis
    st = ''
    for i, n in enumerate(lis):
        for m in os.listdir(os.path.join(path, n)):
            st += os.path.join(path, n, m)
            st += ' '+str(i)+'\n'
    fd = open(os.path.join(args.data, data+'.txt'), 'w')
    fd.write(st)
    fd.close()
