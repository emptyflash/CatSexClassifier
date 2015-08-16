import os
import shutil

destfile = "/home/ubuntu/.theanorc"
try:
    open(destfile, 'a').close()
    shutil.copyfile("/mnt/.theanorc", destfile)
except IOError as ex:
    print "Couldn't copy file, IOError occurred " + str(ex)

print "Finished setting up Theano"
