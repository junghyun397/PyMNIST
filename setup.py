import os, shutil, urllib.request, gzip, platform

directory, main_url = "./resource/", "http://yann.lecun.com/exdb/mnist/"
py_mnist_url = "http://github.com/junghyun397/PyMNIST/blob/master/PyMNIST.py?raw=true"
p_py_mnist = "mnist.py"

p_train_image = "train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"
p_train_label = "train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"
p_test_image = "t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"
p_test_label = "t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte"
path_list = (p_train_image, p_train_label, p_test_image, p_test_label)

if not os.path.exists(directory):
    os.makedirs(directory)

print("===Start download PyMNIST script =========================")
urllib.request.urlretrieve(py_mnist_url, p_py_mnist)
print("===Success download PyMNIST script =======================")

for path in path_list:
    print("===Start download file:", path[0], "".join(["=" for _ in range(33 - len(path[0]))]))
    urllib.request.urlretrieve(main_url + path[0], directory + path[0])
    print("Success download file:", path[0])

    print("Start extract gzip file:", path[1])
    fc, nf = gzip.open(directory + path[0], "rb"), open(directory + path[1], "wb")
    shutil.copyfileobj(fc, nf); nf.close(), fc.close(); os.remove(directory + path[0])
    print("===Success extract gzip file:", path[1], "".join(["=" for _ in range(27 - len(path[1]))]))

print("===Success download all files and script! ================")
if platform.system() == "Windows":
    print("Run", p_py_mnist, "to continue....")
else:
    _, inp = print("Press ENTER to automatic run ", p_py_mnist, "or type pypy to run with pypy3"), input()
    os.system("clear"); os.system("python3 " + p_py_mnist) \
        if inp != "pypy" else os.system("clear"); os.system("pypy3 " + p_py_mnist)
