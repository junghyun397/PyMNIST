import os, shutil, requests, gzip

directory, main_url = "./resource/", "http://yann.lecun.com/exdb/mnist/"
py_mnist_url = "http://github.com/junghyun397/PyMNIST/blob/master/PyMNIST.py?raw=true"
p_py_mnist = "PyMNIST.py"

p_train_image = "train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"
p_train_label = "train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"
p_test_image = "t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"
p_test_label = "t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte"
path_list = (p_train_image, p_train_label, p_test_image, p_test_label)

if not os.path.exists(directory):
    os.makedirs(directory)

print("===Start download PyMNIST script =========================")
req, compressed = requests.get(url=py_mnist_url), open(p_py_mnist, "wb")
compressed.write(req.content); compressed.close()
print("===Success download PyMNIST script =======================")

for path in path_list:
    print("===Start download file:", path[0], "".join(["=" for _ in range(33 - len(path[0]))]))
    req, compressed = requests.get(url=main_url + path[0]), open(directory + path[0], "wb")
    compressed.write(req.content); compressed.close()
    print("Success download file:", path[0])

    print("Start extract gzip file:", path[1])
    fc = gzip.open(directory + path[0], "rb")
    nf = open(directory + path[1], "wb")
    shutil.copyfileobj(fc, nf)
    nf.close(), fc.close(); os.remove(directory + path[0])
    print("===Success extract gzip file:", path[1], "".join(["=" for _ in range(27 - len(path[1]))]))

print("Press ENTER to automatic run PyMNIST.py or press Ctrl + C to exit"), input()
os.system("clear"); os.system("python3 " + p_py_mnist)
