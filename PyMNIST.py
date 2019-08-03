def softmax(x):
    exp_x = [2.7182818284590452353602874713526624977 ** nx for nx in x]
    sum_exp = sum(exp_x) + 1e-10
    return [es / sum_exp for es in exp_x]


from random import random
rand_range = (lambda w: int(random() * w // 1))

weight = [random() * 0.2 * 2 - 0.2 for _ in range(28 * 28 * 10)]
bias = [random() * 0.01 for _ in range(10)]


def forward(x):
    return softmax([sum([weight[nx * 28 * 28 + i] * x[i] for i in range(28 * 28)]) for nx in range(10)])


def mean_square_error(y, y_hat):
    return sum([(y[i] - y_hat[i]) ** 2 for i in range(10)]) / 10


def backward(src, x, y, lr):
    for nx in range(10):
        diff = (x[nx] - y[nx])
        for i in range(28 * 28):
            weight[nx * 28 * 28 + i] -= lr * src[i] * diff
    return mean_square_error(x, y)


def check_acc(acc_x, acc_y, ratio, seq=False):
    success, acc_range = 0, int(len(acc_x) * ratio // 1)
    for i in range(acc_range):
        l_index = i if seq else rand_range(acc_range)
        y = forward(acc_x[l_index])
        success += 1 if y.index(max(y)) == acc_y[l_index] else 0
    return success / acc_range


print("===Start Loading MNIST DataSet...===")

byte_2_int = (lambda byte: int.from_bytes(byte, "big", signed=False))
read_file = (lambda fd_name: open("resource/" + fd_name, "rb"))

fd_train_image, fd_train_label = read_file("train-images.idx3-ubyte"), read_file("train-labels.idx1-ubyte")
fd_test_image, fd_test_label = read_file("t10k-images.idx3-ubyte"), read_file("t10k-labels.idx1-ubyte")
fd_train_image.read(16); fd_train_label.read(8); fd_test_image.read(16); fd_test_label.read(8)

train_size, test_size = 60000, 10000
train_image = [[byte_2_int(fd_train_image.read(1)) / 255 for _ in range(28 * 28)] for _ in range(train_size)]
train_label = [byte_2_int(fd_train_label.read(1)) for _ in range(train_size)]
test_image = [[byte_2_int(fd_test_image.read(1)) / 255 for _ in range(28 * 28)] for _ in range(test_size)]
test_label = [byte_2_int(fd_test_label.read(1)) for _ in range(test_size)]

test_size = test_size - (test_size // 5)
valid_image, valid_label = test_image[test_size:], test_label[test_size:]
test_image, test_label = test_image[:test_size], test_label[:test_size]

_, vis_count = print("===Success Loading MNIST DataSet!===", "\nEnter visualise case count:"), int(input())

for seq in range(vis_count):
    index = rand_range(train_size)
    print("Label:", train_label[index], "Index:", index, "Sequence:", seq + 1)
    for j in range(28):
        print("".join(["#" if train_image[index][j * 28 + f] > 0 else "." for f in range(28)]))

_, total_epoch = print("===End Visualize data set===========", "\nEnter epoch count:"), int(input())
print("===Start Training...================")

for epoch in range(total_epoch):
    l_cost = 0
    for seq in range(train_size):
        index = rand_range(train_size)
        x, y = forward(train_image[index]), [0.0] * 10
        y[train_label[index]] = 1.0
        l_cost += backward(train_image[index], x, y, lr=0.002)
        if seq % 500 == 0 and seq != 0:
            print("acc:", str(check_acc(test_image, test_label, ratio=.05) * 100) + "%", "mean-cost:", l_cost / 1000)
            prc, l_cost = int(seq / train_size * 100 // 2), 0
            print("epoch:", str(epoch+1)+"/"+str(total_epoch), "["+"".join([">"]*prc)+"".join(["."]*(50-prc))+"]")

print("===End, Validation acc:", str(check_acc(valid_image, valid_label, ratio=1, seq=True) * 100) + "%=======")
_, vis_count = print("Enter Visualise case count:"), int(input())

for seq in range(vis_count):
    index = rand_range(len(valid_label))
    x = forward(valid_image[index])
    print("Prediction:", x.index(max(x)), "Label:", valid_label[index])
    for j in range(28):
        print("".join(["#" if valid_image[index][j * 28 + f] > 0 else "." for f in range(28)]))
