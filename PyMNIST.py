from random import random
rand_range = lambda r: int(random() * r // 1)


def softmax(x):
    exp_x = [2.7182818284590452353602874713526624977 ** nx for nx in x]
    sum_exp = sum(exp_x) + 1e-10
    return [es / sum_exp for es in exp_x]


def mean_square_error(y, y_hat):
    return [(y[i] - y_hat[i]) ** 2 for i in range(10)]


def forward(model, x):
    return softmax([sum([model[nx * 28 * 28 + i] * x[i] for i in range(28 * 28)]) for nx in range(10)])


def backward(model, src, x, y, lr):
    for nx in range(10):
        diff = (x[nx] - y[nx])
        for i in range(28 * 28):
            model[nx * 28 * 28 + i] -= lr * src[i] * diff
    return sum(mean_square_error(x, y)) / 10


def check_acc(model, acc_x, acc_y, ratio, seq=False):
    success, acc_range = 0, int(len(acc_x) * ratio // 1)
    for i in range(acc_range):
        l_index = i if seq else rand_range(acc_range)
        y = forward(model, acc_x[l_index])
        success += 1 if y.index(max(y)) == acc_y[l_index] else 0
    return success / acc_range


def print_graph(history):
    acc_history, cost_history = [dt[1][0] for dt in history.items()], [dt[1][1] for dt in history.items()]
    get_pixel = (lambda va, vc, y: "@" if va/100*40 // 1 == y else ("#" if vc/max(cost_history)*40 // 1 == y else "."))
    print("@: accuracy, #: cost")
    for y in reversed(range(1, 42)):
        print("".join([get_pixel(da, dc, y) for da, dc in zip(acc_history, cost_history)]))


print("Loading MNIST dataset...")

byte_2_int = lambda byte: int.from_bytes(byte, "big", signed=False)
read_file = lambda fd_name: open("resource/" + fd_name, "rb")

fd_train_image, fd_train_label = read_file("train-images.idx3-ubyte"), read_file("train-labels.idx1-ubyte")
fd_test_image, fd_test_label = read_file("t10k-images.idx3-ubyte"), read_file("t10k-labels.idx1-ubyte")
fd_train_image.read(16), fd_train_label.read(8), fd_test_image.read(16), fd_test_label.read(8)

train_size, test_size = 60000, 10000
train_image = [[byte_2_int(fd_train_image.read(1)) / 255 for _ in range(28 * 28)] for _ in range(train_size)]
train_label = [byte_2_int(fd_train_label.read(1)) for _ in range(train_size)]
test_image = [[byte_2_int(fd_test_image.read(1)) / 255 for _ in range(28 * 28)] for _ in range(test_size)]
test_label = [byte_2_int(fd_test_label.read(1)) for _ in range(test_size)]

test_size = test_size - (test_size // 5)
valid_image, valid_label = test_image[test_size:], test_label[test_size:]
test_image, test_label = test_image[:test_size], test_label[:test_size]

_, showcase_count = print("Complete load MNIST dataset.", "\nEnter showcase count:"), int(input())

print_data = lambda data, index: [print("".join(["#" if data[index][vx * 28 + vy] > 0 else "." for vy in range(28)])) for vx in range(28)]

for seq in range(showcase_count):
    index = rand_range(train_size)
    print(f"Sequence#{seq + 1} Label: {train_label[index]} Index: {index}")
    print_data(train_image, index)

_, total_epoch, _ = print("End showcase. \nEnter epoch count:"), int(input()), print("Start training...")

model = [random() * 0.1 * 2 - 0.1 for _ in range(28 * 28 * 10)]
train_history = dict()
for epoch in range(total_epoch):
    l_cost = 0
    for seq in range(train_size):
        index = rand_range(train_size)
        x, y = forward(model, train_image[index]), [0.0] * 10
        y[train_label[index]] = 1.0
        l_cost += backward(model, train_image[index], x, y, lr=0.002)
        if seq % 500 == 0 and seq != 0:
            acc, m_cost = check_acc(model, test_image, test_label, ratio=.05) * 100, l_cost / 500
            prc, l_cost = int(seq / train_size * 100 // 2), 0
            train_history[min(300, len(train_history))] = acc, m_cost
            print_graph(train_history), print(f"accuracy: {str(acc)}% mean-cost:{m_cost}")
            print(f"epoch: {str(epoch+1)}/{str(total_epoch)}[{''.join(['>']*prc)}{''.join(['.']*(50-prc))}]")

print(f"Complete, validation accuracy:{str(check_acc(model, valid_image, valid_label, ratio=1, seq=True) * 100)}%")
_, showcase_count = print("Enter showcase count:"), int(input())

for seq in range(showcase_count):
    index = rand_range(len(valid_label))
    x = forward(model, valid_image[index])
    print(f"Sequence#{seq + 1} Hit:{x.index(max(x)) == valid_label[index]} Prediction:{x.index(max(x))} Label:{valid_label[index]}")
    print_data(valid_image, index)
