# PyMNIST

MNIST-handwritten-classification problem-solving code using Softmax regression in only 100 lines of pure Python3 code without any other library. All code is in file `PyMNIST.py`, and includes a very simple graph output feature.

![printhistory](https://i.imgur.com/k4OKic8.png)

## Getting Started

Download and run `setup.py` or paste the code into the Python3 REPL and the `setup.py` script will automatically download and decompress both the `mnist.py` code and the MNIST data set.

```shell
wget https://raw.githubusercontent.com/junghyun397/PyMNIST/master/setup.py
python3 setup.py
```

After running `setup.py`, you can run `mnist.py` to start learning. Highly recommended using ``pypy`` as pure python runs very slowly. It is fully compatible with pypy and can expect more than 5x learning speed.


```bash
python3 mnist.py  # run with python3
pypy3 mnist.py  # run with pypy3
```

---------------------------

# PyMNIST

다른 라이브러리 없이, 순수한 Python3 코드 100줄로 이루어진 Softmax 회귀를 이용한 MNIST 손글씨 분류 문제 해결 코드입니다. `PyMNIST.py` 파일에서 모든 코드를 확인 해 볼 수 있습니다. 아주 간단한 그래프 출력 기능을 포함하고 있습니다.

## 시작하기

`setup.py` 를 다운로드받아 실행 하거나, Python3 REPL에 코드를 붙여넣으면 `setup.py` 스크립트가 `mnist.py` 코드와 MNIST 데이터 세트 다운로드 및 압축해제를 모두 자동으로 수행합니다.

```shell
wget https://raw.githubusercontent.com/junghyun397/PyMNIST/master/setup.py
python3 setup.py
```

`setup.py` 를 실행한 이후, `mnist.py` 를 실행하여 학습을 시작할 수 있습니다. 실행 속도가 매우 느리므로 pypy 사용을 권장합니다. pypy와 완벽히 호환되며, 5배 이상의 실행 속도를 기대 할 수 있습니다.


```shell
python3 mnist.py  # Python3 로 실행
pypy3 mnist.py  # pypy3 로 실행
```
