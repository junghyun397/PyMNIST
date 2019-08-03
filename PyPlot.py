from typing import List

default_height = 50

graph_num = [0]


def shift(target: List[float],
          shift_size: float = None) -> List[float]:
    if shift_size is None:
        shift_size = max(target) / default_height
    return [dt + shift_size for dt in target]


def draw_plot(target: List[float],
              pixel_height: int = default_height,
              graph_name: str = str(graph_num) + " Graph") -> None:
    target = shift(target, max(target) / pixel_height)
    graph_num[0] += 1
    array_range = max(target)

    print(graph_name)
    for y in reversed(range(0, pixel_height + 1)):
        print("".join(["#" if dt / array_range * pixel_height // 1 == y else "." for dt in target]))


def parse_array(text: str) -> List[float]:
    return [float(dt) if dt is not "" else 0.0 for dt in text.split(",")]


while True:
    print(graph_num, "Input float array: ")
    draw_plot(parse_array(input()))
