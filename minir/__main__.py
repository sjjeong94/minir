import onnx
import argparse
from minir.onnx import from_onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="minir")
    parser.add_argument("input", type=str)
    args = parser.parse_args()
    model = onnx.load(args.input)
    ir = from_onnx(model)
    print(ir)
