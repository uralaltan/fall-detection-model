import numpy as np
import onnxruntime as ort


def main():
    model_path = "fall_detector.onnx"
    session = ort.InferenceSession(model_path)

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_name = input_meta.name
    output_name = output_meta.name
    print(f"Model expects input '{input_name}' and will produce output '{output_name}'")

    dummy_features = np.zeros((1, 24), dtype=np.float32)

    outputs = session.run([output_name], {input_name: dummy_features})
    prediction = outputs[0][0]

    print("Dummy input prediction:", prediction)


if __name__ == "__main__":
    main()
