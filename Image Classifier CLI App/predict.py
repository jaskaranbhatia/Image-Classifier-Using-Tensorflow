import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def cli_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", action="store")
    parser.add_argument("saved_model", action="store")
    parser.add_argument("--top_k", action="store", default=1,type=int)
    parser.add_argument("--category_names", action="store", default="label_map.json")
    return parser.parse_args()

def predict():
    args = cli_options()
    model = tf.keras.models.load_model("./{}".format(args.saved_model),custom_objects={'KerasLayer':hub.KerasLayer})
    def process_image(image):
        image_size = 224
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        image /= 255
        return image
    def predict(image_path, model, top_k):
        from PIL import Image
        image = Image.open(image_path)
        image_arr = np.asarray(image)
        processed_image = process_image(image_arr)
        processed_image = y = np.expand_dims(processed_image, axis=0)
        ps = model.predict(processed_image)
        prediction = ps[0]
        top_5_idx = np.argsort(prediction)[-5:]
        top_5_values = [prediction[i] for i in top_5_idx]
        probs = top_5_values
        classes = top_5_idx + 1
        return probs,classes
    probs,classes = predict(args.image_path,model,args.top_k)
    probs = probs[::-1]
    classes = classes[::-1]
    import json
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    for i in range(0,args.top_k):
        print(class_names[str(classes[i])] + "->" + str(probs[i]))

if __name__ == "__main__":
    predict()