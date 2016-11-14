api_key = u'08a4c9520038b3e91424eae6e62b8a4a'
api_secret = u'a219f8055bf83fcc'

import flickrapi

from flask import Flask
from flask import request
import json
from PIL import Image
from io import BytesIO
import numpy as np
from ann_benchmark_eval import evaluate_cifar_10_image

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'Hello!'


@app.route('/find_images', methods=['GET'])
def find_images():
    page_index = 1
    images_per_page = 10
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
    sets = flickr.photos.search(text=request.args.get('search_string'), page=str(page_index),
                                per_page=str(images_per_page))

    totalImages = int(sets['photos']['total'])

    # https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}.jpg
    url_format_1 = "https://farm{}.staticflickr.com/{}/{}_{}.jpg"
    # https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}_[mstzb].jpg
    url_format_2 = "https://farm{}.staticflickr.com/{}/{}_{}_{}.jpg"
    # https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{o-secret}_o.(jpg|gif|png)
    url_format_3 = "https://farm{}.staticflickr.com/{}/{}_{}_o.{}"

    photos = sets['photos']['photo']

    image_links = []

    for i in range(images_per_page):
        photo = photos[i]
        farm_id = photo['farm']
        server_id = photo['server']
        image_id = photo['id']
        secret = photo['secret']

        # Using url format 1 to format image url
        image_url = url_format_1.format(farm_id, server_id, image_id, secret)
        image_links.append(image_url)

    return json.dumps({'images_list': image_links})


@app.route('/classify_image', methods=['POST'])
def classify():
    file = request.files['image_file']
    print(request.files)
    if not file:
        return json.dumps({"classified_image": "No file"})

    file_contents = file.stream.read()

    img = Image.open(BytesIO(file_contents))
    img.save('image.jpg')
    num_img = np.asarray(img)

    num_img = num_img[:, :, :3]

    im = Image.fromarray(num_img)
    im.save("image.jpeg")

    return json.dumps({"classified_image": evaluate_cifar_10_image(num_img)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
