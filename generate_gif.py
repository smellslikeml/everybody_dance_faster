import os
import time
import numpy as np
from train import *
from PIL import Image
import tensorflow as tf
from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(source_file_name)

    print(
        "File uploaded to {}.".format(
            destination_blob_name
        )
    )

if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    source_data = './dataset/source/'
    source_imgs = os.listdir(source_data)

    vid_length = 200  # Source video is 100 frames
    results_dir = './results/'
    bucket_name = 'YOUR-BUCKET-NAME'
    destination_blob_name = '{}_{}_output.gif'.format('DANCER-NAME', str(int(time.time() * 10000)))

    # Generate dance images from source
    for f in source_imgs:
        input_img, real_img = load(source_data + f)
        input_img, _ = normalize(input_img, real_img)
        input_img = tf.expand_dims(input_img,0)

        generate_dance(generator, input_img)

    # Aggregate frames into a gif and upload to Google Cloud Storage
    results_fls = sorted(os.listdir(results_dir))
    all_frames = []
    img = Image.open(results_dir + fls[0])
    for f in fls[1:]:
        im = Image.open(results_dir + f)
        all_frames.append(im)
    img.save(results_dir + 'output.gif', save_all=True, append_images=all_frames)
    upload_blob(bucket_name, results_dir + 'output.gif', destination_blob_name)
