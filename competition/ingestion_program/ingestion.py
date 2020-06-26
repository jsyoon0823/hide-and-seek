#!/usr/bin/env python3

import argparse
import docker
import numpy as np
import os
import sys

from data.data_utils import data_division
from data.data_preprocess import data_preprocess


MAX_SEQ_LEN = 100
SEED = 0
TRAIN_RATE = 0.8
DEFAULT_IMAGE = "tavianator/hide-and-seek-codalab"


def _load_data(data_dir):
    file_name = os.path.join(data_dir, 'train_longitudinal_data.csv')
    ori_data = data_preprocess(file_name, MAX_SEQ_LEN)

    divided_data, _ = data_division(ori_data, seed = SEED, divide_rates = [TRAIN_RATE, 1-TRAIN_RATE])

    train_data = np.asarray(divided_data[0])
    test_data = np.asarray(divided_data[1])

    return train_data, test_data


def _run_hider(args):
    from hider import hider

    print("Loading data...")
    train_data, test_data = _load_data(args.data_dir)

    print("Running hider...")
    generated_data = hider(train_data)
    print("Hider done")

    enlarge_data = np.concatenate((train_data, test_data), axis = 0)
    enlarge_data_label = np.concatenate((np.ones([train_data.shape[0],]), np.zeros([test_data.shape[0],])), axis = 0)

    np.savez(
        os.path.join(args.output_dir, "hider_output.npz"),
        train_data=train_data,
        test_data=test_data,
        generated_data=generated_data,
        enlarge_data=enlarge_data,
        enlarge_data_label=enlarge_data_label,
    )
    print("Saved hider output")


def _run_seeker(args):
    from seeker import seeker

    print("Loading data...")
    with np.load(os.path.join(args.data_dir, "hider_output.npz")) as data:
        train_data = data['train_data']
        test_data = data['test_data']
        generated_data = data['generated_data']
        enlarge_data = data['enlarge_data']
        enlarge_data_label = data['enlarge_data_label']

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    print("Running seeker...")
    reidentified_data = seeker(generated_data, enlarge_data)
    print("Seeker done")

    np.savez(
        os.path.join(args.output_dir, "seeker_output.npz"),
        train_data=train_data,
        test_data=test_data,
        generated_data=generated_data,
        enlarge_data=enlarge_data,
        enlarge_data_label=enlarge_data_label,
        reidentified_data=reidentified_data,
    )
    print("Saved seeker output")


def _docker_pull(client, image):
    split = image.split(":")
    if len(split) > 1:
        image, tag = split
    else:
        tag = "latest"
    client.images.pull(image, tag)


def _dockerize_submission(args, client, image, runtime):
    submission_dir = os.path.abspath(args.submission_dir)
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    self_dir = os.path.dirname(os.path.abspath(__file__))

    volumes = {
        submission_dir: {
            "bind": "/usr/src/submission",
            "mode": "ro",
        },
        data_dir: {
            "bind": "/usr/share/data",
            "mode": "ro",
        },
        output_dir: {
            "bind": "/usr/share/output",
            "mode": "rw",
        },
        self_dir: {
            "bind": "/usr/bin/ingestion",
            "mode": "ro",
        },
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
    }

    command = [
        "python3",
        "/usr/bin/ingestion/ingestion.py",
        "--in-docker",
        "/usr/src/submission",
        "/usr/share/data",
        "/usr/share/output",
    ]

    _docker_pull(client, image)
    return client.containers.run(
        image,
        command,
        runtime=runtime,
        detach=True,
        network="hide-and-seek",
        volumes=volumes,
        environment=environment,
    )


def _dockerize(args):
    client = docker.from_env()

    info = client.info()
    if "nvidia" in info["Runtimes"]:
        runtime = "nvidia"
    else:
        runtime = info["DefaultRuntime"]

    # If it doesn't exist already, create a docker network with no internet access
    try:
        client.networks.create("hide-and-seek", internal=True, check_duplicate=True)
    except docker.errors.APIError as e:
        # HTTP 409: Conflict, aka the network already existed
        if e.status_code != 409:
            raise

    image_path = os.path.join(args.submission_dir, "Dockerimage")
    if os.path.exists(image_path):
        with open(image_path, "r") as f:
            image = f.read().strip()
    else:
        image = DEFAULT_IMAGE

    print("Loading {}...".format(image))
    container = _dockerize_submission(args, client, image, runtime)

    try:
        print("Running {}...".format(image))

        if args.verbose:
            for log in container.logs(stream=True, follow=True):
                sys.stdout.buffer.write(log)

        container.wait()
    finally:
        container.stop()
        container.remove(force=True)

    print("Done")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a submission.")
    parser.add_argument("--in-docker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("submission_dir")
    parser.add_argument("data_dir")
    parser.add_argument("output_dir")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.submission_dir):
        parser.error("Submission directory {} does not exist".format(args.submission_dir))

    if not os.path.isdir(args.data_dir):
        parser.error("Data directory {} does not exist".format(args.data_dir))

    if not os.path.isdir(args.output_dir):
        parser.error("Output directory {} does not exist".format(args.output_dir))

    submission_metadata = os.path.join(args.submission_dir, "metadata")
    if not os.path.isfile(submission_metadata):
        parser.error("Metadata file {} does not exist".format(submission_metadata))

    is_hider = os.path.isfile(os.path.join(args.submission_dir, "hider.py")) or os.path.isdir(os.path.join(args.submission_dir, "hider"))
    is_seeker = os.path.isfile(os.path.join(args.submission_dir, "seeker.py")) or os.path.isdir(os.path.join(args.submission_dir, "seeker"))
    if is_hider and is_seeker:
        parser.error("Submission cannot be both a hider and a seeker")
    elif not (is_hider or is_seeker):
        parser.error("Either a hider.py or seeker.py module must be present")

    if args.in_docker:
        args.submission_dir = os.path.abspath(args.submission_dir)
        args.data_dir = os.path.abspath(args.data_dir)
        args.output_dir = os.path.abspath(args.output_dir)

        os.chdir(args.submission_dir)
        sys.path = [args.submission_dir] + sys.path

        if is_hider:
            _run_hider(args)
        else:
            _run_seeker(args)
    else:
        _dockerize(args)


if __name__ == "__main__":
    main()
