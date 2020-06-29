#!/usr/bin/env python3

import argparse
import docker
import numpy as np
import os
import shutil
import sys

from data.data_utils import data_division
from data.data_preprocess import data_preprocess
from metrics.metric_utils import reidentify_score


MAX_SEQ_LEN = 100
SEED = 0
TRAIN_RATE = 0.8
DEFAULT_IMAGE = "tavianator/hide-and-seek-codalab"


def _load_data(path):
    ori_data = data_preprocess(path, MAX_SEQ_LEN)

    divided_data, _ = data_division(ori_data, seed = SEED, divide_rates = [TRAIN_RATE, 1-TRAIN_RATE])

    train_data = np.asarray(divided_data[0])
    test_data = np.asarray(divided_data[1])

    return train_data, test_data


def _run_hider(args):
    from hider import hider

    print("Loading data...")
    data_path = os.path.join(args.opt_dir, "data", "train_longitudinal_data.csv")
    train_data, test_data = _load_data(data_path)

    print("Running hider...")
    generated_data = hider(train_data)
    print("Hider done")

    enlarge_data = np.concatenate((train_data, test_data), axis = 0)
    enlarge_data_label = np.concatenate((np.ones([train_data.shape[0],]), np.zeros([test_data.shape[0],])), axis = 0)

    np.savez(
        os.path.join(args.opt_dir, "hiders", args.user, "data.npz"),
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
    data_path = os.path.join(args.opt_dir, "hiders", args.vs, "data.npz")
    with np.load(data_path) as data:
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
        os.path.join(args.opt_dir, "seekers", args.user, "vs", args.vs, "data.npz"),
        train_data=train_data,
        test_data=test_data,
        generated_data=generated_data,
        enlarge_data=enlarge_data,
        enlarge_data_label=enlarge_data_label,
        reidentified_data=reidentified_data,
    )
    print("Saved seeker output")


def _run(parser, args):
    code_dir = os.path.join(args.input_dir, "res")
    os.chdir(code_dir)
    sys.path = [code_dir] + sys.path

    if args.hider and args.seeker:
        parser.error("Can't be both a hider and a seeker")
    elif args.hider:
        _run_hider(args)
    elif args.seeker:
        _run_seeker(args)
    else:
        parser.error("Must be either a hider or a seeker")


def _docker_image(code_dir):
    image_path = os.path.join(code_dir, "Dockerimage")
    if os.path.exists(image_path):
        with open(image_path, "r") as f:
            return f.read().strip()
    else:
        return DEFAULT_IMAGE


def _docker_pull(client, image):
    split = image.split(":")
    if len(split) > 1:
        image, tag = split
    else:
        tag = "latest"
    client.images.pull(image, tag)


def _dockerize_vs(client, runtime, args, seeker, hider):
    code_dir = os.path.join(args.opt_dir, "seekers", seeker, "res")
    vs_dir = os.path.join(args.opt_dir, "seekers", seeker, "vs", hider)
    self_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(vs_dir, exist_ok=True)

    hider_data = os.path.join(args.opt_dir, "hiders", hider, "data.npz")
    hider_mtime = os.path.getmtime(hider_data)
    score_file = os.path.join(vs_dir, "score.txt")
    if os.path.exists(score_file):
        if hider_mtime < os.path.getmtime(score_file):
            print("Re-using previously computed score for {} vs. {}".format(seeker, hider))
            with open(score_file, "r") as f:
                return float(f.read().strip())
        else:
            os.remove(score_file)

    volumes = {
        args.opt_dir: {
            "bind": "/opt/hide-and-seek",
            "mode": "ro",
        },
        vs_dir: {
            "bind": os.path.join("/opt/hide-and-seek/seekers", seeker, "vs", hider),
            "mode": "rw",
        },
        self_dir: {
            "bind": "/opt/hide-and-seek/scoring",
            "mode": "ro",
        },
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
    }

    command = [
        "python3",
        "/opt/hide-and-seek/scoring/scoring.py",
        "--in-docker",
        "--seeker",
        "--user", seeker,
        "--vs", hider,
        os.path.join("/opt/hide-and-seek/seekers", seeker),
        "/tmp",
        "/opt/hide-and-seek",
    ]

    image = _docker_image(code_dir)

    print("Running {} vs. {} in {}...".format(seeker, hider, image))
    container = client.containers.run(
        image,
        command,
        runtime=runtime,
        detach=True,
        network="hide-and-seek",
        volumes=volumes,
        environment=environment,
    )

    try:
        if not args.quiet:
            for log in container.logs(stream=True, follow=True):
                sys.stdout.buffer.write(log)

        container.wait()
    finally:
        container.stop()

        with open(os.path.join(vs_dir, "stdout"), "wb") as f:
            f.write(container.logs(stdout=True, stderr=False))
        with open(os.path.join(vs_dir, "stderr"), "wb") as f:
            f.write(container.logs(stdout=False, stderr=True))

        container.remove(force=True)

    print("Computing reidentification score...")

    seeker_data = os.path.join(vs_dir, "data.npz")
    with np.load(seeker_data) as data:
        enlarge_data_label = data['enlarge_data_label']
        reidentified_data = data['reidentified_data']

    reidentification_score = reidentify_score(enlarge_data_label, reidentified_data)
    with open(score_file, "w") as f:
        print(reidentification_score.astype(str), file=f)

    return reidentification_score


def _dockerize_hider(client, runtime, args):
    code_dir = os.path.join(args.input_dir, "res")
    hider_dir = os.path.join(args.opt_dir, "hiders", args.user)
    self_dir = os.path.dirname(os.path.abspath(__file__))

    copy_dir = os.path.join(hider_dir, "res")
    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)
    shutil.copytree(code_dir, copy_dir)

    os.makedirs(hider_dir, exist_ok=True)

    volumes = {
        args.opt_dir: {
            "bind": "/opt/hide-and-seek",
            "mode": "ro",
        },
        hider_dir: {
            "bind": os.path.join("/opt/hide-and-seek/hiders", args.user),
            "mode": "rw",
        },
        args.input_dir: {
            "bind": "/opt/hide-and-seek/input",
            "mode": "ro",
        },
        self_dir: {
            "bind": "/opt/hide-and-seek/scoring",
            "mode": "ro",
        },
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
    }

    command = [
        "python3",
        "/opt/hide-and-seek/scoring/scoring.py",
        "--in-docker",
        "--hider",
        "--user", args.user,
        "/opt/hide-and-seek/input",
        "/tmp",
        "/opt/hide-and-seek",
    ]

    image = _docker_image(code_dir)

    print("Pulling {}...".format(image))
    _docker_pull(client, image)

    print("Running {}...".format(image))
    container = client.containers.run(
        image,
        command,
        runtime=runtime,
        detach=True,
        network="hide-and-seek",
        volumes=volumes,
        environment=environment,
    )

    try:
        if not args.quiet:
            for log in container.logs(stream=True, follow=True):
                sys.stdout.buffer.write(log)

        container.wait()
    finally:
        container.stop()

        with open(os.path.join(hider_dir, "stdout"), "wb") as f:
            f.write(container.logs(stdout=True, stderr=False))
        with open(os.path.join(hider_dir, "stderr"), "wb") as f:
            f.write(container.logs(stdout=False, stderr=True))

        container.remove(force=True)

    print("Done generating data")

    scores = []
    for seeker in os.listdir(os.path.join(args.opt_dir, "seekers")):
        scores.append(_dockerize_vs(client, runtime, args, seeker, args.user))
    score = np.mean(scores) if scores else np.float64(0)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print("hider_score: {}".format(score.astype(str)), file=f)
        print("seeker_score: 0", file=f)


def _dockerize_seeker(client, runtime, args):
    code_dir = os.path.join(args.input_dir, "res")

    copy_dir = os.path.join(args.opt_dir, "seekers", args.user, "res")
    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)
    shutil.copytree(code_dir, copy_dir)

    image = _docker_image(code_dir)
    print("Pulling {}...".format(image))
    _docker_pull(client, image)

    scores = []
    for hider in os.listdir(os.path.join(args.opt_dir, "hiders")):
        scores.append(_dockerize_vs(client, runtime, args, args.user, hider))
    score = np.mean(scores) if scores else np.float64(0)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print("hider_score: 0", file=f)
        print("seeker_score: {}".format(score.astype(str)), file=f)


def _dockerize(parser, args):
    code_dir = os.path.join(args.input_dir, "res")

    with open(os.path.join(args.input_dir, "metadata"), "r") as f:
        for line in f:
            if line.startswith("submitted-by: "):
                args.user = line[14:].strip()
                break
        else:
            parser.error("Could not determine submitting user")

    if args.user == "tavianator":
        if os.path.exists(os.path.join(code_dir, "add_noise.py")):
            args.user = "baseline_add_noise"
        elif os.path.exists(os.path.join(code_dir, "timegan")):
            args.user = "baseline_timegan"
        elif os.path.exists(os.path.join(code_dir, "knn_seeker.py")):
            args.user = "baseline_knn"
        elif os.path.exists(os.path.join(code_dir, "binary_predictor")):
            args.user = "baseline_binary_predictor"

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

    is_hider = os.path.isfile(os.path.join(code_dir, "hider.py")) or os.path.isdir(os.path.join(code_dir, "hider"))
    is_seeker = os.path.isfile(os.path.join(code_dir, "seeker.py")) or os.path.isdir(os.path.join(code_dir, "seeker"))
    if is_hider and is_seeker:
        parser.error("Submission cannot be both a hider and a seeker")
    elif is_hider:
        _dockerize_hider(client, runtime, args)
    elif is_seeker:
        _dockerize_seeker(client, runtime, args)
    else:
        parser.error("Either a hider.py or seeker.py module must be present")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a submission.")
    parser.add_argument("--in-docker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--hider", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--seeker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--user", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vs", default=None, help=argparse.SUPPRESS)
    parser.add_argument("-q", "--quiet", action="store_true", default=False)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("opt_dir")
    args = parser.parse_args()

    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.opt_dir = os.path.abspath(args.opt_dir)

    if args.in_docker:
        _run(parser, args)
    else:
        _dockerize(parser, args)


if __name__ == "__main__":
    main()
