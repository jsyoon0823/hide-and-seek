#!/usr/bin/env python3

import argparse
import docker
import os
import sys


MAX_SEQ_LEN = 100
DEFAULT_IMAGE = "tavianator/hide-and-seek-codalab"


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
    self_file = os.path.abspath(__file__)

    volumes = {
        submission_dir: {
            "bind": "/usr/src/submission",
            "mode": "ro",
        },
        data_dir: {
            "bind": "/usr/share/data",
            "mode": "ro",
        },
        self_file: {
            "bind": "/usr/bin/ingestion.py",
            "mode": "ro",
        },
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
    }

    command = [
        "python3",
        "/usr/bin/ingestion.py",
        "--in-docker",
        "/usr/src/submission",
        "/usr/share/data",
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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.submission_dir):
        parser.error("Submission directory {} does not exist".format(args.submission_dir))

    if not os.path.isdir(args.data_dir):
        parser.error("Data directory {} does not exist".format(args.data_dir))

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
        submission_dir = os.path.abspath(args.submission_dir)
        data_dir = os.path.abspath(args.data_dir)

        os.chdir(submission_dir)
        sys.path = [submission_dir] + sys.path
        if is_hider:
            import hider
        else:
            import seeker
    else:
        _dockerize(args)

if __name__ == "__main__":
    main()
