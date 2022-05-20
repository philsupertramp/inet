# demo webapp
Dockerized web-application built using the [flask-framework](https://flask.palletsprojects.com/en/).

The web application contains models trained during the development process of a model to localize and classify insects in
images. Read more in [About the App](#About the App).

## Requirements:
- docker

## Execution
To build the docker images a two-step build process has been implemented in order to minimize the build time for the actual web-application.
In order to build the images execute
```shell
docker build -f src/webapp/Dockerfile.baseimage -t ml-webapp-base:latest .
docker build -f src/webapp/Dockerfile -t ml-webapp:latest --no-cache .
```
or to run with preloading the model weights
```shell
docker build -f src/webapp/Dockerfile -t ml-webapp:latest --no-cache --build-arg hot_build=1 .
```
**Note:** it is important that you provide `--no-cache` once you change the behaviour of the `hot_build` flag.

`hot_build` downloads weights of reused models, such as VGG16 and tests building the available models in the environment.

To run execute
```shell
docker run -it -p 8000:8000 --rm ml-webapp:latest
```
you can find the webapp deployed at [http://localhost:8000/](https://localhost:8000).

## About the App
The app is currently WIP, hence incomplete see [TODOs](#TODO) for more.

The app is aimed to demonstrate bounding box generation and classification of insects within image files.
The user can select from different example images as well as upload images to the website using a smartphone or
static files stored on the users machine.

The used models can be provided prior to building the container image by placing `.h5` files in the `./model-weights` directory within the root directory of the project.
To use the models in the app extend the `./model-weights/content.json` file accordingly.

## TODO
- [ ] implement classification view
- [ ] add more models
- [ ] implement info about used models
