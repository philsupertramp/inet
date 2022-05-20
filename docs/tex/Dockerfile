FROM ubuntu:latest

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install texlive-latex-extra

RUN mkdir -p /usr/app

WORKDIR /usr/app
