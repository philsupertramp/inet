FROM ml-webapp-base:latest

COPY . /usr/src/app

ARG hot_build
RUN if [ -z "$hot_build" ] ; then echo "Downloading weights on demand."; else  python -m src.webapp.preload_weights ; fi

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "src.webapp.wsgi:app"]
