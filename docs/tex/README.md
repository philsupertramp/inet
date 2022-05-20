Thesis content, can be build using the provided dockerfile.

## Workflow:

1. Build image:
```shell
docker build -t latex-builder:latest .
```
2. Run image locally, don't forget to mount the local directory
```shell

docker run --rm -v ${PWD}:/usr/app -w /usr/app/src latex-builder:latest pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -recorder -output-directory="/usr/app/build" "/usr/app/src/BA.tex"

```
3. find outputs generated in `build/`
