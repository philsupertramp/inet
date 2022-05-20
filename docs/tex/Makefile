BUILD_DIR       =       $(shell pwd)

image:
	docker build -t latex-builder:latest .

build:image
	mkdir -p build
	docker run --rm -v ${BUILD_DIR}:/usr/app -w /usr/app/src latex-builder:latest pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -recorder -output-directory="/usr/app/build" "/usr/app/src/BA.tex"

clean:
	rm -rf build/*
