EXE ?= nvcc
FLAGS := -arch=sm_50
FLAGS +=  -Wno-deprecated-gpu-targets
FLAGS += -rdc=true
all: 
	$(EXE) -I ./includes -o exec *.cpp *.cu $(FLAGS)

run:
	$(EXE) -I ./includes -o exec *.cpp *.cu $(FLAGS)
	./exec