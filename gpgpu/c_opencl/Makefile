OS := $(shell uname)

ifeq ($(OS),Darwin)
  LIBS=-framework OpenCL
else
  LIBS=-lOpenCL
endif

target/multiply_arrays: multiply_arrays.c
	mkdir -p target
	cc -std=c99 $< $(LIBS) -g -o $@
