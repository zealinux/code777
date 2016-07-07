OS := $(shell uname)
CC = cc
CFLAGS = -g -O2

ifeq ($(OS),Darwin)
  LIBS=-framework OpenCL
else
  LIBS=-lOpenCL
endif

OBJDIR := target
SOURCES := $(wildcard *.c)
OBJECTS := $(SOURCES:%.c=$(OBJDIR)/%.o)

$(OBJDIR)/multiply_arrays: $(OBJECTS)
	$(CC) -std=c99 $^ $(LIBS) $(CFLAGS) -o $@

$(OBJDIR)/%.o: %.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf ./*.o target
