ifndef TTG_DIR
$(error TTG_DIR variable is not defined, aborting build)
endif

DEBUG ?= 0

CC         = mpic++
cc         = mpicc

CFLAGS     = -std=c++17 -D_GNU_SOURCE
LDFLAGS    = -m64 -std=c++11 -Wall -D_GNU_SOURCE

CFLAGS_JDF     = -D_GNU_SOURCE

ifeq ($(strip $(DEBUG)),0)
CFLAGS		 += -O3 -march=native -DNDEBUG
LDFLAGS		 += -O3
else
CFLAGS		 += -g -O0
LDFLAGS		 += -g -O0
endif

PARSEC_DIR = $(HOME)/opt/parsec-ttg-task-benchmarks/
# Include directories
INC        = -I$(TTG_DIR)/include -I$(PWD)/../core -I$(PARSEC_DIR)/include
LIB        = -L$(TTG_DIR)/lib -L$(PWD)/../core -L$(PARSEC_DIR)/lib -lparsec -lcore -lttg -latomic

CFLAGS += $(INC)

include ../core/make_blas.mk

TARGET = main
all: $(TARGET)

.PRECIOUS: %.cc %.o

main.o: main.cc ../core/timer.h
	$(CC) -c $(CFLAGS) $<

main: main.o
	$(CC) $^ $(LIB) $(LDFLAGS) -o $@ 

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
