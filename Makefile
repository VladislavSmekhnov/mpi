CXX = mpicxx
CPP = mpic++
RUN_CMD = mpirun

# Количество процессов указывать здесь:
NUM_THREADS = 3

TASK_SRC = $(shell find . -name "*.cc")

# только, если не запускаются несколько потоков:
# CPUS = --use-hwthread-cpus

TARGET_BIN = target_bin

ifeq ($(TASK_NUM),10)
	NUM_THREADS = 2
endif

all: build run

build:
	$(CPP) $(TASK_SRC) -o $(TARGET_BIN)

run: build
	$(RUN_CMD) -np $(NUM_THREADS) ./$(TARGET_BIN)

clean:
	rm $(TARGET_BIN)

.PHONY: build run all clean