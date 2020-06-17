INC="./inc"
FLAGS=-I$(INC)
CC=nvcc

all: serialNeuralNet cudaNeuralNet MPINeuralNet run clean

serialNeuralNet: serialNeuralNet.cu
	$(CC) serialNeuralNet.cu -o serialNeuralNet.out

cudaNeuralNet: cudaNeuralNet.cu
	$(CC) cudaNeuralNet.cu -o cudaNeuralNet.out

MPINeuralNet: MPINeuralNet.cu
	$(CC) MPINeuralNet.cu -o MPINeuralNet.out

run:
	# srun -N1 ./serialNeuralNet.out
	# srun -N1 ./cudaNeuralNet.out
	# srun -N1 ./MPINeuralNet.out
	./serialNeuralNet.out
	./cudaNeuralNet.out
	./MPINeuralNet.out

clean:
	rm serialNeuralNet.out cudaNeuralNet.out MPINeuralNet.out