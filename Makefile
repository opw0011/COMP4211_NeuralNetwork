all: pa

pa: main.o TrainingData.o Net.o Neuron.o
	g++ main.o TrainingData.o Net.o Neuron.o -o pa

main.o: main.cpp
	g++ -c main.cpp

TrainingData.o: TrainingData.cpp
	g++ -c TrainingData.cpp

Net.o: Net.cpp
	g++ -c Net.cpp

Neuron.o: Neuron.cpp
	g++ -c Neuron.cpp    

clean:
	rm *o pa
