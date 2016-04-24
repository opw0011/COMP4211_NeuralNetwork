all: pa

pa: main.o TrainingData.o Net.o
	g++ main.o TrainingData.o Net.o -o pa

main.o: main.cpp
	g++ -c main.cpp

TrainingData.o: TrainingData.cpp
	g++ -c TrainingData.cpp

Net.o: Net.cpp
	g++ -c Net.cpp

clean:
	rm *o pa
