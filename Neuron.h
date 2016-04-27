#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <time.h>

class Neuron;
typedef std::vector<Neuron> Layer;
struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(int numOutputs, int myIndex);
	void setOutputVal(double val) { m_outputVal = val; };
	double getOutputVal(void) const { return m_outputVal; };
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta; // [0..1] overall net training rate
	// static double alpha; // [0..n] multiplier of last weight change (momentum)
	// static double transferFunction(double x);
	// static double transferFunctionDerivative(double x);
	// static double randomWeight(void) {  double r = rand() / double(RAND_MAX)/2; std::cout << r << std::endl; return r; }
	// sigmoid function
	static double sigmoid(double x);
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	int m_myIndex;
	double m_gradient;
};

#endif//NEURON_H
