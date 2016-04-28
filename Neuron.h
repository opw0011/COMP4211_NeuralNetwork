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
	Neuron(int numOutputs, int index);
	void setOutputVal(double val) { outputVal = val; };
	double getOutputVal(void) const { return outputVal; };
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	static double eta; // range:0 to 1, training rate

private:
	double sumDOW(const Layer &nextLayer) const;
	double outputVal;
	std::vector<Connection> outputWeights;
	int index;
	double gradient;
};

#endif//NEURON_H
