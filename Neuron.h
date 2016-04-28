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
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	int m_myIndex;
	double m_gradient;
};

#endif//NEURON_H
