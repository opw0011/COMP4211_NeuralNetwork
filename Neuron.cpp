#include "Neuron.h"

double Neuron::eta = 0;

Neuron::Neuron(int numOutputs, int inputIndex)
{
	// srand (time(0));

	// initial random weight to each neurons
	for (int c = 0; c < numOutputs; ++c) {
		outputWeights.push_back(Connection());
		double r = rand() / double(RAND_MAX) ;
		// std::cout << r << std::endl;
		outputWeights.back().weight = r;
	}

	index = inputIndex;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sumOutput = 0.0;

	// NET : sum of output of previous layer
	for (int n = 0; n < prevLayer.size(); ++n) {
		sumOutput += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[index].weight;
	}
	// sigmoid function
	outputVal = 1.0 / (1.0 + exp(-sumOutput));
}

void Neuron::calcOutputGradients(double targetVal)
{
	gradient = outputVal * (1.0 - outputVal) * (targetVal - outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = outputVal * (1.0 - outputVal) * dow;
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (int n = 0; n < nextLayer.size() - 1; ++n) {
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for (int n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * gradient;


		neuron.outputWeights[index].deltaWeight = newDeltaWeight;
		neuron.outputWeights[index].weight += newDeltaWeight;
	}
}
