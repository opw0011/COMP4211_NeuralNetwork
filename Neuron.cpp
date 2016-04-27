#include "Neuron.h"

double Neuron::eta = 0.5; // overall net learning rate
// double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight

Neuron::Neuron(int numOutputs, int myIndex)
{
	// srand (time(0));

	// initial random weight to each neurons
	for (int c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		double r = rand() / double(RAND_MAX) ;
		// std::cout << r << std::endl;
		m_outputWeights.back().weight = r;
	}

	m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sumOutput = 0.0;

	// sum the previous layer's outputs (which are our inputs)
	// include the bias node from previous layer

	for (int n = 0; n < prevLayer.size(); ++n) {
		sumOutput += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	// sigmoid function
	m_outputVal = 1.0 / (1.0 + exp(-sumOutput));
}

// double Neuron::transferFunction(double x)
// {
// 	// tanh - output range [-1...1]
// 	// return tanh(x);
// 	// sigmoid
// 	return sigmoid(x);
// }

// double Neuron::transferFunctionDerivative(double x) 
// {
// 	// tanh derivative
// 	// return 1.0 - x * x;
// 	// sigmoid derivative function
// 	// return (sigmoid(x)*(1 - sigmoid(x)));
// 	return x*(1.0 - x);
// }

// double Neuron::sigmoid(double x) {
// 	return 1.0 / (1.0 + exp(-x));
// }

void Neuron::calcOutputGradients(double targetVal)
{
	// double delta = targetVal - m_outputVal;
	// m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
	m_gradient = m_outputVal * (1.0 - m_outputVal) * (targetVal - m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	// m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
	m_gradient = m_outputVal * (1.0 - m_outputVal) * dow;
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// sum our contributions of the errors at the nodes we feed

	for (int n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	// the weights to be updated are in the Connection container
	// in the neurons in the preceeding layer

	for (int n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
				// individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				* m_gradient;
				// also add momentum = a fraction of the previous delta weight
				// + alpha
				// * oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
