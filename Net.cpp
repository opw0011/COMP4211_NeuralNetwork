#include "Net.h"
#include <math.h>

using namespace std;

double Net::getError(void) const {
    return m_error;
}

// double Net::sigmoid(double x) const{
// 	double exp_val;
// 	exp_val = exp((double) -x);
// 	return 1 / (1 + exp_val);
// }

Net::Net(const std::vector<unsigned> &topology, const double eta)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// we have made a new layer, now fill it with neurons and 
		// add a bias neuron to the layer:
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "Made a Neuron!" << std::endl;
		}
		// force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const std::vector<double> &inputVals) 
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	// assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// calc overall net error (rms of output neuron errors)

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	// 1/2 sum_k=outputUnits(tk- ok)^2

	// for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
	// 	double delta = targetVals[n] - outputLayer[n].getOutputVal();
	// 	m_error += delta * delta;
	// }
	// m_error /= outputLayer.size() - 1;
	// m_error = sqrt(m_error);  // RMS
	for (int i = 0; i < outputLayer.size() -1; i++) {
		double delta = targetVals[i] - outputLayer[i].getOutputVal();
		m_error += 0.5 * delta * delta;
	}

	// implement a recent avg measurement
	// m_recentAverageError = 
	// 		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
	// 		/ (m_recentAverageSmoothingFactor + 1.0);

	// calc output layer gradients

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// calc gradients on hidden layers

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// for all layers from outputs to first hidden layer,
	// update connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(std::vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
