#include "Net.h"
#include <math.h>

using namespace std;

double Net::getError(void) const {
    return std_error;
}

Net::Net(const std::vector<unsigned> &topology, const double eta)
{
	// set learning rate
	Neuron::eta = eta;

	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {

		// make a new layer
		layers.push_back(Layer());

		unsigned numOutputs = layerNum;
		if(numOutputs == topology.size() - 1)
			numOutputs = 0;
		else
			numOutputs = topology[layerNum + 1];

		// create a bias neuron to the layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
			layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "Neuron made." << std::endl;
		}

		// bias node's assign output value to 1.0
		layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const std::vector<double> &inputVals) 
{
	assert(inputVals.size() == layers[0].size() - 1);

	// assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		layers[0][i].setOutputVal(inputVals[i]);
	}

	// forward propagate
	for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++) {
		Layer &prevLayer = layers[layerNum - 1];
		for (unsigned n = 0; n < layers[layerNum].size() - 1; n++) {
			layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	Layer &outputLayer = layers.back();
	std_error = 0.0;

	// calculate the standard squared error
	for (int i = 0; i < outputLayer.size() -1; i++) {
		double delta = targetVals[i] - outputLayer[i].getOutputVal();
		std_error += 0.5 * delta * delta;
	}

	// calculate output layer gradients

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// calculate hidden latyer gradients

	for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// update the weight for all layers starting from the 1st hidden layer to output layer

	for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(std::vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		resultVals.push_back(layers.back()[n].getOutputVal());
	}
}
